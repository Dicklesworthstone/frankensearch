//! Prepared, parity-gated six-arm execution for the QG-6 query benchmark.
//!
//! The generic runner deliberately owns the lifecycle boundary: engines are
//! constructed and populated through [`Qg6PreparedExperiment::prepare_with`],
//! validated for exact or explicitly proven semantic result parity, warmed
//! equally, and only then exposed to the timed schedule. This keeps corpus
//! construction, commits, configuration, warmup, and parity checks outside
//! every timed interval.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write as _};
use std::hint::black_box;
use std::ops::Bound;
use std::time::{Duration, Instant};

use frankensearch_quill::{
    BooleanOperator, DEFAULT_SCHEMA, DefaultQueryParser, Occur, Query, QueryValue,
    canonicalize_query,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use thiserror::Error;
use unicode_normalization::UnicodeNormalization;

use crate::perf::{PerfQueryClass, QG6_QUERY_GROUPS};

const QG6_QUERY_MANIFEST_VERSION: &str = "frankensearch-qg6-query-manifest-v3";
const QG6_RESULT_RECEIPT_VERSION: &str = "frankensearch-qg6-result-receipt-v1";
const QG6_RESULT_SEQUENCE_VERSION: &str = "frankensearch-qg6-result-sequence-v1";
const QG6_TIMING_LEAF_RECEIPT_VERSION: &str = "frankensearch-qg6-timing-leaf-receipt-v3";
const QG6_SEMANTIC_CONTRACT_VERSION: &str = "frankensearch-qg6-semantic-contract-v2";
const QG6_SCHEDULE_AUTHORITY_VERSION: &str = "frankensearch-qg6-schedule-authority-v1";
const QG6_QUERY_IDENTITY_VERSION: &str = "frankensearch-qg6-query-identity-v1";
const QG6_QUERY_GENERATOR_REVISION: &str = "frankensearch-qg6-frozen-80-query-generator-v2";
const QG6_CORPUS_GENERATOR_REVISION: &str =
    "frankensearch-quill-gauntlet/generator-v2;schema=2;zipf=s11;vocab=8192;max_doc=4096";
/// GOLDEN-CHANGE (Quill default-field term parity, 50c14df5).
///
/// This anchor hashes each normative query's parsed AST, not just its text.
/// Commit 50c14df5 intentionally changed an unfielded one-token query from one
/// multi-field `Term` into an implicit `Boolean` with `Should` leaves for
/// content and boosted title, matching Tantivy's default-field semantics. All
/// normative ASTs containing a qualifying multi-default-field unfielded term
/// therefore changed, potentially across query classes. The public
/// `public_unfielded_three_term_or_matches_tantivy_score_bits_and_order` test
/// independently witnesses that the parser change is intentional. Query text,
/// sampling, class membership, support state, supported k, and generator
/// revisions did not change.
///
/// CONSEQUENCE: every artifact carrying the previous manifest identity
/// `0d9176a839fc468eb0c3f8a4e427bd2e81f7b2998a0f8974c27f8cc47620b20b`
/// was measured against the pre-50c14df5 query universe and is incomparable to
/// evidence measured here. Re-freezing records that; it does not restore
/// comparability or establish a performance win.
const QG6_FROZEN_MANIFEST_SHA256: &str =
    "6207a48e57714f2acf39f34d0f30e20e1f3eaa209afafaa4b56cb5118ccca748";
const EMPTY_DOCUMENT_ID_SHA256: &str =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
const QG6_AD_HOC_QUERY_GENERATOR_REVISION: &str = "frankensearch-qg6-ad-hoc-query-v1";
const QG6_AD_HOC_CORPUS_GENERATOR_REVISION: &str = "frankensearch-qg6-ad-hoc-corpus-v1";
const QG6_SAMPLING_FRAME: &str = "five public query classes; sixteen independently frozen \
    normalized/AST identities per class; every query receives equal weight; repeated leaves are \
    within-query measurements and never independent queries; wider intervals require more \
    independently frozen queries, never threshold weakening or leaf pseudoreplication";
const QG6_SUPPORTED_K: [usize; 2] = [10, 100];
const QG6_TOTAL_QUERY_COUNT: usize = PerfQueryClass::ALL.len() * QG6_QUERY_GROUPS;
const QG6_REVIEWED_DIVERGENCE_ID: &str = "quill-divergence/qg6-native-tie-and-score-epsilon-v3";
const QG6_REVIEWED_DIVERGENCE_CONTRACT: &str = "rank-exact or reviewed native cutoff tie order / score epsilon 0.0001 caused only by \
     oracle segment geometry; exact total count and live document count remain mandatory";
const MAX_QUERY_COUNT: usize = 4_096;
const MAX_QUERY_ID_BYTES: usize = 256;
const MAX_QUERY_TEXT_BYTES: usize = 16 * 1_024;
const MAX_DOC_ID_BYTES: usize = 4 * 1_024;
const MAX_UNSUPPORTED_REASON_CODE_BYTES: usize = 64;
const MAX_K: usize = 100_000;
/// Canonical number of individually timed searches in every QG-6 parent sample.
pub const QG6_TIMED_SEARCHES_PER_SAMPLE: usize = 128;
const QG6_TIMING_LEAF_MAX_DECIMAL_U64_BYTES: usize = 20;
const QG6_TIMING_LEAF_MAX_PAIR_WIRE_BYTES: usize = QG6_TIMING_LEAF_MAX_DECIMAL_U64_BYTES * 2 + 2;

/// Frozen support state and any reviewed cross-engine result divergence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum Qg6SupportDivergence {
    /// Both engines must be exactly result-identical.
    SupportedExact,
    /// Native tie order or score bits may differ only under this reviewed rule.
    SupportedWithReviewedDivergence {
        /// Stable reviewed-register entry.
        register_id: String,
        /// SHA-256 of the exact reviewed semantic rule.
        contract_sha256: String,
    },
    /// Explicit unsupported syntax. A normative manifest containing this state
    /// fails before preparation rather than silently dropping the query.
    Unsupported {
        /// Stable bounded reason code with no raw query text.
        reason_code: String,
    },
}

impl Qg6SupportDivergence {
    fn reviewed() -> Self {
        Self::SupportedWithReviewedDivergence {
            register_id: QG6_REVIEWED_DIVERGENCE_ID.to_owned(),
            contract_sha256: sha256_hex(QG6_REVIEWED_DIVERGENCE_CONTRACT.as_bytes()),
        }
    }

    /// Whether a non-rank-exact result may enter the reviewed comparator.
    #[must_use]
    pub const fn allows_reviewed_divergence(&self) -> bool {
        matches!(self, Self::SupportedWithReviewedDivergence { .. })
    }

    fn hash_into(&self, hasher: &mut Sha256) {
        match self {
            Self::SupportedExact => hasher.update([0]),
            Self::SupportedWithReviewedDivergence {
                register_id,
                contract_sha256,
            } => {
                hasher.update([1]);
                hash_len_prefixed(hasher, register_id.as_bytes());
                hash_len_prefixed(hasher, contract_sha256.as_bytes());
            }
            Self::Unsupported { reason_code } => {
                hasher.update([2]);
                hash_len_prefixed(hasher, reason_code.as_bytes());
            }
        }
    }
}

fn valid_supported_divergence(value: &Qg6SupportDivergence) -> bool {
    match value {
        Qg6SupportDivergence::SupportedExact => true,
        Qg6SupportDivergence::SupportedWithReviewedDivergence {
            register_id,
            contract_sha256,
        } => {
            !register_id.is_empty()
                && register_id.len() <= 256
                && is_lower_hex_sha256(contract_sha256)
        }
        Qg6SupportDivergence::Unsupported { .. } => false,
    }
}

/// Redacted query identity safe for benchmark logs and evidence diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg6QueryLogIdentity {
    /// Stable non-sensitive query identifier.
    pub query_id: String,
    /// Declared public class.
    pub class: PerfQueryClass,
    /// SHA-256 of normalized source text.
    pub normalized_text_sha256: String,
    /// SHA-256 of the canonical parsed Quill AST.
    pub parsed_ast_sha256: String,
}

/// The six independent logical indexes in the QG-6 admission experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg6ArmRole {
    /// Left side of the Tantivy/Tantivy null comparison.
    TantivyNullLeft,
    /// Right side of the Tantivy/Tantivy null comparison.
    TantivyNullRight,
    /// Left side of the Quill/Quill null comparison.
    QuillNullLeft,
    /// Right side of the Quill/Quill null comparison.
    QuillNullRight,
    /// Tantivy side of the Quill/Tantivy effect comparison.
    EffectControl,
    /// Quill side of the Quill/Tantivy effect comparison.
    EffectTreatment,
}

impl Qg6ArmRole {
    /// Stable order used by preparation, preflight, and lifecycle receipts.
    pub const ALL: [Self; 6] = [
        Self::TantivyNullLeft,
        Self::TantivyNullRight,
        Self::QuillNullLeft,
        Self::QuillNullRight,
        Self::EffectControl,
        Self::EffectTreatment,
    ];

    const fn index(self) -> usize {
        match self {
            Self::TantivyNullLeft => 0,
            Self::TantivyNullRight => 1,
            Self::QuillNullLeft => 2,
            Self::QuillNullRight => 3,
            Self::EffectControl => 4,
            Self::EffectTreatment => 5,
        }
    }
}

/// Which independently measured pair a scheduled block belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg6Comparison {
    /// Tantivy/Tantivy A/A null.
    TantivyNull,
    /// Quill/Quill A/A null.
    QuillNull,
    /// Tantivy/Quill A/B effect.
    Effect,
}

/// First or second position inside one paired timing block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg6SampleOrder {
    /// First arm invoked in the block.
    First,
    /// Second arm invoked in the block.
    Second,
}

const fn qg6_arm_role_tag(role: Qg6ArmRole) -> u8 {
    match role {
        Qg6ArmRole::TantivyNullLeft => 0,
        Qg6ArmRole::TantivyNullRight => 1,
        Qg6ArmRole::QuillNullLeft => 2,
        Qg6ArmRole::QuillNullRight => 3,
        Qg6ArmRole::EffectControl => 4,
        Qg6ArmRole::EffectTreatment => 5,
    }
}

const fn qg6_comparison_tag(comparison: Qg6Comparison) -> u8 {
    match comparison {
        Qg6Comparison::TantivyNull => 0,
        Qg6Comparison::QuillNull => 1,
        Qg6Comparison::Effect => 2,
    }
}

const fn qg6_sample_order_tag(order: Qg6SampleOrder) -> u8 {
    match order {
        Qg6SampleOrder::First => 0,
        Qg6SampleOrder::Second => 1,
    }
}

/// Lifecycle phase attached to a bounded failure diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg6Phase {
    /// Engine construction, population, and commit.
    Prepare,
    /// Exact ordered-result validation before timing.
    Preflight,
    /// Equal untimed warmup.
    Warmup,
    /// Timed measurement.
    Measurement,
    /// Full native result verification after every timed interval is complete.
    Postflight,
}

/// Whether the selected cells constitute the complete QG-6 gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg6SelectionScope {
    /// Every normative QG-6 cell is present.
    CompleteGate,
    /// An explicit fixture filter selected only part of QG-6.
    FilteredPreAdmission,
}

/// Claim state contributed by the QG-6 selection boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg6SelectionClaim {
    /// Selection completeness does not itself forbid gate validation.
    EligibleForGateValidation,
    /// Partial evidence is diagnostic and cannot support a publication claim.
    NoClaim,
}

impl Qg6SelectionScope {
    /// Classify a selected QG-6 slice against the normative cell count.
    ///
    /// # Errors
    ///
    /// Rejects an empty normative gate, an empty selection, or a selected
    /// count greater than the normative matrix.
    pub fn from_cell_counts(selected: usize, normative: usize) -> Result<Self, Qg6HarnessError> {
        if normative == 0 || selected == 0 || selected > normative {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "QG-6 selected/normative cell counts are inconsistent".to_owned(),
            });
        }
        Ok(if selected == normative {
            Self::CompleteGate
        } else {
            Self::FilteredPreAdmission
        })
    }

    /// Convert an explicit selection boundary into its fail-closed claim state.
    #[must_use]
    pub const fn claim(self) -> Qg6SelectionClaim {
        match self {
            Self::CompleteGate => Qg6SelectionClaim::EligibleForGateValidation,
            Self::FilteredPreAdmission => Qg6SelectionClaim::NoClaim,
        }
    }
}

/// One frozen query in the prepared QG-6 workload.
///
/// `Debug` is deliberately redacted. The raw query remains available to the
/// engine adapter and the immutable manifest serializer, but benchmark logs
/// get only [`Self::log_identity`].
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg6QuerySpec {
    id: String,
    text: String,
    class: PerfQueryClass,
    normalized_text_sha256: String,
    parsed_ast_sha256: String,
    coverage_row: u8,
    coverage_column: u8,
    support_divergence: Qg6SupportDivergence,
    supported_k: [usize; 2],
    query_generator_revision: String,
    corpus_generator_revision: String,
}

impl Qg6QuerySpec {
    /// Construct a bounded ad-hoc query for focused harness tests.
    ///
    /// Normative QG-6 execution must use [`Self::normative_for_class`]. The
    /// ad-hoc constructor still derives and validates normalized/AST identity,
    /// so generic prepared-run tests exercise the same fail-closed boundary.
    ///
    /// # Errors
    ///
    /// Rejects malformed IDs, raw query bounds, parser recovery, or a query
    /// shape inconsistent with the stable ID's class prefix.
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Result<Self, Qg6HarnessError> {
        let id = id.into();
        let text = text.into();
        let class = class_from_query_id(&id)?;
        Self::build(
            id,
            text,
            class,
            0,
            0,
            Qg6SupportDivergence::reviewed(),
            QG6_AD_HOC_QUERY_GENERATOR_REVISION,
            QG6_AD_HOC_CORPUS_GENERATOR_REVISION,
        )
    }

    fn build(
        id: String,
        text: String,
        class: PerfQueryClass,
        coverage_row: u8,
        coverage_column: u8,
        support_divergence: Qg6SupportDivergence,
        query_generator_revision: &str,
        corpus_generator_revision: &str,
    ) -> Result<Self, Qg6HarnessError> {
        if id.is_empty() || id.len() > MAX_QUERY_ID_BYTES {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "query ID must be non-empty and at most 256 bytes".to_owned(),
            });
        }
        if text.is_empty() || text.len() > MAX_QUERY_TEXT_BYTES {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "query text must be non-empty and at most 16384 bytes".to_owned(),
            });
        }
        let normalized = normalize_query_text(&text);
        if normalized.is_empty() {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: format!("query {id:?} normalizes to empty text"),
            });
        }
        let (parsed_ast_sha256, parsed) = parsed_ast_sha256(&normalized, &id)?;
        validate_query_shape(class, &parsed.query, &id)?;
        let query = Self {
            id,
            text,
            class,
            normalized_text_sha256: sha256_hex(normalized.as_bytes()),
            parsed_ast_sha256,
            coverage_row,
            coverage_column,
            support_divergence,
            supported_k: QG6_SUPPORTED_K,
            query_generator_revision: query_generator_revision.to_owned(),
            corpus_generator_revision: corpus_generator_revision.to_owned(),
        };
        query.validate_entry()?;
        Ok(query)
    }

    /// Stable, non-sensitive query identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Query text supplied only to the engine adapter.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Frozen public query class.
    #[must_use]
    pub const fn class(&self) -> PerfQueryClass {
        self.class
    }

    /// Neutral row coordinate in the frozen 4x4 coverage grid.
    #[must_use]
    pub const fn coverage_row(&self) -> u8 {
        self.coverage_row
    }

    /// Neutral column coordinate in the frozen 4x4 coverage grid.
    #[must_use]
    pub const fn coverage_column(&self) -> u8 {
        self.coverage_column
    }

    /// Whether the frozen comparator contract permits a reviewed native
    /// cutoff-tie or score-epsilon divergence for this query.
    #[must_use]
    pub const fn allows_reviewed_divergence(&self) -> bool {
        self.support_divergence.allows_reviewed_divergence()
    }

    /// Stable redacted support label for benchmark diagnostics.
    #[must_use]
    pub const fn support_label(&self) -> &'static str {
        match &self.support_divergence {
            Qg6SupportDivergence::SupportedExact => "supported_exact",
            Qg6SupportDivergence::SupportedWithReviewedDivergence { .. } => {
                "supported_reviewed_divergence"
            }
            Qg6SupportDivergence::Unsupported { .. } => "unsupported",
        }
    }

    /// Redacted identity for logs and bounded diagnostics.
    #[must_use]
    pub fn log_identity(&self) -> Qg6QueryLogIdentity {
        Qg6QueryLogIdentity {
            query_id: self.id.clone(),
            class: self.class,
            normalized_text_sha256: self.normalized_text_sha256.clone(),
            parsed_ast_sha256: self.parsed_ast_sha256.clone(),
        }
    }

    /// Domain-separated identity of the redacted query contract.
    #[must_use]
    pub fn identity_sha256(&self) -> String {
        Qg6QueryIdentityReceipt::from_query(self).query_identity_sha256
    }

    /// Sampling frame that governs every normative QG-6 query cell.
    #[must_use]
    pub const fn sampling_frame() -> &'static str {
        QG6_SAMPLING_FRAME
    }

    /// Stable query-generator revision bound into every normative query hash.
    #[must_use]
    pub const fn normative_query_generator_revision() -> &'static str {
        QG6_QUERY_GENERATOR_REVISION
    }

    /// Stable corpus-generator revision bound into every normative query hash.
    #[must_use]
    pub const fn normative_corpus_generator_revision() -> &'static str {
        QG6_CORPUS_GENERATOR_REVISION
    }

    /// Construct the exact sixteen-query slice for one public class.
    ///
    /// This first constructs and validates the complete 80-query manifest, so
    /// a class slice cannot be generated from a different or incomplete global
    /// workload.
    ///
    /// # Errors
    ///
    /// Fails if the built-in frozen manifest violates any count, identity,
    /// parser, coverage-coordinate, support, or revision invariant.
    pub fn normative_for_class(class: PerfQueryClass) -> Result<Vec<Self>, Qg6HarnessError> {
        let manifest = build_normative_query_manifest()?;
        validate_complete_query_manifest(&manifest)?;
        Ok(manifest
            .into_iter()
            .filter(|query| query.class == class)
            .collect())
    }

    /// SHA-256 of the complete order-independent frozen 80-query manifest.
    ///
    /// # Errors
    ///
    /// Fails if the built-in manifest no longer validates.
    pub fn normative_manifest_sha256() -> Result<String, Qg6HarnessError> {
        let manifest = build_normative_query_manifest()?;
        validate_complete_query_manifest(&manifest)?;
        Ok(query_manifest_sha256(&manifest))
    }

    fn validate_entry(&self) -> Result<(), Qg6HarnessError> {
        if self.id.is_empty() || self.id.len() > MAX_QUERY_ID_BYTES {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "query ID must be non-empty and at most 256 bytes".to_owned(),
            });
        }
        if self.text.is_empty() || self.text.len() > MAX_QUERY_TEXT_BYTES {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: format!("query {:?} has invalid raw-text bounds", self.id),
            });
        }
        if class_from_query_id(&self.id)? != self.class {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: format!("query {:?} was reclassified", self.id),
            });
        }
        if self.supported_k != QG6_SUPPORTED_K {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: format!(
                    "query {:?} does not support exactly k=10 and k=100",
                    self.id
                ),
            });
        }
        if self.coverage_row >= 4 || self.coverage_column >= 4 {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: format!("query {:?} has an invalid 4x4 coverage coordinate", self.id),
            });
        }
        if self.query_generator_revision.is_empty()
            || self.query_generator_revision.len() > 256
            || self.corpus_generator_revision.is_empty()
            || self.corpus_generator_revision.len() > 512
        {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: format!(
                    "query {:?} has invalid generator revision identity",
                    self.id
                ),
            });
        }
        match &self.support_divergence {
            Qg6SupportDivergence::SupportedExact => {}
            Qg6SupportDivergence::SupportedWithReviewedDivergence {
                register_id,
                contract_sha256,
            } => {
                if register_id.is_empty()
                    || register_id.len() > 256
                    || !is_lower_hex_sha256(contract_sha256)
                {
                    return Err(Qg6HarnessError::InvalidSpec {
                        reason: format!(
                            "query {:?} has an invalid reviewed divergence binding",
                            self.id
                        ),
                    });
                }
            }
            Qg6SupportDivergence::Unsupported { reason_code } => {
                if reason_code.is_empty()
                    || reason_code.len() > MAX_UNSUPPORTED_REASON_CODE_BYTES
                    || !reason_code.bytes().all(|byte| {
                        byte.is_ascii_lowercase()
                            || byte.is_ascii_digit()
                            || matches!(byte, b'.' | b'_' | b'-')
                    })
                {
                    return Err(Qg6HarnessError::InvalidSpec {
                        reason: format!(
                            "query {:?} has an invalid unsupported-reason token",
                            self.id
                        ),
                    });
                }
                return Err(Qg6HarnessError::InvalidSpec {
                    reason: format!(
                        "query {:?} is explicitly unsupported; reason_sha256={}, reason_bytes={}; \
                         refusing silent skip",
                        self.id,
                        sha256_hex(reason_code.as_bytes()),
                        reason_code.len()
                    ),
                });
            }
        }
        let normalized = normalize_query_text(&self.text);
        let normalized_sha256 = sha256_hex(normalized.as_bytes());
        if normalized_sha256 != self.normalized_text_sha256 {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: format!("query {:?} normalized-text hash drifted", self.id),
            });
        }
        let (parsed_ast_sha256, parsed) = parsed_ast_sha256(&normalized, &self.id)?;
        if parsed_ast_sha256 != self.parsed_ast_sha256 {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: format!("query {:?} parsed-AST hash drifted", self.id),
            });
        }
        validate_query_shape(self.class, &parsed.query, &self.id)
    }
}

impl fmt::Debug for Qg6QuerySpec {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Qg6QuerySpec")
            .field("id", &self.id)
            .field("class", &self.class)
            .field("normalized_text_sha256", &self.normalized_text_sha256)
            .field("parsed_ast_sha256", &self.parsed_ast_sha256)
            .field("coverage_row", &self.coverage_row)
            .field("coverage_column", &self.coverage_column)
            .field("support_divergence", &self.support_divergence)
            .field("supported_k", &self.supported_k)
            .field("query_generator_revision", &self.query_generator_revision)
            .field("corpus_generator_revision", &self.corpus_generator_revision)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy)]
struct Qg6QuerySeed {
    text: &'static str,
}

const fn query_seed(text: &'static str) -> Qg6QuerySeed {
    Qg6QuerySeed { text }
}

const IDENTIFIER_QUERY_SEEDS: [Qg6QuerySeed; QG6_QUERY_GROUPS] = [
    query_seed("term00042"),
    query_seed("term00137"),
    query_seed("src/main.rs"),
    query_seed(r"crate\:\:module\:\:TypeName"),
    query_seed("snake_case_identifier"),
    query_seed("camelCaseIdentifier"),
    query_seed("HTTPServer2"),
    query_seed("config.toml"),
    query_seed("path/to/module.rs"),
    query_seed("qgupdateg7d42"),
    query_seed("sha256deadbeef"),
    query_seed("user_id"),
    query_seed("nonexistentIdentifierAlpha"),
    query_seed("missing/path/file.rs"),
    query_seed(r"UnknownModule\:\:Type"),
    query_seed("qg6_nohit_identifier_15"),
];

const SHORT_KEYWORD_QUERY_SEEDS: [Qg6QuerySeed; QG6_QUERY_GROUPS] = [
    query_seed("term00001"),
    query_seed("term00002"),
    query_seed("generated"),
    query_seed("record"),
    query_seed("term00005"),
    query_seed("term00011"),
    query_seed("term00017"),
    query_seed("term00029"),
    query_seed("term02048"),
    query_seed("term04096"),
    query_seed("term06000"),
    query_seed("term08190"),
    query_seed("missingkeywordalpha"),
    query_seed("missingkeywordbeta"),
    query_seed("missingkeywordgamma"),
    query_seed("missingkeyworddelta"),
];

const NATURAL_LANGUAGE_QUERY_SEEDS: [Qg6QuerySeed; QG6_QUERY_GROUPS] = [
    query_seed("term00001 term00007 generated record"),
    query_seed("term00002 term00013 generated record"),
    query_seed("term00003 term00017 generated record"),
    query_seed("term00005 term00019 generated record"),
    query_seed("search record containing term00023 term00031"),
    query_seed("generated document mentions term00037 term00041"),
    query_seed("find term00043 beside term00047 in record"),
    query_seed("which generated record includes term00053 term00059"),
    query_seed("rare generated record term02048 term03001"),
    query_seed("locate term04096 with term05003 in generated content"),
    query_seed("record containing rare terms term06000 term07001"),
    query_seed("generated content near term08180 and term08190"),
    query_seed("no matching prose alpha qg6missingone"),
    query_seed("find absent generated record qg6missingtwo"),
    query_seed("where is qg6missingthree in this corpus"),
    query_seed("adversarial but valid prose qg6missingfour term08191"),
];

const PHRASE_QUERY_SEEDS: [Qg6QuerySeed; QG6_QUERY_GROUPS] = [
    query_seed("\"term00001 term00002\""),
    query_seed("\"term00002 term00003\""),
    query_seed("\"generated record\""),
    query_seed("\"term00005 term00006 term00007\""),
    query_seed("\"term00011 term00012\""),
    query_seed("\"term00017 term00018\""),
    query_seed("\"term00023 term00024 term00025\""),
    query_seed("\"record term00031 generated\""),
    query_seed("\"term02048 term02049\""),
    query_seed("\"term04096 term04097\""),
    query_seed("\"term06000 term06001 term06002\""),
    query_seed("\"term08180 term08181\""),
    query_seed("\"qg6 missing phrase alpha\""),
    query_seed("\"qg6 missing phrase beta\""),
    query_seed("\"qg6 missing phrase gamma delta\""),
    query_seed("\"qg6 adversarial nohit phrase epsilon\""),
];

const BOOLEAN_QUERY_SEEDS: [Qg6QuerySeed; QG6_QUERY_GROUPS] = [
    query_seed("term00001 OR term00002"),
    query_seed("term00003 AND term00004"),
    query_seed("term00005 OR term00007 OR term00011"),
    query_seed("(term00013 OR term00017) AND term00019"),
    query_seed("term00023 AND NOT term08191"),
    query_seed("term00029 OR NOT term08190"),
    query_seed("(term00031 AND term00037) OR term00041"),
    query_seed("term00043 AND (term00047 OR term00053) AND NOT term08189"),
    query_seed("term02048 OR term03001"),
    query_seed("term04096 AND term05003"),
    query_seed("(term06000 OR term07001) AND term08001"),
    query_seed("term08180 AND NOT (term00001 OR term00002)"),
    query_seed("qg6missingboolalpha AND term00001"),
    query_seed("qg6missingboolbeta OR qg6missingboolgamma"),
    query_seed("(qg6missingbooldelta AND term08191) OR qg6missingboolepsilon"),
    query_seed("qg6missingboolzeta AND NOT (term00001 OR term00002 OR term00003)"),
];

fn seeds_for_class(class: PerfQueryClass) -> &'static [Qg6QuerySeed; QG6_QUERY_GROUPS] {
    match class {
        PerfQueryClass::Identifier => &IDENTIFIER_QUERY_SEEDS,
        PerfQueryClass::ShortKeyword => &SHORT_KEYWORD_QUERY_SEEDS,
        PerfQueryClass::NaturalLanguage => &NATURAL_LANGUAGE_QUERY_SEEDS,
        PerfQueryClass::Phrase => &PHRASE_QUERY_SEEDS,
        PerfQueryClass::Boolean => &BOOLEAN_QUERY_SEEDS,
    }
}

fn build_normative_query_manifest() -> Result<Vec<Qg6QuerySpec>, Qg6HarnessError> {
    let mut queries = Vec::with_capacity(QG6_TOTAL_QUERY_COUNT);
    for class in PerfQueryClass::ALL {
        for (index, seed) in seeds_for_class(class).iter().enumerate() {
            queries.push(Qg6QuerySpec::build(
                format!("{}-{index:02}", class_slug(class)),
                seed.text.to_owned(),
                class,
                u8::try_from(index / 4).expect("sixteen-query grid row fits u8"),
                u8::try_from(index % 4).expect("sixteen-query grid column fits u8"),
                Qg6SupportDivergence::reviewed(),
                QG6_QUERY_GENERATOR_REVISION,
                QG6_CORPUS_GENERATOR_REVISION,
            )?);
        }
    }
    queries.sort_unstable_by(|left, right| left.id.cmp(&right.id));
    Ok(queries)
}

fn validate_complete_query_manifest(queries: &[Qg6QuerySpec]) -> Result<(), Qg6HarnessError> {
    if queries.len() != QG6_TOTAL_QUERY_COUNT {
        return Err(Qg6HarnessError::InvalidSpec {
            reason: "QG-6 frozen manifest requires exactly 80 queries".to_owned(),
        });
    }
    let mut ids = BTreeSet::new();
    let mut normalized_hashes = BTreeSet::new();
    let mut ast_hashes = BTreeSet::new();
    let mut class_counts = BTreeMap::new();
    let mut coordinates = BTreeMap::<PerfQueryClass, BTreeSet<(u8, u8)>>::new();
    for query in queries {
        query.validate_entry()?;
        if query.query_generator_revision != QG6_QUERY_GENERATOR_REVISION
            || query.corpus_generator_revision != QG6_CORPUS_GENERATOR_REVISION
        {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: format!(
                    "query {:?} has a non-normative generator revision",
                    query.id
                ),
            });
        }
        if !ids.insert(query.id.as_str())
            || !normalized_hashes.insert(query.normalized_text_sha256.as_str())
            || !ast_hashes.insert(query.parsed_ast_sha256.as_str())
        {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: format!(
                    "query {:?} aliases an existing stable ID, normalized text, or parsed AST",
                    query.id
                ),
            });
        }
        *class_counts.entry(query.class).or_insert(0_usize) += 1;
        coordinates
            .entry(query.class)
            .or_default()
            .insert((query.coverage_row, query.coverage_column));
    }
    for class in PerfQueryClass::ALL {
        if class_counts.get(&class) != Some(&QG6_QUERY_GROUPS) {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: format!(
                    "QG-6 class {} requires exactly sixteen queries",
                    class_slug(class)
                ),
            });
        }
        let expected_ids = (0..QG6_QUERY_GROUPS)
            .map(|index| format!("{}-{index:02}", class_slug(class)))
            .collect::<BTreeSet<_>>();
        let observed_ids = queries
            .iter()
            .filter(|query| query.class == class)
            .map(|query| query.id.clone())
            .collect::<BTreeSet<_>>();
        if observed_ids != expected_ids {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: format!(
                    "QG-6 class {} has a missing or replaced ID",
                    class_slug(class)
                ),
            });
        }
        let expected_coordinates = (0_u8..4)
            .flat_map(|row| (0_u8..4).map(move |column| (row, column)))
            .collect::<BTreeSet<_>>();
        if coordinates.get(&class) != Some(&expected_coordinates) {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: format!(
                    "QG-6 class {} must cover every neutral 4x4 coordinate exactly once",
                    class_slug(class)
                ),
            });
        }
    }
    let observed_sha256 = query_manifest_sha256(queries);
    if observed_sha256 != QG6_FROZEN_MANIFEST_SHA256 {
        return Err(Qg6HarnessError::InvalidSpec {
            reason: format!(
                "QG-6 frozen manifest hash drifted: expected={} observed={observed_sha256}",
                QG6_FROZEN_MANIFEST_SHA256
            ),
        });
    }
    Ok(())
}

const fn class_slug(class: PerfQueryClass) -> &'static str {
    match class {
        PerfQueryClass::Identifier => "identifier",
        PerfQueryClass::ShortKeyword => "short_keyword",
        PerfQueryClass::NaturalLanguage => "natural_language",
        PerfQueryClass::Phrase => "phrase",
        PerfQueryClass::Boolean => "boolean",
    }
}

fn class_from_query_id(id: &str) -> Result<PerfQueryClass, Qg6HarnessError> {
    PerfQueryClass::ALL
        .into_iter()
        .find(|class| {
            id.strip_prefix(class_slug(*class))
                .is_some_and(|suffix| suffix.starts_with('-'))
        })
        .ok_or_else(|| Qg6HarnessError::InvalidSpec {
            reason: format!("query ID {id:?} does not carry one canonical class prefix"),
        })
}

/// Immutable corpus, query, and semantic configuration identity shared by all arms.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qg6ExperimentIdentity {
    /// SHA-256 of the exact ordered corpus.
    pub corpus_sha256: String,
    /// SHA-256 of the frozen ordered query manifest.
    pub query_manifest_sha256: String,
    /// SHA-256 of the cross-engine semantic configuration contract.
    pub config_contract_sha256: String,
    /// Documents supplied to each logical index.
    pub document_count: u64,
    /// Requested result count for every query.
    pub k: usize,
}

/// One native ranked hit retained until the result receipt is built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qg6SearchHit {
    document_id: String,
    score_bits: u32,
}

impl Qg6SearchHit {
    /// Construct one native ranked hit from its external ID and exact score bits.
    #[must_use]
    pub fn new(document_id: impl Into<String>, score_bits: u32) -> Self {
        Self {
            document_id: document_id.into(),
            score_bits,
        }
    }
}

/// A search result whose optional native receipt digest is checked outside the timer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qg6SearchResult {
    ordered_hits: Vec<Qg6SearchHit>,
    total_count: u64,
    doc_count: Option<u64>,
    claimed_sha256: Option<String>,
}

impl Qg6SearchResult {
    /// Wrap exact ordered document IDs for focused tests.
    ///
    /// Production adapters should use [`Self::from_ranked_hits`] so score bits
    /// and the engine-native total count are bound explicitly.
    #[must_use]
    pub fn from_ordered_doc_ids(ordered_doc_ids: Vec<String>) -> Self {
        let total_count = usize_to_u64_infallible(ordered_doc_ids.len());
        Self {
            ordered_hits: ordered_doc_ids
                .into_iter()
                .map(|document_id| Qg6SearchHit::new(document_id, 0))
                .collect(),
            total_count,
            doc_count: None,
            claimed_sha256: None,
        }
    }

    /// Wrap native ordered hits and exact corpus/result cardinalities.
    #[must_use]
    pub fn from_ranked_hits(
        ordered_hits: Vec<Qg6SearchHit>,
        total_count: u64,
        doc_count: u64,
    ) -> Self {
        Self {
            ordered_hits,
            total_count,
            doc_count: Some(doc_count),
            claimed_sha256: None,
        }
    }

    /// Attach a native full-receipt digest. The runner independently recomputes it.
    #[must_use]
    pub fn with_claimed_sha256(mut self, claimed_sha256: impl Into<String>) -> Self {
        self.claimed_sha256 = Some(claimed_sha256.into());
        self
    }
}

impl From<Vec<String>> for Qg6SearchResult {
    fn from(value: Vec<String>) -> Self {
        Self::from_ordered_doc_ids(value)
    }
}

/// One redacted ranked hit in a sealed semantic receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qg6RankedHitReceipt {
    /// SHA-256 of the external document ID.
    pub document_id_sha256: String,
    /// Exact IEEE-754 score bits returned by the native engine.
    pub score_bits: u32,
}

impl Serialize for Qg6RankedHitReceipt {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (&self.document_id_sha256, self.score_bits).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Qg6RankedHitReceipt {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (document_id_sha256, score_bits) = <(String, u32)>::deserialize(deserializer)?;
        Ok(Self {
            document_id_sha256,
            score_bits,
        })
    }
}

/// Stable full result facts retained for parity and post-timing stability checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg6ResultReceipt {
    /// Exact number of returned hits.
    pub returned_count: usize,
    /// Redacted ordered native hits.
    pub ordered_hits: Vec<Qg6RankedHitReceipt>,
    /// Exact total number of matching documents before the top-k cutoff.
    pub total_count: u64,
    /// Exact live document cardinality of the searched index.
    pub doc_count: u64,
    /// Domain-separated SHA-256 over all preceding fields.
    pub receipt_sha256: String,
}

impl Qg6ResultReceipt {
    #[cfg(test)]
    pub(crate) fn from_redacted_hits(
        ordered_hits: Vec<Qg6RankedHitReceipt>,
        total_count: u64,
        doc_count: u64,
        k: usize,
    ) -> Result<Self, Qg6HarnessError> {
        let mut receipt = Self {
            returned_count: ordered_hits.len(),
            ordered_hits,
            total_count,
            doc_count,
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = receipt.canonical_sha256();
        receipt.verify(k, doc_count)?;
        Ok(receipt)
    }

    fn canonical_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hash_len_prefixed(&mut hasher, QG6_RESULT_RECEIPT_VERSION.as_bytes());
        hasher.update(usize_to_u64_infallible(self.returned_count).to_le_bytes());
        for hit in &self.ordered_hits {
            hash_len_prefixed(&mut hasher, hit.document_id_sha256.as_bytes());
            hasher.update(hit.score_bits.to_le_bytes());
        }
        hasher.update(self.total_count.to_le_bytes());
        hasher.update(self.doc_count.to_le_bytes());
        lower_hex(hasher.finalize())
    }

    #[cfg(test)]
    pub(crate) fn reseal_for_test(&mut self) {
        self.receipt_sha256 = self.canonical_sha256();
    }

    /// Verify shape, cardinality, hashes, and the self-seal.
    ///
    /// # Errors
    ///
    /// Rejects malformed or internally inconsistent receipt data.
    pub fn verify(&self, k: usize, expected_doc_count: u64) -> Result<(), Qg6HarnessError> {
        let returned_count = usize_to_u64(self.returned_count)?;
        let requested_count = usize_to_u64(k)?;
        let mut unique_document_ids = BTreeSet::new();
        if self.returned_count != self.ordered_hits.len()
            || self.returned_count > k
            || returned_count != self.total_count.min(requested_count)
            || self.total_count > self.doc_count
            || self.doc_count != expected_doc_count
            || self.ordered_hits.iter().any(|hit| {
                !is_lower_hex_sha256(&hit.document_id_sha256)
                    || hit.document_id_sha256 == EMPTY_DOCUMENT_ID_SHA256
                    || !f32::from_bits(hit.score_bits).is_finite()
                    || !unique_document_ids.insert(hit.document_id_sha256.as_str())
            })
            || self.receipt_sha256 != self.canonical_sha256()
        {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "QG-6 result receipt is malformed or has an invalid self-seal".to_owned(),
            });
        }
        Ok(())
    }
}

/// Named receipts for all six independent logical roles.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg6SixArmResultReceipts {
    /// Left Tantivy arm of the A/A null.
    pub tantivy_null_left: Qg6ResultReceipt,
    /// Right Tantivy arm of the A/A null.
    pub tantivy_null_right: Qg6ResultReceipt,
    /// Left Quill arm of the A/A null.
    pub quill_null_left: Qg6ResultReceipt,
    /// Right Quill arm of the A/A null.
    pub quill_null_right: Qg6ResultReceipt,
    /// Tantivy control arm of the A/B effect.
    pub effect_control: Qg6ResultReceipt,
    /// Quill treatment arm of the A/B effect.
    pub effect_treatment: Qg6ResultReceipt,
}

impl Qg6SixArmResultReceipts {
    /// Resolve a named logical role without relying on array position.
    #[must_use]
    pub const fn get(&self, role: Qg6ArmRole) -> &Qg6ResultReceipt {
        match role {
            Qg6ArmRole::TantivyNullLeft => &self.tantivy_null_left,
            Qg6ArmRole::TantivyNullRight => &self.tantivy_null_right,
            Qg6ArmRole::QuillNullLeft => &self.quill_null_left,
            Qg6ArmRole::QuillNullRight => &self.quill_null_right,
            Qg6ArmRole::EffectControl => &self.effect_control,
            Qg6ArmRole::EffectTreatment => &self.effect_treatment,
        }
    }
}

/// Recomputable redacted identity of one query in a semantic contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg6QueryIdentityReceipt {
    /// Stable non-sensitive query identifier.
    pub query_id: String,
    /// Frozen public query class.
    pub class: PerfQueryClass,
    /// SHA-256 of normalized source text.
    pub normalized_text_sha256: String,
    /// SHA-256 of the canonical parsed AST.
    pub parsed_ast_sha256: String,
    /// Neutral row in the frozen 4x4 grid.
    pub coverage_row: u8,
    /// Neutral column in the frozen 4x4 grid.
    pub coverage_column: u8,
    /// Reviewed support contract.
    pub support_divergence: Qg6SupportDivergence,
    /// Exactly supported result cutoffs.
    pub supported_k: [usize; 2],
    /// Query-generator revision.
    pub query_generator_revision: String,
    /// Corpus-generator revision.
    pub corpus_generator_revision: String,
    /// Domain-separated digest over every preceding field.
    pub query_identity_sha256: String,
}

impl Qg6QueryIdentityReceipt {
    pub(crate) fn from_query(query: &Qg6QuerySpec) -> Self {
        let mut receipt = Self {
            query_id: query.id.clone(),
            class: query.class,
            normalized_text_sha256: query.normalized_text_sha256.clone(),
            parsed_ast_sha256: query.parsed_ast_sha256.clone(),
            coverage_row: query.coverage_row,
            coverage_column: query.coverage_column,
            support_divergence: query.support_divergence.clone(),
            supported_k: query.supported_k,
            query_generator_revision: query.query_generator_revision.clone(),
            corpus_generator_revision: query.corpus_generator_revision.clone(),
            query_identity_sha256: String::new(),
        };
        receipt.query_identity_sha256 = receipt.canonical_sha256();
        receipt
    }

    /// Recompute the domain-separated redacted query digest.
    #[must_use]
    pub fn canonical_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hash_len_prefixed(&mut hasher, QG6_QUERY_IDENTITY_VERSION.as_bytes());
        hash_query_identity_receipt(&mut hasher, self);
        lower_hex(hasher.finalize())
    }

    fn verify(&self) -> Result<(), Qg6HarnessError> {
        if self.query_id.is_empty()
            || self.query_id.len() > MAX_QUERY_ID_BYTES
            || class_from_query_id(&self.query_id)? != self.class
            || !is_lower_hex_sha256(&self.normalized_text_sha256)
            || !is_lower_hex_sha256(&self.parsed_ast_sha256)
            || self.coverage_row >= 4
            || self.coverage_column >= 4
            || self.supported_k != QG6_SUPPORTED_K
            || self.query_generator_revision.is_empty()
            || self.query_generator_revision.len() > 256
            || self.corpus_generator_revision.is_empty()
            || self.corpus_generator_revision.len() > 512
            || !valid_supported_divergence(&self.support_divergence)
            || self.query_identity_sha256 != self.canonical_sha256()
        {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "QG-6 query identity receipt is malformed or has an invalid self-seal"
                    .to_owned(),
            });
        }
        Ok(())
    }
}

/// Ordered query-to-group mapping and its complete six-role semantic receipts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg6QueryGroupReceipt {
    /// Canonical zero-based query group.
    pub group_id: u64,
    /// Full recomputable redacted query identity.
    pub query: Qg6QueryIdentityReceipt,
    /// Full receipts for every native role.
    pub roles: Qg6SixArmResultReceipts,
}

/// Cell-local sealed semantic contract consumed by QG-6 evidence validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg6SemanticContract {
    /// Local contract schema.
    pub schema_version: String,
    /// Exact ordered prepared corpus.
    pub prepared_corpus_sha256: String,
    /// Exact canonical query manifest.
    pub query_manifest_sha256: String,
    /// Cross-engine configuration contract.
    pub config_contract_sha256: String,
    /// Exact live corpus cardinality.
    pub document_count: u64,
    /// Requested result cutoff.
    pub k: usize,
    /// Canonically ordered query mapping and six-role results.
    pub groups: Vec<Qg6QueryGroupReceipt>,
    /// Domain-separated SHA-256 over all preceding fields.
    pub contract_sha256: String,
}

impl Qg6SemanticContract {
    fn seal(mut self) -> Self {
        self.contract_sha256 = self.canonical_sha256();
        self
    }

    pub(crate) fn from_receipts(
        identity: &Qg6ExperimentIdentity,
        queries: &[Qg6QuerySpec],
        expected_results: &[Qg6SixArmResultReceipts],
    ) -> Result<Self, Qg6HarnessError> {
        validate_experiment_inputs(identity.document_count, identity.k, queries)?;
        if queries.len() != expected_results.len()
            || query_manifest_sha256(queries) != identity.query_manifest_sha256
        {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "QG-6 query/result receipts do not match the experiment identity"
                    .to_owned(),
            });
        }
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(queries.len())
            .map_err(|_| Qg6HarnessError::InvalidSpec {
                reason: "QG-6 semantic-contract allocation failed".to_owned(),
            })?;
        entries.extend(queries.iter().zip(expected_results));
        entries.sort_unstable_by(|(left, _), (right, _)| left.id.cmp(&right.id));

        let mut groups = Vec::new();
        groups
            .try_reserve_exact(entries.len())
            .map_err(|_| Qg6HarnessError::InvalidSpec {
                reason: "QG-6 semantic-contract allocation failed".to_owned(),
            })?;
        for (index, (query, roles)) in entries.into_iter().enumerate() {
            groups.push(Qg6QueryGroupReceipt {
                group_id: usize_to_u64(index)?,
                query: Qg6QueryIdentityReceipt::from_query(query),
                roles: roles.clone(),
            });
        }
        let contract = Self {
            schema_version: QG6_SEMANTIC_CONTRACT_VERSION.to_owned(),
            prepared_corpus_sha256: identity.corpus_sha256.clone(),
            query_manifest_sha256: identity.query_manifest_sha256.clone(),
            config_contract_sha256: identity.config_contract_sha256.clone(),
            document_count: identity.document_count,
            k: identity.k,
            groups,
            contract_sha256: String::new(),
        }
        .seal();
        contract.verify()?;
        Ok(contract)
    }

    /// Recompute the domain-separated semantic-contract digest.
    #[must_use]
    pub fn canonical_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hash_len_prefixed(&mut hasher, QG6_SEMANTIC_CONTRACT_VERSION.as_bytes());
        hash_len_prefixed(&mut hasher, self.prepared_corpus_sha256.as_bytes());
        hash_len_prefixed(&mut hasher, self.query_manifest_sha256.as_bytes());
        hash_len_prefixed(&mut hasher, self.config_contract_sha256.as_bytes());
        hasher.update(self.document_count.to_le_bytes());
        hasher.update(usize_to_u64_infallible(self.k).to_le_bytes());
        hasher.update(usize_to_u64_infallible(self.groups.len()).to_le_bytes());
        for group in &self.groups {
            hasher.update(group.group_id.to_le_bytes());
            hash_len_prefixed(&mut hasher, group.query.query_identity_sha256.as_bytes());
            for role in Qg6ArmRole::ALL {
                hash_len_prefixed(&mut hasher, group.roles.get(role).receipt_sha256.as_bytes());
            }
        }
        lower_hex(hasher.finalize())
    }

    /// Verify canonical ordering, complete role receipts, context, and self-seal.
    ///
    /// # Errors
    ///
    /// Rejects any malformed hash, mapping, receipt, or contract digest.
    pub fn verify(&self) -> Result<(), Qg6HarnessError> {
        if self.schema_version != QG6_SEMANTIC_CONTRACT_VERSION
            || !is_lower_hex_sha256(&self.prepared_corpus_sha256)
            || !is_lower_hex_sha256(&self.query_manifest_sha256)
            || !is_lower_hex_sha256(&self.config_contract_sha256)
            || self.document_count == 0
            || self.k == 0
            || self.k > MAX_K
            || !QG6_SUPPORTED_K.contains(&self.k)
            || self.groups.is_empty()
            || self.groups.len() > MAX_QUERY_COUNT
            || self.contract_sha256 != self.canonical_sha256()
        {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "QG-6 semantic contract context or self-seal is invalid".to_owned(),
            });
        }
        let observed_manifest_sha256 =
            query_identity_manifest_sha256(self.groups.iter().map(|group| &group.query));
        if observed_manifest_sha256 != self.query_manifest_sha256 {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "QG-6 semantic contract query-manifest digest does not recompute"
                    .to_owned(),
            });
        }
        let mut previous_query_id: Option<&str> = None;
        for (index, group) in self.groups.iter().enumerate() {
            if group.group_id != usize_to_u64(index)?
                || previous_query_id
                    .is_some_and(|previous| previous >= group.query.query_id.as_str())
            {
                return Err(Qg6HarnessError::InvalidSpec {
                    reason: "QG-6 semantic contract query mapping is not canonical".to_owned(),
                });
            }
            group.query.verify()?;
            for role in Qg6ArmRole::ALL {
                group.roles.get(role).verify(self.k, self.document_count)?;
            }
            previous_query_id = Some(group.query.query_id.as_str());
        }
        Ok(())
    }
}

/// Domain-separated digest of one validated result repeated for a raw sample.
///
/// `work_units` may later represent true leaves rather than a median aggregate;
/// this function binds only the explicit sequence length and receipt, without
/// assigning any permanent estimator meaning to one raw row.
///
/// # Errors
///
/// Rejects zero work or a malformed receipt digest.
pub fn qg6_result_sequence_sha256(
    receipt: &Qg6ResultReceipt,
    work_units: u64,
) -> Result<String, Qg6HarnessError> {
    qg6_result_digest_sequence_sha256(&receipt.receipt_sha256, work_units)
}

fn qg6_result_digest_sequence_sha256(
    receipt_sha256: &str,
    work_units: u64,
) -> Result<String, Qg6HarnessError> {
    if work_units == 0 || !is_lower_hex_sha256(receipt_sha256) {
        return Err(Qg6HarnessError::InvalidSpec {
            reason: "QG-6 result sequence requires positive work and a valid receipt".to_owned(),
        });
    }
    let mut hasher = Sha256::new();
    hash_len_prefixed(&mut hasher, QG6_RESULT_SEQUENCE_VERSION.as_bytes());
    hasher.update(work_units.to_le_bytes());
    hash_len_prefixed(&mut hasher, receipt_sha256.as_bytes());
    Ok(lower_hex(hasher.finalize()))
}

/// Per-arm lifecycle counts proving preparation never re-enters measurement.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qg6ArmLifecycle {
    /// Engine constructor invocations.
    pub build_calls: u64,
    /// Population batches supplied to the engine.
    pub populate_calls: u64,
    /// Total documents supplied across population batches.
    pub populated_documents: u64,
    /// Commit invocations.
    pub commit_calls: u64,
    /// Untimed exact-parity queries.
    pub preflight_search_calls: u64,
    /// Untimed warmup queries.
    pub warmup_search_calls: u64,
    /// Timed queries.
    pub timed_search_calls: u64,
    /// Untimed full native stability queries after measurement.
    pub postflight_search_calls: u64,
    /// Setup operations observed after timing began. This must remain zero.
    pub timed_setup_calls: u64,
}

/// Complete lifecycle receipt in stable arm order.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qg6LifecycleReceipt {
    /// One counter set for each [`Qg6ArmRole::ALL`] entry.
    pub arms: [Qg6ArmLifecycle; 6],
}

impl Qg6LifecycleReceipt {
    /// Counters for one logical arm.
    #[must_use]
    pub fn arm(&self, role: Qg6ArmRole) -> &Qg6ArmLifecycle {
        &self.arms[role.index()]
    }

    fn arm_mut(&mut self, role: Qg6ArmRole) -> &mut Qg6ArmLifecycle {
        &mut self.arms[role.index()]
    }
}

/// Mutable setup-only capability passed to the arm builder.
///
/// The type is not retained by the prepared experiment, so setup counters
/// cannot be incremented from warmup or timed search code.
pub struct Qg6SetupRecorder<'a> {
    counters: &'a mut Qg6ArmLifecycle,
}

impl Qg6SetupRecorder<'_> {
    /// Record one population call and its document count.
    pub fn record_population_batch(&mut self, document_count: u64) {
        self.counters.populate_calls = self.counters.populate_calls.saturating_add(1);
        self.counters.populated_documents = self
            .counters
            .populated_documents
            .saturating_add(document_count);
    }

    /// Record the commit that makes this arm searchable.
    pub fn record_commit(&mut self) {
        self.counters.commit_calls = self.counters.commit_calls.saturating_add(1);
    }
}

/// One paired block in the deterministic six-arm schedule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg6PairBlock {
    /// Globally unique block identifier in this experiment.
    pub block_id: u64,
    /// Query-round cluster containing all three formal comparisons.
    pub unit_id: u64,
    /// Index into the frozen ordered query manifest.
    pub query_index: usize,
    /// Tantivy null, Quill null, or cross-engine effect comparison.
    pub comparison: Qg6Comparison,
    /// Arm executed first.
    pub first: Qg6ArmRole,
    /// Arm executed second.
    pub second: Qg6ArmRole,
}

/// Externally retained authority for the exact schedule frozen before timing.
///
/// The prepared runner consumes this receipt by reference and executes its
/// sealed schedule directly. The completed measurement can therefore be
/// checked against an authority retained outside the observed sample table;
/// timed rows never authorize their own schedule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg6ScheduleAuthority {
    /// Schema identifier for this authority contract.
    pub schema_version: String,
    /// Exact prepared experiment for which the schedule was frozen.
    pub identity: Qg6ExperimentIdentity,
    /// Number of frozen query groups scheduled.
    pub query_count: usize,
    /// Equal query-round units per query.
    pub rounds_per_query: usize,
    /// Individually timed searches summarized by every parent sample.
    pub searches_per_sample: usize,
    /// Seed that deterministically generated the schedule.
    pub schedule_seed: u64,
    /// Complete three-comparison-per-unit schedule frozen before timing.
    pub schedule: Vec<Qg6PairBlock>,
    /// Domain-separated self-seal over every preceding field.
    pub authority_sha256: String,
}

impl Qg6ScheduleAuthority {
    /// Freeze and seal one canonical schedule for an externally prepared experiment.
    ///
    /// Parent processes and evidence replay can retain this authority before
    /// handing timing work to a child. Measurement still verifies the exact
    /// authority against its validated experiment identity before any warmup.
    ///
    /// # Errors
    ///
    /// Rejects malformed experiment identity, query, round, or timing-leaf
    /// cardinality and any schedule allocation failure.
    pub fn for_experiment(
        identity: Qg6ExperimentIdentity,
        query_count: usize,
        rounds_per_query: usize,
        searches_per_sample: usize,
        schedule_seed: u64,
    ) -> Result<Self, Qg6HarnessError> {
        if searches_per_sample == 0 || searches_per_sample > QG6_TIMED_SEARCHES_PER_SAMPLE {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "QG-6 schedule authority has invalid timing-leaf cardinality".to_owned(),
            });
        }
        let schedule =
            seeded_interleaved_six_arm_schedule(query_count, rounds_per_query, schedule_seed)?;
        let mut authority = Self {
            schema_version: QG6_SCHEDULE_AUTHORITY_VERSION.to_owned(),
            identity,
            query_count,
            rounds_per_query,
            searches_per_sample,
            schedule_seed,
            schedule,
            authority_sha256: String::new(),
        };
        authority.authority_sha256 = authority.recomputed_sha256()?;
        authority.verify()?;
        Ok(authority)
    }

    fn recomputed_sha256(&self) -> Result<String, Qg6HarnessError> {
        let mut hasher = Sha256::new();
        hash_len_prefixed(&mut hasher, QG6_SCHEDULE_AUTHORITY_VERSION.as_bytes());
        hash_len_prefixed(&mut hasher, self.schema_version.as_bytes());
        hash_len_prefixed(&mut hasher, self.identity.corpus_sha256.as_bytes());
        hash_len_prefixed(&mut hasher, self.identity.query_manifest_sha256.as_bytes());
        hash_len_prefixed(&mut hasher, self.identity.config_contract_sha256.as_bytes());
        hasher.update(self.identity.document_count.to_le_bytes());
        hasher.update(usize_to_u64(self.identity.k)?.to_le_bytes());
        hasher.update(usize_to_u64(self.query_count)?.to_le_bytes());
        hasher.update(usize_to_u64(self.rounds_per_query)?.to_le_bytes());
        hasher.update(usize_to_u64(self.searches_per_sample)?.to_le_bytes());
        hasher.update(self.schedule_seed.to_le_bytes());
        hasher.update(usize_to_u64(self.schedule.len())?.to_le_bytes());
        for block in &self.schedule {
            hasher.update(block.block_id.to_le_bytes());
            hasher.update(block.unit_id.to_le_bytes());
            hasher.update(usize_to_u64(block.query_index)?.to_le_bytes());
            hasher.update([qg6_comparison_tag(block.comparison)]);
            hasher.update([qg6_arm_role_tag(block.first)]);
            hasher.update([qg6_arm_role_tag(block.second)]);
        }
        Ok(lower_hex(hasher.finalize()))
    }

    /// Verify that this authority is canonical and reproduces its full schedule.
    ///
    /// # Errors
    ///
    /// Rejects malformed identities, invalid cardinalities, schedule mutations,
    /// role relabeling, and a self-seal that no longer recomputes.
    pub fn verify(&self) -> Result<(), Qg6HarnessError> {
        if self.schema_version != QG6_SCHEDULE_AUTHORITY_VERSION
            || !is_lower_hex_sha256(&self.identity.corpus_sha256)
            || !is_lower_hex_sha256(&self.identity.query_manifest_sha256)
            || !is_lower_hex_sha256(&self.identity.config_contract_sha256)
            || self.identity.document_count == 0
            || self.identity.k == 0
            || self.identity.k > MAX_K
            || self.searches_per_sample == 0
            || self.searches_per_sample > QG6_TIMED_SEARCHES_PER_SAMPLE
            || self.schedule
                != seeded_interleaved_six_arm_schedule(
                    self.query_count,
                    self.rounds_per_query,
                    self.schedule_seed,
                )?
            || self.authority_sha256 != self.recomputed_sha256()?
        {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "QG-6 schedule authority is malformed, mutated, or incorrectly sealed"
                    .to_owned(),
            });
        }
        Ok(())
    }

    fn verify_for(
        &self,
        identity: &Qg6ExperimentIdentity,
        query_count: usize,
    ) -> Result<(), Qg6HarnessError> {
        self.verify()?;
        if &self.identity != identity || self.query_count != query_count {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "QG-6 schedule authority belongs to a different prepared experiment"
                    .to_owned(),
            });
        }
        Ok(())
    }
}

/// One exact same-invocation search interval retained within a timed sample.
///
/// Every fact shared by the leaves lives once on [`Qg6TimedSample`]. On the
/// wire a standalone leaf serializes as `[started_ns, ended_ns]`. Inside a
/// [`Qg6TimedSample`], the bounded ordered leaf vector is encoded once as
/// canonical `start_delta:duration` decimal pairs relative to the parent.
/// Result normalization and parity verification remain outside the interval,
/// and the parent binds the one shared result receipt proven for every
/// invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qg6SearchTimingLeafReceipt {
    /// Monotonic start offset relative to the measurement origin.
    pub started_ns: u64,
    /// Monotonic end offset relative to the measurement origin.
    pub ended_ns: u64,
}

impl Qg6SearchTimingLeafReceipt {
    pub(crate) fn from_interval(started_ns: u64, ended_ns: u64) -> Result<Self, Qg6HarnessError> {
        let receipt = Self {
            started_ns,
            ended_ns,
        };
        receipt.verify()?;
        Ok(receipt)
    }

    /// Exact raw monotonic interval in nanoseconds.
    #[must_use]
    pub fn observed_latency_ns(&self) -> u64 {
        self.ended_ns.saturating_sub(self.started_ns)
    }

    /// The exact raw interval converted to milliseconds.
    #[must_use]
    pub fn observed_latency_ms(&self) -> f64 {
        nanoseconds_to_millis(self.observed_latency_ns())
    }

    fn verify(&self) -> Result<(), Qg6HarnessError> {
        self.ended_ns
            .checked_sub(self.started_ns)
            .filter(|elapsed| *elapsed != 0)
            .ok_or_else(|| Qg6HarnessError::InvalidSpec {
                reason: "QG-6 timing leaf has an invalid monotonic interval".to_owned(),
            })?;
        Ok(())
    }
}

impl Serialize for Qg6SearchTimingLeafReceipt {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        [self.started_ns, self.ended_ns].serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Qg6SearchTimingLeafReceipt {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let [started_ns, ended_ns] = <[u64; 2]>::deserialize(deserializer)?;
        Self::from_interval(started_ns, ended_ns).map_err(serde::de::Error::custom)
    }
}

/// One directly observed timed sample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qg6TimedSample {
    /// Paired block shared with exactly one other arm.
    pub block_id: u64,
    /// Unique sample identifier.
    pub sample_id: u64,
    /// Stable, non-sensitive query ID.
    pub query_id: String,
    /// Index into the frozen ordered query manifest.
    pub query_index: usize,
    /// Formal Tantivy-null, Quill-null, or effect stream.
    pub comparison: Qg6Comparison,
    /// Logical arm.
    pub arm: Qg6ArmRole,
    /// Execution order inside the block.
    pub order: Qg6SampleOrder,
    /// Monotonic start offset relative to the measurement origin.
    pub started_ns: u64,
    /// Monotonic end offset relative to the measurement origin.
    pub ended_ns: u64,
    /// Median latency across the fixed per-arm search subsample.
    pub observed_latency_ns: u64,
    /// Number of individually timed searches summarized by this sample.
    pub subsample_count: u64,
    /// One result receipt proven independently for every invocation in this sample.
    pub result_receipt_sha256: String,
    /// Digest over every independently recomputed result receipt.
    pub result_sha256: String,
    /// Ordered same-invocation raw timing leaves summarized by this p50 sample.
    /// The persisted v7 wire uses one bounded canonical parent-relative string.
    pub timing_leaves: Vec<Qg6SearchTimingLeafReceipt>,
    /// Domain-separated seal over the ordered leaf receipts and parent sample facts.
    pub timing_leaves_sha256: String,
}

struct Qg6TimingLeavesWire<'a> {
    parent_started_ns: u64,
    leaves: &'a [Qg6SearchTimingLeafReceipt],
}

impl Serialize for Qg6TimingLeavesWire<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let capacity = self
            .leaves
            .len()
            .checked_mul(QG6_TIMING_LEAF_MAX_PAIR_WIRE_BYTES)
            .ok_or_else(|| serde::ser::Error::custom("QG-6 timing leaf wire size overflows"))?;
        let mut encoded = String::new();
        encoded
            .try_reserve_exact(capacity)
            .map_err(|_| serde::ser::Error::custom("QG-6 timing leaf wire allocation failed"))?;
        for leaf in self.leaves {
            let started_delta_ns = leaf
                .started_ns
                .checked_sub(self.parent_started_ns)
                .ok_or_else(|| {
                    serde::ser::Error::custom("QG-6 timing leaf starts before its parent sample")
                })?;
            let duration_ns = leaf.ended_ns.checked_sub(leaf.started_ns).ok_or_else(|| {
                serde::ser::Error::custom("QG-6 timing leaf has an inverted interval")
            })?;
            if !encoded.is_empty() {
                encoded.push(',');
            }
            write!(&mut encoded, "{started_delta_ns}:{duration_ns}")
                .map_err(serde::ser::Error::custom)?;
        }
        serializer.serialize_str(&encoded)
    }
}

#[derive(Serialize)]
struct Qg6TimedSampleWire<'a> {
    block_id: u64,
    sample_id: u64,
    query_id: &'a str,
    query_index: usize,
    comparison: Qg6Comparison,
    arm: Qg6ArmRole,
    order: Qg6SampleOrder,
    started_ns: u64,
    ended_ns: u64,
    observed_latency_ns: u64,
    subsample_count: u64,
    result_receipt_sha256: &'a str,
    result_sha256: &'a str,
    timing_leaves: Qg6TimingLeavesWire<'a>,
    timing_leaves_sha256: &'a str,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Qg6TimedSampleWireOwned {
    block_id: u64,
    sample_id: u64,
    query_id: String,
    query_index: usize,
    comparison: Qg6Comparison,
    arm: Qg6ArmRole,
    order: Qg6SampleOrder,
    started_ns: u64,
    ended_ns: u64,
    observed_latency_ns: u64,
    subsample_count: u64,
    result_receipt_sha256: String,
    result_sha256: String,
    timing_leaves: String,
    timing_leaves_sha256: String,
}

impl Serialize for Qg6TimedSample {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Qg6TimedSampleWire {
            block_id: self.block_id,
            sample_id: self.sample_id,
            query_id: &self.query_id,
            query_index: self.query_index,
            comparison: self.comparison,
            arm: self.arm,
            order: self.order,
            started_ns: self.started_ns,
            ended_ns: self.ended_ns,
            observed_latency_ns: self.observed_latency_ns,
            subsample_count: self.subsample_count,
            result_receipt_sha256: &self.result_receipt_sha256,
            result_sha256: &self.result_sha256,
            timing_leaves: Qg6TimingLeavesWire {
                parent_started_ns: self.started_ns,
                leaves: &self.timing_leaves,
            },
            timing_leaves_sha256: &self.timing_leaves_sha256,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Qg6TimedSample {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = Qg6TimedSampleWireOwned::deserialize(deserializer)?;
        let maximum_wire_bytes = QG6_TIMED_SEARCHES_PER_SAMPLE
            .checked_mul(QG6_TIMING_LEAF_MAX_PAIR_WIRE_BYTES)
            .ok_or_else(|| serde::de::Error::custom("QG-6 timing leaf wire limit overflows"))?;
        if wire.timing_leaves.len() > maximum_wire_bytes {
            return Err(serde::de::Error::custom(
                "QG-6 timing leaf wire exceeds its bounded representation",
            ));
        }
        let leaf_count = if wire.timing_leaves.is_empty() {
            0
        } else {
            wire.timing_leaves.split(',').count()
        };
        if leaf_count > QG6_TIMED_SEARCHES_PER_SAMPLE {
            return Err(serde::de::Error::custom(
                "QG-6 timing leaf wire exceeds its bounded cardinality",
            ));
        }
        let mut timing_leaves = Vec::new();
        timing_leaves
            .try_reserve_exact(leaf_count)
            .map_err(serde::de::Error::custom)?;
        if !wire.timing_leaves.is_empty() {
            for pair in wire.timing_leaves.split(',') {
                if pair.is_empty() {
                    return Err(serde::de::Error::custom(
                        "QG-6 timing leaf wire contains an empty pair",
                    ));
                }
                let (started_delta_text, duration_text) =
                    pair.split_once(':').ok_or_else(|| {
                        serde::de::Error::custom("QG-6 timing leaf wire pair lacks its separator")
                    })?;
                let is_canonical_decimal = |value: &str| {
                    !value.is_empty()
                        && value.bytes().all(|byte| byte.is_ascii_digit())
                        && (value.len() == 1 || !value.starts_with('0'))
                };
                if !is_canonical_decimal(started_delta_text) || !is_canonical_decimal(duration_text)
                {
                    return Err(serde::de::Error::custom(
                        "QG-6 timing leaf wire contains noncanonical decimal data",
                    ));
                }
                let started_delta_ns = started_delta_text
                    .parse::<u64>()
                    .map_err(serde::de::Error::custom)?;
                let duration_ns = duration_text
                    .parse::<u64>()
                    .map_err(serde::de::Error::custom)?;
                let started_ns = wire
                    .started_ns
                    .checked_add(started_delta_ns)
                    .ok_or_else(|| serde::de::Error::custom("QG-6 timing leaf start overflows"))?;
                let ended_ns = started_ns
                    .checked_add(duration_ns)
                    .ok_or_else(|| serde::de::Error::custom("QG-6 timing leaf end overflows"))?;
                timing_leaves.push(
                    Qg6SearchTimingLeafReceipt::from_interval(started_ns, ended_ns)
                        .map_err(serde::de::Error::custom)?,
                );
            }
        }
        let sample = Self {
            block_id: wire.block_id,
            sample_id: wire.sample_id,
            query_id: wire.query_id,
            query_index: wire.query_index,
            comparison: wire.comparison,
            arm: wire.arm,
            order: wire.order,
            started_ns: wire.started_ns,
            ended_ns: wire.ended_ns,
            observed_latency_ns: wire.observed_latency_ns,
            subsample_count: wire.subsample_count,
            result_receipt_sha256: wire.result_receipt_sha256,
            result_sha256: wire.result_sha256,
            timing_leaves,
            timing_leaves_sha256: wire.timing_leaves_sha256,
        };
        sample
            .verify_timing_leaves()
            .map_err(serde::de::Error::custom)?;
        Ok(sample)
    }
}

impl Qg6TimedSample {
    /// Verify the bounded ordered leaf receipts that support this p50 sample.
    ///
    /// # Errors
    ///
    /// Rejects missing, extra, reordered, zero-width, overlapping, or
    /// out-of-parent intervals, as well as a leaf/result digest that cannot
    /// reproduce the parent sample's p50 and result sequence.
    pub fn verify_timing_leaves(&self) -> Result<(), Qg6HarnessError> {
        let expected_count =
            usize::try_from(self.subsample_count).map_err(|_| Qg6HarnessError::InvalidSpec {
                reason: "QG-6 timing leaf cardinality does not fit usize".to_owned(),
            })?;
        if expected_count == 0
            || expected_count > QG6_TIMED_SEARCHES_PER_SAMPLE
            || self.timing_leaves.len() != expected_count
            || self.started_ns >= self.ended_ns
            || self.observed_latency_ns == 0
            || self.query_id.is_empty()
            || self.query_id.len() > MAX_QUERY_ID_BYTES
            || self.query_index >= MAX_QUERY_COUNT
            || !is_lower_hex_sha256(&self.result_receipt_sha256)
            || !is_lower_hex_sha256(&self.result_sha256)
            || !is_lower_hex_sha256(&self.timing_leaves_sha256)
        {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "QG-6 timing leaf receipt has invalid cardinality or parent facts"
                    .to_owned(),
            });
        }

        let mut latencies_ns = Vec::new();
        latencies_ns
            .try_reserve_exact(expected_count)
            .map_err(|_| Qg6HarnessError::InvalidSpec {
                reason: "QG-6 timing leaf validation allocation failed".to_owned(),
            })?;
        let mut previous_ended_ns = None;
        for leaf in &self.timing_leaves {
            leaf.verify()?;
            if leaf.started_ns < self.started_ns
                || leaf.ended_ns > self.ended_ns
                || previous_ended_ns.is_some_and(|previous| leaf.started_ns < previous)
            {
                return Err(Qg6HarnessError::InvalidSpec {
                    reason: "QG-6 timing leaves do not have the exact parent order and interval"
                        .to_owned(),
                });
            }
            previous_ended_ns = Some(leaf.ended_ns);
            let latency_ns = leaf.ended_ns.checked_sub(leaf.started_ns).ok_or_else(|| {
                Qg6HarnessError::InvalidSpec {
                    reason: "QG-6 timing leaf interval changed after verification".to_owned(),
                }
            })?;
            latencies_ns.push(latency_ns);
        }
        latencies_ns.sort_unstable();
        if self.observed_latency_ns != median_sorted_u64(&latencies_ns)
            || self.result_sha256
                != qg6_result_digest_sequence_sha256(
                    &self.result_receipt_sha256,
                    self.subsample_count,
                )?
            || self.timing_leaves_sha256 != self.recomputed_timing_leaves_sha256()?
        {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "QG-6 timing leaves do not reproduce their parent sample".to_owned(),
            });
        }
        Ok(())
    }

    pub(crate) fn recomputed_timing_leaves_sha256(&self) -> Result<String, Qg6HarnessError> {
        let mut hasher = Sha256::new();
        hash_len_prefixed(&mut hasher, QG6_TIMING_LEAF_RECEIPT_VERSION.as_bytes());
        hasher.update(self.block_id.to_le_bytes());
        hasher.update(self.sample_id.to_le_bytes());
        hash_len_prefixed(&mut hasher, self.query_id.as_bytes());
        hasher.update(usize_to_u64(self.query_index)?.to_le_bytes());
        hasher.update([qg6_comparison_tag(self.comparison)]);
        hasher.update([qg6_arm_role_tag(self.arm)]);
        hasher.update([qg6_sample_order_tag(self.order)]);
        hasher.update(self.started_ns.to_le_bytes());
        hasher.update(self.ended_ns.to_le_bytes());
        hasher.update(self.observed_latency_ns.to_le_bytes());
        hasher.update(self.subsample_count.to_le_bytes());
        hash_len_prefixed(&mut hasher, self.result_receipt_sha256.as_bytes());
        hash_len_prefixed(&mut hasher, self.result_sha256.as_bytes());
        hasher.update(usize_to_u64(self.timing_leaves.len())?.to_le_bytes());
        for leaf in &self.timing_leaves {
            hasher.update(leaf.started_ns.to_le_bytes());
            hasher.update(leaf.ended_ns.to_le_bytes());
        }
        Ok(lower_hex(hasher.finalize()))
    }
}

/// Output of one prepared six-arm measurement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg6Measurement {
    /// Frozen identity shared by all six arms.
    pub identity: Qg6ExperimentIdentity,
    /// Seed that fully determines the timed schedule.
    pub schedule_seed: u64,
    /// Equal warmup count per arm and query.
    pub warmup_rounds: usize,
    /// Equal timed pair count per comparison and query.
    pub rounds_per_query: usize,
    /// Equal individually timed searches summarized by each arm sample.
    pub searches_per_sample: usize,
    /// Interleaved three-comparison schedule.
    pub schedule: Vec<Qg6PairBlock>,
    /// Exact pre-timing schedule authority supplied and retained by the caller.
    pub schedule_authority: Qg6ScheduleAuthority,
    /// Raw per-arm monotonic intervals.
    pub samples: Vec<Qg6TimedSample>,
    /// Lifecycle contamination proof.
    pub lifecycle: Qg6LifecycleReceipt,
    /// Sealed query mapping and full six-role semantic receipts.
    pub semantic_contract: Qg6SemanticContract,
}

impl Qg6Measurement {
    /// Verify every observed row against an independently retained pre-timing authority.
    ///
    /// # Errors
    ///
    /// Rejects authority substitution, schedule mutation, missing or relabeled
    /// samples, duplicate identities, malformed leaves, or lifecycle drift.
    pub fn verify_against_schedule_authority(
        &self,
        authority: &Qg6ScheduleAuthority,
    ) -> Result<(), Qg6HarnessError> {
        authority.verify_for(&self.identity, self.semantic_contract.groups.len())?;
        self.semantic_contract.verify()?;
        if &self.schedule_authority != authority
            || self.schedule_seed != authority.schedule_seed
            || self.warmup_rounds == 0
            || self.rounds_per_query != authority.rounds_per_query
            || self.searches_per_sample != authority.searches_per_sample
            || self.schedule != authority.schedule
            || self.semantic_contract.prepared_corpus_sha256 != self.identity.corpus_sha256
            || self.semantic_contract.query_manifest_sha256 != self.identity.query_manifest_sha256
            || self.semantic_contract.config_contract_sha256 != self.identity.config_contract_sha256
            || self.semantic_contract.document_count != self.identity.document_count
            || self.semantic_contract.k != self.identity.k
        {
            return Err(Qg6HarnessError::InvalidSpec {
                reason:
                    "QG-6 measurement does not carry the externally retained schedule authority"
                        .to_owned(),
            });
        }
        let expected_samples = authority.schedule.len().checked_mul(2).ok_or_else(|| {
            Qg6HarnessError::InvalidSpec {
                reason: "QG-6 authority sample cardinality overflow".to_owned(),
            }
        })?;
        if self.samples.len() != expected_samples {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "QG-6 measurement sample cardinality differs from its authority".to_owned(),
            });
        }
        let mut sample_ids = BTreeSet::new();
        let mut rows = BTreeMap::new();
        for sample in &self.samples {
            sample.verify_timing_leaves()?;
            if !sample_ids.insert(sample.sample_id)
                || rows
                    .insert((sample.block_id, sample.order), sample)
                    .is_some()
            {
                return Err(Qg6HarnessError::InvalidSpec {
                    reason: "QG-6 measurement repeats one authority-bound sample identity"
                        .to_owned(),
                });
            }
        }
        let mut previous_ended_ns = None;
        for block in &authority.schedule {
            for (order, role) in [
                (Qg6SampleOrder::First, block.first),
                (Qg6SampleOrder::Second, block.second),
            ] {
                let sample = rows.remove(&(block.block_id, order)).ok_or_else(|| {
                    Qg6HarnessError::InvalidSpec {
                        reason: "QG-6 measurement omits one authority-bound sample".to_owned(),
                    }
                })?;
                let expected_sample_id = block
                    .block_id
                    .checked_mul(2)
                    .and_then(|base| base.checked_add(u64::from(order == Qg6SampleOrder::Second)))
                    .ok_or_else(|| Qg6HarnessError::InvalidSpec {
                        reason: "QG-6 authority sample ID overflow".to_owned(),
                    })?;
                let expected_result_receipt = self.semantic_contract.groups[block.query_index]
                    .roles
                    .get(role);
                let expected_result_sha256 = qg6_result_sequence_sha256(
                    expected_result_receipt,
                    usize_to_u64(authority.searches_per_sample)?,
                )?;
                if sample.sample_id != expected_sample_id
                    || sample.query_index != block.query_index
                    || sample.query_id
                        != self.semantic_contract.groups[block.query_index]
                            .query
                            .query_id
                    || sample.comparison != block.comparison
                    || sample.arm != role
                    || sample.subsample_count != usize_to_u64(authority.searches_per_sample)?
                    || sample.result_receipt_sha256 != expected_result_receipt.receipt_sha256
                    || sample.result_sha256 != expected_result_sha256
                    || previous_ended_ns.is_some_and(|previous| sample.started_ns < previous)
                {
                    return Err(Qg6HarnessError::InvalidSpec {
                        reason:
                            "QG-6 measurement relabels or reorders an authority-bound schedule row"
                                .to_owned(),
                    });
                }
                previous_ended_ns = Some(sample.ended_ns);
            }
        }
        if !rows.is_empty() {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "QG-6 measurement contains rows outside its schedule authority".to_owned(),
            });
        }
        verify_lifecycle(
            &self.lifecycle,
            self.identity.document_count,
            authority.query_count,
            self.warmup_rounds,
            authority.rounds_per_query,
            authority.searches_per_sample,
        )
    }
}

/// Bounded, non-sensitive failure surface for prepared QG-6 execution.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Qg6HarnessError {
    /// Invalid immutable input or run configuration.
    #[error("invalid QG-6 prepared-run specification: {reason}")]
    InvalidSpec {
        /// Bounded static reason; never includes corpus or query text.
        reason: String,
    },
    /// An engine adapter failed.
    #[error(
        "QG-6 {phase:?} adapter failure for {arm:?}, query {query_id:?}: \
         sha256={error_sha256}, bytes={error_bytes}"
    )]
    AdapterFailure {
        /// Lifecycle phase.
        phase: Qg6Phase,
        /// Arm being prepared or queried.
        arm: Qg6ArmRole,
        /// Stable query ID, or `"<setup>"`.
        query_id: String,
        /// SHA-256 of the adapter error string.
        error_sha256: String,
        /// Error string byte length.
        error_bytes: usize,
    },
    /// A prepared arm did not ingest the exact declared corpus once.
    #[error("invalid QG-6 setup lifecycle for {arm:?}: {reason}")]
    InvalidSetup {
        /// Logical arm.
        arm: Qg6ArmRole,
        /// Bounded reason.
        reason: String,
    },
    /// A native result digest disagreed with the independent digest.
    #[error(
        "QG-6 {phase:?} result digest mismatch for {arm:?}, query {query_id:?}: \
         claimed={claimed_sha256}, computed={computed_sha256}"
    )]
    ResultDigestMismatch {
        /// Lifecycle phase.
        phase: Qg6Phase,
        /// Logical arm.
        arm: Qg6ArmRole,
        /// Stable query ID.
        query_id: String,
        /// Digest supplied by the adapter.
        claimed_sha256: String,
        /// Independently recomputed digest.
        computed_sha256: String,
    },
    /// A result exceeded the declared `k` or a document ID exceeded its bound.
    #[error("invalid QG-6 {phase:?} result for {arm:?}, query {query_id:?}: {reason}")]
    InvalidResult {
        /// Lifecycle phase.
        phase: Qg6Phase,
        /// Logical arm.
        arm: Qg6ArmRole,
        /// Stable query ID.
        query_id: String,
        /// Bounded reason with no raw document ID.
        reason: String,
    },
    /// Exact hit counts differ across arms before timing.
    #[error(
        "QG-6 preflight hit-count mismatch for query {query_id:?}: \
         {expected_arm:?}={expected_count}, {observed_arm:?}={observed_count}"
    )]
    HitCountMismatch {
        /// Stable query ID.
        query_id: String,
        /// Baseline arm.
        expected_arm: Qg6ArmRole,
        /// Compared arm.
        observed_arm: Qg6ArmRole,
        /// Baseline hit count.
        expected_count: usize,
        /// Compared hit count.
        observed_count: usize,
    },
    /// Exact ordered document IDs differ across arms before timing.
    #[error(
        "QG-6 preflight order mismatch for query {query_id:?} at rank \
         {first_differing_rank}: {expected_arm:?} doc={expected_doc_sha256}, \
         {observed_arm:?} doc={observed_doc_sha256}"
    )]
    OrderedDocIdsMismatch {
        /// Stable query ID.
        query_id: String,
        /// Baseline arm.
        expected_arm: Qg6ArmRole,
        /// Compared arm.
        observed_arm: Qg6ArmRole,
        /// First zero-based differing rank.
        first_differing_rank: usize,
        /// SHA-256 of the baseline document ID.
        expected_doc_sha256: String,
        /// SHA-256 of the compared document ID.
        observed_doc_sha256: String,
    },
    /// An explicit semantic-parity proof rejected one preflight comparison.
    #[error(
        "QG-6 preflight semantic-parity failure for query {query_id:?}: \
         {expected_arm:?} vs {observed_arm:?}, sha256={error_sha256}, bytes={error_bytes}"
    )]
    SemanticParityFailure {
        /// Stable query ID.
        query_id: String,
        /// Baseline arm.
        expected_arm: Qg6ArmRole,
        /// Compared arm.
        observed_arm: Qg6ArmRole,
        /// SHA-256 of the bounded adapter diagnostic.
        error_sha256: String,
        /// Diagnostic byte length.
        error_bytes: usize,
    },
    /// A warmup or timed result drifted from its accepted preflight receipt.
    #[error(
        "QG-6 {phase:?} result drift for {arm:?}, query {query_id:?}: \
         expected count/digest={expected_count}/{expected_sha256}, \
         observed={observed_count}/{observed_sha256}"
    )]
    ResultDrift {
        /// Warmup, measurement, or postflight.
        phase: Qg6Phase,
        /// Logical arm.
        arm: Qg6ArmRole,
        /// Stable query ID.
        query_id: String,
        /// Accepted preflight count.
        expected_count: usize,
        /// Observed count.
        observed_count: usize,
        /// Accepted preflight digest.
        expected_sha256: String,
        /// Observed digest.
        observed_sha256: String,
    },
    /// Final lifecycle counts do not match the declared experiment.
    #[error("invalid QG-6 lifecycle receipt: {reason}")]
    LifecycleViolation {
        /// Bounded reason.
        reason: String,
    },
}

struct Qg6SixArms<A> {
    tantivy_null_left: A,
    tantivy_null_right: A,
    quill_null_left: A,
    quill_null_right: A,
    effect_control: A,
    effect_treatment: A,
}

impl<A> Qg6SixArms<A> {
    fn get(&self, role: Qg6ArmRole) -> &A {
        match role {
            Qg6ArmRole::TantivyNullLeft => &self.tantivy_null_left,
            Qg6ArmRole::TantivyNullRight => &self.tantivy_null_right,
            Qg6ArmRole::QuillNullLeft => &self.quill_null_left,
            Qg6ArmRole::QuillNullRight => &self.quill_null_right,
            Qg6ArmRole::EffectControl => &self.effect_control,
            Qg6ArmRole::EffectTreatment => &self.effect_treatment,
        }
    }
}

/// Six independently built arms before result parity has been established.
pub struct Qg6PreparedExperiment<A> {
    identity: Qg6ExperimentIdentity,
    queries: Vec<Qg6QuerySpec>,
    arms: Qg6SixArms<A>,
    lifecycle: Qg6LifecycleReceipt,
}

/// Prepared arms whose complete frozen query set passed result parity.
pub struct Qg6ValidatedExperiment<A> {
    prepared: Qg6PreparedExperiment<A>,
    expected_results: Vec<Qg6SixArmResultReceipts>,
}

impl<A> Qg6PreparedExperiment<A> {
    /// Build, populate, and commit six independent arms exactly once.
    ///
    /// The builder must record every population batch and the searchable
    /// commit through [`Qg6SetupRecorder`]. All six builders receive the same
    /// immutable identity.
    ///
    /// # Errors
    ///
    /// Rejects malformed hashes, query manifests, result limits, adapter
    /// failures, population mismatches, and missing or duplicate commits.
    pub fn prepare_with<F>(
        corpus_sha256: impl Into<String>,
        config_contract_sha256: impl Into<String>,
        document_count: u64,
        k: usize,
        queries: Vec<Qg6QuerySpec>,
        mut build: F,
    ) -> Result<Self, Qg6HarnessError>
    where
        F: FnMut(
            Qg6ArmRole,
            &Qg6ExperimentIdentity,
            &mut Qg6SetupRecorder<'_>,
        ) -> Result<A, String>,
    {
        validate_experiment_inputs(document_count, k, &queries)?;
        let mut queries = queries;
        queries.sort_unstable_by(|left, right| left.id.cmp(&right.id));
        let corpus_sha256 = corpus_sha256.into();
        let config_contract_sha256 = config_contract_sha256.into();
        if !is_lower_hex_sha256(&corpus_sha256) || !is_lower_hex_sha256(&config_contract_sha256) {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "corpus and configuration identities must be lowercase SHA-256 values"
                    .to_owned(),
            });
        }
        let identity = Qg6ExperimentIdentity {
            corpus_sha256,
            query_manifest_sha256: query_manifest_sha256(&queries),
            config_contract_sha256,
            document_count,
            k,
        };
        let mut lifecycle = Qg6LifecycleReceipt::default();
        let tantivy_null_left = build_one(
            Qg6ArmRole::TantivyNullLeft,
            &identity,
            &mut lifecycle,
            &mut build,
        )?;
        let tantivy_null_right = build_one(
            Qg6ArmRole::TantivyNullRight,
            &identity,
            &mut lifecycle,
            &mut build,
        )?;
        let quill_null_left = build_one(
            Qg6ArmRole::QuillNullLeft,
            &identity,
            &mut lifecycle,
            &mut build,
        )?;
        let quill_null_right = build_one(
            Qg6ArmRole::QuillNullRight,
            &identity,
            &mut lifecycle,
            &mut build,
        )?;
        let effect_control = build_one(
            Qg6ArmRole::EffectControl,
            &identity,
            &mut lifecycle,
            &mut build,
        )?;
        let effect_treatment = build_one(
            Qg6ArmRole::EffectTreatment,
            &identity,
            &mut lifecycle,
            &mut build,
        )?;
        Ok(Self {
            identity,
            queries,
            arms: Qg6SixArms {
                tantivy_null_left,
                tantivy_null_right,
                quill_null_left,
                quill_null_right,
                effect_control,
                effect_treatment,
            },
            lifecycle,
        })
    }

    /// Validate every frozen query against all six arms before timing.
    ///
    /// # Errors
    ///
    /// Stops at the first adapter, digest, hit-count, or exact-order failure.
    pub fn validate_exact_parity<F>(
        self,
        search: &mut F,
    ) -> Result<Qg6ValidatedExperiment<A>, Qg6HarnessError>
    where
        F: FnMut(&A, &Qg6QuerySpec, usize) -> Result<Qg6SearchResult, String>,
    {
        self.validate_exact_parity_with(search, &mut |result| result)
    }

    /// Validate exact parity while normalizing engine-native result types
    /// outside any future timing boundary.
    ///
    /// # Errors
    ///
    /// Stops at the first adapter, digest, hit-count, or exact-order failure.
    pub fn validate_exact_parity_with<R, F, N>(
        mut self,
        search: &mut F,
        normalize: &mut N,
    ) -> Result<Qg6ValidatedExperiment<A>, Qg6HarnessError>
    where
        F: FnMut(&A, &Qg6QuerySpec, usize) -> Result<R, String>,
        N: FnMut(R) -> Qg6SearchResult,
    {
        let mut expected_results = Vec::with_capacity(self.queries.len());
        for query in &self.queries {
            let tantivy_null_left = invoke_search(
                &self.arms,
                query,
                self.identity.k,
                self.identity.document_count,
                Qg6ArmRole::TantivyNullLeft,
                Qg6Phase::Preflight,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::TantivyNullLeft)
                .preflight_search_calls += 1;
            let tantivy_null_right = invoke_search(
                &self.arms,
                query,
                self.identity.k,
                self.identity.document_count,
                Qg6ArmRole::TantivyNullRight,
                Qg6Phase::Preflight,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::TantivyNullRight)
                .preflight_search_calls += 1;
            let quill_null_left = invoke_search(
                &self.arms,
                query,
                self.identity.k,
                self.identity.document_count,
                Qg6ArmRole::QuillNullLeft,
                Qg6Phase::Preflight,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::QuillNullLeft)
                .preflight_search_calls += 1;
            let quill_null_right = invoke_search(
                &self.arms,
                query,
                self.identity.k,
                self.identity.document_count,
                Qg6ArmRole::QuillNullRight,
                Qg6Phase::Preflight,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::QuillNullRight)
                .preflight_search_calls += 1;
            let effect_control = invoke_search(
                &self.arms,
                query,
                self.identity.k,
                self.identity.document_count,
                Qg6ArmRole::EffectControl,
                Qg6Phase::Preflight,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::EffectControl)
                .preflight_search_calls += 1;
            let effect_treatment = invoke_search(
                &self.arms,
                query,
                self.identity.k,
                self.identity.document_count,
                Qg6ArmRole::EffectTreatment,
                Qg6Phase::Preflight,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::EffectTreatment)
                .preflight_search_calls += 1;
            for (role, observed) in [
                (Qg6ArmRole::TantivyNullRight, &tantivy_null_right),
                (Qg6ArmRole::QuillNullLeft, &quill_null_left),
                (Qg6ArmRole::QuillNullRight, &quill_null_right),
                (Qg6ArmRole::EffectControl, &effect_control),
                (Qg6ArmRole::EffectTreatment, &effect_treatment),
            ] {
                compare_exact(
                    query.id(),
                    Qg6ArmRole::TantivyNullLeft,
                    &tantivy_null_left,
                    role,
                    observed,
                )?;
            }
            expected_results.push(Qg6SixArmResultReceipts {
                tantivy_null_left: tantivy_null_left.receipt,
                tantivy_null_right: tantivy_null_right.receipt,
                quill_null_left: quill_null_left.receipt,
                quill_null_right: quill_null_right.receipt,
                effect_control: effect_control.receipt,
                effect_treatment: effect_treatment.receipt,
            });
        }
        Ok(Qg6ValidatedExperiment {
            prepared: self,
            expected_results,
        })
    }

    /// Validate parity with an explicit untimed semantic proof.
    ///
    /// The preflight adapter may return engine-native score and cutoff-tie
    /// evidence. `normalize` extracts the exact native top-k result whose
    /// per-arm receipt must remain stable during warmup, measurement, and
    /// postflight, while
    /// `compare` proves that each result is semantically equivalent to the
    /// Tantivy null-left baseline. This permits reviewed native tie-order differences
    /// without requiring every engine to return the same external ID at a
    /// top-k cutoff.
    ///
    /// # Errors
    ///
    /// Stops at the first adapter, digest, result-shape, or semantic-parity
    /// failure.
    pub fn validate_semantic_parity_with<R, F, N, C>(
        mut self,
        search: &mut F,
        normalize: &mut N,
        compare: &mut C,
    ) -> Result<Qg6ValidatedExperiment<A>, Qg6HarnessError>
    where
        F: FnMut(&A, &Qg6QuerySpec, usize) -> Result<R, String>,
        N: FnMut(&R) -> Qg6SearchResult,
        C: FnMut(&Qg6QuerySpec, Qg6ArmRole, &R, Qg6ArmRole, &R) -> Result<(), String>,
    {
        let mut expected_results = Vec::with_capacity(self.queries.len());
        for query in &self.queries {
            let (tantivy_null_left_native, tantivy_null_left) = invoke_search_borrowed(
                &self.arms,
                query,
                self.identity.k,
                self.identity.document_count,
                Qg6ArmRole::TantivyNullLeft,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::TantivyNullLeft)
                .preflight_search_calls += 1;
            let (tantivy_null_right_native, tantivy_null_right) = invoke_search_borrowed(
                &self.arms,
                query,
                self.identity.k,
                self.identity.document_count,
                Qg6ArmRole::TantivyNullRight,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::TantivyNullRight)
                .preflight_search_calls += 1;
            let (quill_null_left_native, quill_null_left) = invoke_search_borrowed(
                &self.arms,
                query,
                self.identity.k,
                self.identity.document_count,
                Qg6ArmRole::QuillNullLeft,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::QuillNullLeft)
                .preflight_search_calls += 1;
            let (quill_null_right_native, quill_null_right) = invoke_search_borrowed(
                &self.arms,
                query,
                self.identity.k,
                self.identity.document_count,
                Qg6ArmRole::QuillNullRight,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::QuillNullRight)
                .preflight_search_calls += 1;
            let (effect_control_native, effect_control) = invoke_search_borrowed(
                &self.arms,
                query,
                self.identity.k,
                self.identity.document_count,
                Qg6ArmRole::EffectControl,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::EffectControl)
                .preflight_search_calls += 1;
            let (effect_treatment_native, effect_treatment) = invoke_search_borrowed(
                &self.arms,
                query,
                self.identity.k,
                self.identity.document_count,
                Qg6ArmRole::EffectTreatment,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::EffectTreatment)
                .preflight_search_calls += 1;
            for (observed_role, observed) in [
                (Qg6ArmRole::TantivyNullRight, &tantivy_null_right_native),
                (Qg6ArmRole::QuillNullLeft, &quill_null_left_native),
                (Qg6ArmRole::QuillNullRight, &quill_null_right_native),
                (Qg6ArmRole::EffectControl, &effect_control_native),
                (Qg6ArmRole::EffectTreatment, &effect_treatment_native),
            ] {
                compare(
                    query,
                    Qg6ArmRole::TantivyNullLeft,
                    &tantivy_null_left_native,
                    observed_role,
                    observed,
                )
                .map_err(|error| {
                    semantic_parity_failure(
                        query.id(),
                        Qg6ArmRole::TantivyNullLeft,
                        observed_role,
                        &error,
                    )
                })?;
            }
            compare_exact(
                query.id(),
                Qg6ArmRole::TantivyNullLeft,
                &tantivy_null_left,
                Qg6ArmRole::TantivyNullRight,
                &tantivy_null_right,
            )?;
            compare_exact(
                query.id(),
                Qg6ArmRole::QuillNullLeft,
                &quill_null_left,
                Qg6ArmRole::QuillNullRight,
                &quill_null_right,
            )?;
            expected_results.push(Qg6SixArmResultReceipts {
                tantivy_null_left: tantivy_null_left.receipt,
                tantivy_null_right: tantivy_null_right.receipt,
                quill_null_left: quill_null_left.receipt,
                quill_null_right: quill_null_right.receipt,
                effect_control: effect_control.receipt,
                effect_treatment: effect_treatment.receipt,
            });
        }
        Ok(Qg6ValidatedExperiment {
            prepared: self,
            expected_results,
        })
    }
}

impl<A> Qg6ValidatedExperiment<A> {
    /// Freeze and seal the exact schedule that a later measurement must consume.
    ///
    /// The caller retains the returned authority and passes it by reference to
    /// measurement. This call performs no search and occurs before timing.
    ///
    /// # Errors
    ///
    /// Rejects invalid query, round, or timing-leaf cardinality.
    pub fn schedule_authority(
        &self,
        rounds_per_query: usize,
        searches_per_sample: usize,
        schedule_seed: u64,
    ) -> Result<Qg6ScheduleAuthority, Qg6HarnessError> {
        Qg6ScheduleAuthority::for_experiment(
            self.prepared.identity.clone(),
            self.prepared.queries.len(),
            rounds_per_query,
            searches_per_sample,
            schedule_seed,
        )
    }

    /// Run equal warmups and an interleaved, balanced six-arm timing schedule.
    ///
    /// Result digest computation and stability checks occur after each timed
    /// interval. The engine adapter receives only an already prepared arm.
    ///
    /// # Errors
    ///
    /// Rejects invalid run counts, adapter failures, post-preflight result
    /// drift, or any lifecycle-count mismatch.
    pub fn measure<F>(
        self,
        warmup_rounds: usize,
        authority: &Qg6ScheduleAuthority,
        search: &mut F,
    ) -> Result<Qg6Measurement, Qg6HarnessError>
    where
        F: FnMut(&A, &Qg6QuerySpec, usize, Qg6Phase) -> Result<Qg6SearchResult, String>,
    {
        self.measure_with_normalizer(warmup_rounds, authority, search, &mut |result| result)
    }

    /// Run the prepared measurement while converting engine-native results
    /// only after each timed interval has ended.
    ///
    /// # Errors
    ///
    /// Rejects invalid run counts, adapter failures, post-preflight result
    /// drift, or any lifecycle-count mismatch.
    pub fn measure_with_normalizer<R, F, N>(
        self,
        warmup_rounds: usize,
        authority: &Qg6ScheduleAuthority,
        search: &mut F,
        normalize: &mut N,
    ) -> Result<Qg6Measurement, Qg6HarnessError>
    where
        F: FnMut(&A, &Qg6QuerySpec, usize, Qg6Phase) -> Result<R, String>,
        N: FnMut(R) -> Qg6SearchResult,
    {
        self.measure_query_p50_with_normalizer(warmup_rounds, authority, search, normalize)
    }

    /// Run the prepared measurement with a fixed per-arm search subsample.
    ///
    /// Every search receives its own monotonic interval and post-interval
    /// result-stability check. One scheduled arm sample reports the median of
    /// those individual intervals, preserving the QG-6 p50 estimand while
    /// making sub-millisecond A/A controls statistically resolvable.
    ///
    /// # Errors
    ///
    /// Rejects zero-sized subsamples in addition to the errors documented by
    /// [`Self::measure_with_normalizer`].
    pub fn measure_query_p50_with_normalizer<R, F, N>(
        mut self,
        warmup_rounds: usize,
        authority: &Qg6ScheduleAuthority,
        search: &mut F,
        normalize: &mut N,
    ) -> Result<Qg6Measurement, Qg6HarnessError>
    where
        F: FnMut(&A, &Qg6QuerySpec, usize, Qg6Phase) -> Result<R, String>,
        N: FnMut(R) -> Qg6SearchResult,
    {
        if warmup_rounds == 0 {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "QG-6 prepared measurement requires at least one warmup per arm and query"
                    .to_owned(),
            });
        }
        authority.verify_for(&self.prepared.identity, self.prepared.queries.len())?;
        let schedule = authority.schedule.clone();
        let rounds_per_query = authority.rounds_per_query;
        let searches_per_sample = authority.searches_per_sample;
        self.run_warmups(warmup_rounds, authority.schedule_seed, search, normalize)?;

        let origin = Instant::now();
        let sample_capacity =
            schedule
                .len()
                .checked_mul(2)
                .ok_or_else(|| Qg6HarnessError::InvalidSpec {
                    reason: "QG-6 timed sample capacity overflow".to_owned(),
                })?;
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(sample_capacity)
            .map_err(|_| Qg6HarnessError::InvalidSpec {
                reason: "QG-6 timed sample capacity allocation failed".to_owned(),
            })?;
        for block in &schedule {
            let query = &self.prepared.queries[block.query_index];
            for (order, role) in [
                (Qg6SampleOrder::First, block.first),
                (Qg6SampleOrder::Second, block.second),
            ] {
                let sample_id = block
                    .block_id
                    .checked_mul(2)
                    .and_then(|base| base.checked_add(u64::from(order == Qg6SampleOrder::Second)))
                    .ok_or_else(|| Qg6HarnessError::InvalidSpec {
                        reason: "timed sample ID overflow".to_owned(),
                    })?;
                let started_ns = monotonic_ns(origin);
                let mut latencies_ns = Vec::new();
                latencies_ns
                    .try_reserve_exact(searches_per_sample)
                    .map_err(|_| Qg6HarnessError::InvalidSpec {
                        reason: "QG-6 latency subsample allocation failed".to_owned(),
                    })?;
                let mut timing_leaves = Vec::new();
                timing_leaves
                    .try_reserve_exact(searches_per_sample)
                    .map_err(|_| Qg6HarnessError::InvalidSpec {
                        reason: "QG-6 timing-leaf allocation failed".to_owned(),
                    })?;
                for _ in 0..searches_per_sample {
                    let search_started_ns = monotonic_ns(origin);
                    let result = search(
                        self.prepared.arms.get(role),
                        black_box(query),
                        black_box(self.prepared.identity.k),
                        Qg6Phase::Measurement,
                    );
                    let search_ended_ns = monotonic_ns(origin);
                    if search_ended_ns <= search_started_ns {
                        return Err(Qg6HarnessError::InvalidSpec {
                            reason: "QG-6 timed search has an invalid monotonic interval"
                                .to_owned(),
                        });
                    }
                    self.prepared.lifecycle.arm_mut(role).timed_search_calls += 1;
                    let result = result.map_err(|error| {
                        adapter_failure(Qg6Phase::Measurement, role, query.id(), &error)
                    })?;
                    let observed = observe_result(
                        normalize(result),
                        self.prepared.identity.k,
                        self.prepared.identity.document_count,
                        Qg6Phase::Measurement,
                        role,
                        query.id(),
                    )?;
                    ensure_stable(
                        Qg6Phase::Measurement,
                        role,
                        query.id(),
                        self.expected_results[block.query_index].get(role),
                        &observed.receipt,
                    )?;
                    let leaf = Qg6SearchTimingLeafReceipt::from_interval(
                        search_started_ns,
                        search_ended_ns,
                    )?;
                    latencies_ns.push(leaf.observed_latency_ns());
                    timing_leaves.push(leaf);
                }
                let ended_ns = monotonic_ns(origin);
                if ended_ns <= started_ns {
                    return Err(Qg6HarnessError::InvalidSpec {
                        reason: "QG-6 timed sample has an invalid monotonic interval".to_owned(),
                    });
                }
                latencies_ns.sort_unstable();
                let observed_latency_ns = median_sorted_u64(&latencies_ns);
                let subsample_count = usize_to_u64(searches_per_sample)?;
                let mut sample = Qg6TimedSample {
                    block_id: block.block_id,
                    sample_id,
                    query_id: query.id().to_owned(),
                    query_index: block.query_index,
                    comparison: block.comparison,
                    arm: role,
                    order,
                    started_ns,
                    ended_ns,
                    observed_latency_ns,
                    subsample_count,
                    result_receipt_sha256: self.expected_results[block.query_index]
                        .get(role)
                        .receipt_sha256
                        .clone(),
                    result_sha256: qg6_result_sequence_sha256(
                        self.expected_results[block.query_index].get(role),
                        subsample_count,
                    )?,
                    timing_leaves,
                    timing_leaves_sha256: String::new(),
                };
                sample.timing_leaves_sha256 = sample.recomputed_timing_leaves_sha256()?;
                sample.verify_timing_leaves()?;
                samples.push(sample);
            }
        }
        self.run_postflight(search, normalize)?;
        let semantic_contract = build_semantic_contract(
            &self.prepared.identity,
            &self.prepared.queries,
            &self.expected_results,
        )?;
        verify_lifecycle(
            &self.prepared.lifecycle,
            self.prepared.identity.document_count,
            self.prepared.queries.len(),
            warmup_rounds,
            rounds_per_query,
            searches_per_sample,
        )?;
        let measurement = Qg6Measurement {
            identity: self.prepared.identity,
            schedule_seed: authority.schedule_seed,
            warmup_rounds,
            rounds_per_query,
            searches_per_sample,
            schedule,
            schedule_authority: authority.clone(),
            samples,
            lifecycle: self.prepared.lifecycle,
            semantic_contract,
        };
        measurement.verify_against_schedule_authority(authority)?;
        Ok(measurement)
    }

    fn run_warmups<R, F, N>(
        &mut self,
        warmup_rounds: usize,
        schedule_seed: u64,
        search: &mut F,
        normalize: &mut N,
    ) -> Result<(), Qg6HarnessError>
    where
        F: FnMut(&A, &Qg6QuerySpec, usize, Qg6Phase) -> Result<R, String>,
        N: FnMut(R) -> Qg6SearchResult,
    {
        for round in 0..warmup_rounds {
            for (query_index, query) in self.prepared.queries.iter().enumerate() {
                let mut roles = Qg6ArmRole::ALL;
                let salt = usize_to_u64(round)?.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    ^ usize_to_u64(query_index)?;
                shuffle(&mut roles, schedule_seed ^ salt);
                for role in roles {
                    let observed = invoke_phased_search(
                        &self.prepared.arms,
                        query,
                        self.prepared.identity.k,
                        self.prepared.identity.document_count,
                        role,
                        Qg6Phase::Warmup,
                        search,
                        normalize,
                    )?;
                    self.prepared.lifecycle.arm_mut(role).warmup_search_calls += 1;
                    ensure_stable(
                        Qg6Phase::Warmup,
                        role,
                        query.id(),
                        self.expected_results[query_index].get(role),
                        &observed.receipt,
                    )?;
                    black_box(observed);
                }
            }
        }
        Ok(())
    }

    fn run_postflight<R, F, N>(
        &mut self,
        search: &mut F,
        normalize: &mut N,
    ) -> Result<(), Qg6HarnessError>
    where
        F: FnMut(&A, &Qg6QuerySpec, usize, Qg6Phase) -> Result<R, String>,
        N: FnMut(R) -> Qg6SearchResult,
    {
        for (query_index, query) in self.prepared.queries.iter().enumerate() {
            for role in Qg6ArmRole::ALL {
                let observed = invoke_phased_search(
                    &self.prepared.arms,
                    query,
                    self.prepared.identity.k,
                    self.prepared.identity.document_count,
                    role,
                    Qg6Phase::Postflight,
                    search,
                    normalize,
                )?;
                self.prepared
                    .lifecycle
                    .arm_mut(role)
                    .postflight_search_calls += 1;
                ensure_stable(
                    Qg6Phase::Postflight,
                    role,
                    query.id(),
                    self.expected_results[query_index].get(role),
                    &observed.receipt,
                )?;
            }
        }
        Ok(())
    }
}

/// Construct the deterministic schedule used by the prepared QG-6 runner.
///
/// Every query receives `rounds_per_query` blocks for each formal comparison.
/// Each three-block unit contains Tantivy/Tantivy, Quill/Quill, and effect
/// comparisons. All six comparison permutations and every within-pair order
/// are independently balanced.
///
/// # Errors
///
/// Requires at least one query and two rounds per query, and rejects arithmetic
/// overflow.
pub fn seeded_interleaved_six_arm_schedule(
    query_count: usize,
    rounds_per_query: usize,
    seed: u64,
) -> Result<Vec<Qg6PairBlock>, Qg6HarnessError> {
    if query_count == 0 || query_count > MAX_QUERY_COUNT {
        return Err(Qg6HarnessError::InvalidSpec {
            reason: "QG-6 schedule query count is outside 1..=4096".to_owned(),
        });
    }
    if rounds_per_query < 2 {
        return Err(Qg6HarnessError::InvalidSpec {
            reason: "QG-6 schedule requires at least two rounds per query".to_owned(),
        });
    }
    let unit_count =
        query_count
            .checked_mul(rounds_per_query)
            .ok_or_else(|| Qg6HarnessError::InvalidSpec {
                reason: "QG-6 schedule unit count overflow".to_owned(),
            })?;
    let mut query_units = Vec::new();
    query_units
        .try_reserve_exact(unit_count)
        .map_err(|_| Qg6HarnessError::InvalidSpec {
            reason: "QG-6 schedule unit allocation failed".to_owned(),
        })?;
    for _ in 0..rounds_per_query {
        query_units.extend(0..query_count);
    }
    shuffle(&mut query_units, seed ^ 0x8d58_ac26_afe1_2e47);
    let mut comparison_permutations = (0..unit_count).map(|index| index % 6).collect::<Vec<_>>();
    shuffle(&mut comparison_permutations, seed ^ 0x243f_6a88_85a3_08d3);
    let tantivy_null_left_first = balanced_bools(unit_count, seed ^ 0x1319_8a2e_0370_7344)?;
    let quill_null_left_first = balanced_bools(unit_count, seed ^ 0x082e_fa98_ec4e_6c89)?;
    let effect_control_first = balanced_bools(unit_count, seed ^ 0xa409_3822_299f_31d0)?;
    let block_capacity = unit_count
        .checked_mul(3)
        .ok_or_else(|| Qg6HarnessError::InvalidSpec {
            reason: "QG-6 block count overflow".to_owned(),
        })?;
    let mut schedule = Vec::new();
    schedule
        .try_reserve_exact(block_capacity)
        .map_err(|_| Qg6HarnessError::InvalidSpec {
            reason: "QG-6 schedule block allocation failed".to_owned(),
        })?;
    for (unit_index, query_index) in query_units.into_iter().enumerate() {
        let tantivy_null = pair_roles(
            Qg6Comparison::TantivyNull,
            tantivy_null_left_first[unit_index],
        );
        let quill_null = pair_roles(Qg6Comparison::QuillNull, quill_null_left_first[unit_index]);
        let effect = pair_roles(Qg6Comparison::Effect, effect_control_first[unit_index]);
        let pairs = match comparison_permutations[unit_index] {
            0 => [tantivy_null, quill_null, effect],
            1 => [tantivy_null, effect, quill_null],
            2 => [quill_null, tantivy_null, effect],
            3 => [quill_null, effect, tantivy_null],
            4 => [effect, tantivy_null, quill_null],
            5 => [effect, quill_null, tantivy_null],
            _ => unreachable!("QG-6 comparison permutation is modulo six"),
        };
        for (comparison, first, second) in pairs {
            schedule.push(Qg6PairBlock {
                block_id: usize_to_u64(schedule.len())?,
                unit_id: usize_to_u64(unit_index)?,
                query_index,
                comparison,
                first,
                second,
            });
        }
    }
    Ok(schedule)
}

const QG6_R1_RESIDUAL_ROLE_COUNT: usize = 6;
const QG6_R1_RESIDUAL_ROLE_COUNT_U8: u8 = 6;
const QG6_R1_RESIDUAL_LEAF_COUNT: usize = QG6_R1_RESIDUAL_ROLE_COUNT * QG6_R1_RESIDUAL_ROLE_COUNT;
const QG6_R1_RESIDUAL_LEAF_COUNT_U8: u8 = 36;
const QG6_R1_RESIDUAL_LEAF_COUNT_U64: u64 = 36;
const QG6_R1_RESIDUAL_WILLIAMS_BASE_ROW: [u8; QG6_R1_RESIDUAL_ROLE_COUNT] = [0, 1, 5, 2, 4, 3];
const QG6_R1_RESIDUAL_SOURCE_ELF_CONSISTENCY_VERSION: &str =
    "frankensearch-qg6-r1-residual-source-elf-consistency-v1";
const QG6_R1_RESIDUAL_STANDARDIZED_WORKLOAD_VERSION: &str =
    "frankensearch-qg6-r1-residual-standardized-workload-v1";
const QG6_R1_RESIDUAL_RANKED_MISS_SEMANTICS_VERSION: &str =
    "frankensearch-qg6-r1-residual-ranked-miss-semantics-v1";
const QG6_R1_RESIDUAL_BOUNDARY_EFFECT_VERSION: &str =
    "frankensearch-qg6-r1-residual-boundary-effect-v1";
const QG6_R1_RESIDUAL_INVOCATION_VERSION: &str = "frankensearch-qg6-r1-residual-invocation-v1";
const QG6_R1_RESIDUAL_REBIND_TRANSITION_VERSION: &str =
    "frankensearch-qg6-r1-residual-rebind-transition-v1";
const QG6_R1_RESIDUAL_COMPLETED_OUTCOME_CODE: &str = "completed";
const QG6_R1_RESIDUAL_ROLE_COUNT_F64: f64 = 6.0;
const QG6_R1_RESIDUAL_MAX_LATENCY: Duration = Duration::from_secs(60);

/// Six physically independent timing roles for the QG-6 R1 residual diagnostic.
///
/// This is deliberately separate from [`Qg6ArmRole`]:
/// the latter is the six-arm formal QG-6 gate and cannot be relabelled into
/// this diagnostic's Old/Current/Tantivy A/A and A/B topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg6ResidualArmRole {
    /// First independently constructed exact-control instance.
    OldA,
    /// Second independently constructed exact-control instance.
    OldB,
    /// First independently constructed reviewed-current instance.
    CurrentA,
    /// Second independently constructed reviewed-current instance.
    CurrentB,
    /// First independently constructed pinned-Tantivy instance.
    TantivyA,
    /// Second independently constructed pinned-Tantivy instance.
    TantivyB,
}

impl Qg6ResidualArmRole {
    /// Canonical role order used by the Williams design and contrast vector.
    pub const ALL: [Self; QG6_R1_RESIDUAL_ROLE_COUNT] = [
        Self::OldA,
        Self::OldB,
        Self::CurrentA,
        Self::CurrentB,
        Self::TantivyA,
        Self::TantivyB,
    ];

    const fn index(self) -> usize {
        match self {
            Self::OldA => 0,
            Self::OldB => 1,
            Self::CurrentA => 2,
            Self::CurrentB => 3,
            Self::TantivyA => 4,
            Self::TantivyB => 5,
        }
    }
}

/// Separate cache/lifecycle strata.
///
/// A meta-block may contain one stratum only;
/// mixing first-touch, ranked miss, and generation-rebind observations makes a
/// residual estimate uninterpretable and is rejected before estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg6ResidualStratum {
    /// Fresh six-arm instances before any query has touched them.
    FirstTouch,
    /// A semantics-preserving nonce established an exact ranked-cache miss.
    SteadyRankedCacheMiss,
    /// A matched mutation/delete schedule completed before observation.
    GenerationRebind,
}

/// Observed cache disposition compatible with one residual stratum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg6ResidualCacheDisposition {
    /// The invocation is the stratum's first touch.
    FirstTouch,
    /// The invocation proved a ranked-cache miss without disabling the cache.
    RankedMiss,
    /// The ranked-cache lookup returned a prior result. This is never
    /// estimable in a steady ranked-miss meta-block.
    RankedHit,
    /// The invocation occurred after the matched rebind schedule.
    GenerationRebind,
}

/// Evidence that a generation-rebind leaf followed one real, parity-checked
/// mutation transition. Every digest is issued by the producer; admission
/// verifies its internal binding and then matches it to the opaque producer
/// authority rather than treating the leaf's copy as authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg6ResidualGenerationRebindEvidence {
    /// Generation observed immediately before the matched mutation.
    pub pre_generation: u64,
    /// Digest of the physical backing bytes immediately before the mutation.
    pub pre_backing_sha256: String,
    /// Frozen matched mutation/delete plan shared by all six physical arms.
    pub mutation_plan_sha256: String,
    /// Receipt for the actual mutation/delete operation.
    pub mutation_receipt_sha256: String,
    /// Receipt proving semantic parity after the rebind.
    pub parity_receipt_sha256: String,
    /// Domain-separated binding across the pre/post transition.
    pub transition_receipt_sha256: String,
}

impl Qg6ResidualStratum {
    const fn expected_cache_disposition(self) -> Qg6ResidualCacheDisposition {
        match self {
            Self::FirstTouch => Qg6ResidualCacheDisposition::FirstTouch,
            Self::SteadyRankedCacheMiss => Qg6ResidualCacheDisposition::RankedMiss,
            Self::GenerationRebind => Qg6ResidualCacheDisposition::GenerationRebind,
        }
    }
}

/// One predeclared position in a complete six-sweep Williams meta-block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg6ResidualWilliamsLeaf {
    /// Globally unique leaf identifier for this meta-block.
    pub leaf_id: u64,
    /// Caller-provided meta-block identity.
    pub meta_block_id: u64,
    /// Williams design row in the range `0..6`.
    pub sweep: u8,
    /// Ordinal within one six-role sweep in the range `0..6`.
    pub ordinal: u8,
    /// The independently constructed physical arm to invoke at this position.
    pub role: Qg6ResidualArmRole,
}

/// Raw, exactly-once timing observation for one residual Williams leaf.
///
/// Every field is a bounded identifier or a monotonic timing value. Raw query,
/// corpus, path, and result payload bytes never enter this in-memory contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg6ResidualLeafObservation {
    /// Scheduled leaf identity.
    pub leaf_id: u64,
    /// Meta-block identity.
    pub meta_block_id: u64,
    /// Scheduled Williams row.
    pub sweep: u8,
    /// Scheduled ordinal.
    pub ordinal: u8,
    /// Scheduled physical arm.
    pub role: Qg6ResidualArmRole,
    /// Lifecycle stratum; all leaves in one admitted block must agree.
    pub stratum: Qg6ResidualStratum,
    /// Public query class; all leaves in one admitted block must agree.
    pub query_class: PerfQueryClass,
    /// Bounded, non-sensitive local query identifier.
    pub query_id: String,
    /// Receipt binding this physical instance to its independent construction.
    pub instance_receipt_sha256: String,
    /// Digest of the backing bytes. Equal byte content is expected for each
    /// same-source A/A pair and is deliberately not a physical identity.
    pub backing_sha256: String,
    /// Receipt for the physical backing allocation/mapping. Unlike
    /// [`Self::backing_sha256`], this must distinguish independently built
    /// physical backings even when their bytes are identical.
    pub backing_instance_receipt_sha256: String,
    /// Receipt for the physical arm's ranked-cache namespace. This must be
    /// independent across all six arms even when the stratum is first touch.
    pub ranked_cache_receipt_sha256: String,
    /// Digest of the resolved storage path identity, never a raw filesystem path.
    pub path_receipt_sha256: String,
    /// Exact source revision digest for this physical arm.
    pub source_sha256: String,
    /// Exact Cargo.lock digest for this physical arm.
    pub cargo_lock_sha256: String,
    /// Self-reported timing ELF digest for this physical arm.
    pub timing_elf_sha256: String,
    /// Domain-separated cross-field consistency digest. This detects an
    /// in-record source/lockfile/ELF mix-up; it is not source-build proof.
    pub source_elf_consistency_sha256: String,
    /// Opaque receipt from the independently frozen source-to-build authority.
    /// The receipt is compared with the authority supplied to admission; it is
    /// never derived from fields carried by this leaf.
    pub source_build_receipt_sha256: String,
    /// Fixture identity shared by all six arms.
    pub fixture_sha256: String,
    /// Query-contract identity shared by all six arms.
    pub query_contract_sha256: String,
    /// SHA-256 of the parsed query AST. A ranked-cache nonce may change raw
    /// bytes but must not change this semantic query identity.
    pub parsed_ast_sha256: String,
    /// Digest of the runner-defined, backend-neutral workload: exactly one
    /// timed dispatch of `query_id` under `query_contract_sha256`. This is
    /// recomputed at admission and is not an engine work counter.
    pub standardized_workload_sha256: String,
    /// Exact result-envelope digest after the timed invocation.
    pub result_envelope_sha256: String,
    /// SHA-256 of the actual ranked-cache key used by this invocation.
    pub ranked_cache_key_sha256: String,
    /// SHA-256 of the fixed-length, semantics-preserving raw-query nonce.
    pub ranked_cache_nonce_sha256: String,
    /// SHA-256 of the nonce-bearing raw query handed to the parser.
    pub raw_query_sha256: String,
    /// Raw query length after nonce insertion. It must be constant within a
    /// steady ranked-miss meta-block.
    pub raw_query_length_bytes: u16,
    /// Recomputable binding proving the nonce/cache-key pair retained the
    /// parsed AST, result envelope, work, fuel, and cancellation semantics.
    pub ranked_miss_semantics_sha256: String,
    /// Observed cache state; cache disabling cannot masquerade as a miss.
    pub cache_disposition: Qg6ResidualCacheDisposition,
    /// Generation witness after the stratum's lifecycle boundary.
    pub generation: u64,
    /// Mandatory backend-neutral completed work count for the one dispatch.
    pub work_units: u64,
    /// Predeclared fuel budget for this invocation.
    pub fuel_budget: u64,
    /// Fuel consumed by this invocation.
    pub fuel_consumed: u64,
    /// Whether cancellation was observed during this invocation.
    pub cancelled: bool,
    /// Stable typed outcome code. Only `completed` is estimable.
    pub outcome_code: String,
    /// Receipt from the cache lookup which named this leaf's disposition.
    pub ranked_cache_lookup_receipt_sha256: String,
    /// Receipt for the observed host, affinity, and scheduler identity.
    pub host_receipt_sha256: String,
    /// Receipt for the boot instance which owns the host observation.
    pub boot_receipt_sha256: String,
    /// Receipt for the monotonic clock identity and sampling contract.
    pub clock_receipt_sha256: String,
    /// Domain-separated binding from the real invocation to query/cache,
    /// result, work/outcome, host/boot, clock, and monotonic interval facts.
    pub invocation_receipt_sha256: String,
    /// Rebind proof when and only when this leaf uses the rebind stratum.
    pub generation_rebind_evidence: Option<Qg6ResidualGenerationRebindEvidence>,
    /// Execution-order leaf immediately before this leaf, if any.
    pub boundary_predecessor_leaf_id: Option<u64>,
    /// Independent-instance receipt of the execution-order predecessor.
    pub boundary_predecessor_instance_receipt_sha256: Option<String>,
    /// Observed boundary effect before this leaf, in nanoseconds.
    pub boundary_effect_ns: u64,
    /// Predeclared inclusive bound for one execution-order boundary effect.
    pub boundary_effect_limit_ns: u64,
    /// Recomputable identity binding the predecessor and bounded effect.
    pub boundary_effect_receipt_sha256: String,
    /// Exact execution position in the randomized six-by-six meta-block.
    pub execution_ordinal: u8,
    /// Monotonic interval start in nanoseconds relative to the meta-block origin.
    pub started_ns: u64,
    /// Monotonic interval end in nanoseconds relative to the meta-block origin.
    pub ended_ns: u64,
    /// Independently recorded latency, required to equal `ended_ns - started_ns`.
    pub latency_ns: u64,
}

/// Five contrasts resampled together by a later hierarchical bootstrap.
///
/// This type deliberately contains one vector per admitted
/// query/meta-block rather than independent per-contrast samples, preserving
/// their covariance for the next estimator stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg6ResidualJointContrastVector {
    /// `mean(log(OldB)) - mean(log(OldA))` A/A null.
    pub old_b_minus_old_a: f64,
    /// `mean(log(CurrentB)) - mean(log(CurrentA))` A/A null.
    pub current_b_minus_current_a: f64,
    /// `mean(log(TantivyB)) - mean(log(TantivyA))` A/A null.
    pub tantivy_b_minus_tantivy_a: f64,
    /// Mean current log latency minus mean old log latency. Maintenance only.
    pub current_mean_minus_old_mean: f64,
    /// Mean current log latency minus mean Tantivy log latency. Diagnostic only.
    pub current_mean_minus_tantivy_mean: f64,
}

/// Schedule-level admission for a provenance-bound meta-block eligible to
/// become one joint bootstrap draw.
///
/// This deliberately does **not** constitute full QG-6 R1 evidence admission:
/// its leaves are source/build-authority-bound and carry host, outcome, work,
/// fuel/cancel, cache, and boundary receipts, but the later hierarchical
/// bootstrap gate still owns any performance decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg6ResidualScheduleAdmission {
    /// Canonical six-by-six Williams design used for admission.
    pub schedule: Vec<Qg6ResidualWilliamsLeaf>,
    /// Exactly one raw observation for each schedule leaf, in scheduled order.
    pub leaves: Vec<Qg6ResidualLeafObservation>,
    /// The five contrasts derived jointly from these same raw leaves.
    pub joint_contrasts: Qg6ResidualJointContrastVector,
}

/// Fail-closed residual-schedule or leaf-admission error.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Qg6ResidualValidationError {
    /// A caller supplied a schedule other than the complete Williams design.
    #[error("QG-6 R1 residual Williams schedule rejected: {reason}")]
    InvalidSchedule {
        /// Bounded reason with no query, path, or corpus payload.
        reason: String,
    },
    /// A raw leaf is malformed, incomplete, duplicated, or arithmetically invalid.
    #[error("QG-6 R1 residual leaf rejected: {reason}")]
    InvalidLeaf {
        /// Bounded reason with no query, path, or corpus payload.
        reason: String,
    },
    /// The six arms do not retain the required physical/provenance separation.
    #[error("QG-6 R1 residual provenance rejected: {reason}")]
    ProvenanceMismatch {
        /// Bounded reason with no query, path, or corpus payload.
        reason: String,
    },
}

/// Non-serializable capability emitted only by the trusted residual runner.
///
/// This deliberately has no production constructor in the prepared-admission
/// module. Until a producer that owns the source/build, parser, cache, clock,
/// and rebind observations is wired in, the public entry point fails closed.
/// A caller cannot replace the capability with coordinated leaf strings.
#[derive(Debug, Clone)]
struct Qg6ResidualProducerAuthority {
    trusted_leaves: BTreeMap<u64, Qg6ResidualLeafObservation>,
}

impl Qg6ResidualProducerAuthority {
    fn validate(
        &self,
        ordered: &[Qg6ResidualLeafObservation],
    ) -> Result<(), Qg6ResidualValidationError> {
        if self.trusted_leaves.len() != QG6_R1_RESIDUAL_LEAF_COUNT {
            return Err(Qg6ResidualValidationError::ProvenanceMismatch {
                reason: "residual producer authority does not cover one complete meta-block"
                    .to_owned(),
            });
        }
        for leaf in ordered {
            if self.trusted_leaves.get(&leaf.leaf_id) != Some(leaf) {
                return Err(Qg6ResidualValidationError::ProvenanceMismatch {
                    reason: "residual leaf disagrees with the independent producer authority"
                        .to_owned(),
                });
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn test_fixture(
        observations: &[Qg6ResidualLeafObservation],
    ) -> Result<Self, Qg6ResidualValidationError> {
        let mut trusted_leaves = BTreeMap::new();
        for leaf in observations {
            if trusted_leaves.insert(leaf.leaf_id, leaf.clone()).is_some() {
                return Err(Qg6ResidualValidationError::InvalidLeaf {
                    reason: "residual producer test fixture contains a duplicate leaf ID"
                        .to_owned(),
                });
            }
        }
        Ok(Self { trusted_leaves })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Qg6ResidualRoleBuildProvenance {
    source: String,
    cargo_lock: String,
    timing_elf: String,
    source_build: String,
}

impl From<&Qg6ResidualLeafObservation> for Qg6ResidualRoleBuildProvenance {
    fn from(leaf: &Qg6ResidualLeafObservation) -> Self {
        Self {
            source: leaf.source_sha256.clone(),
            cargo_lock: leaf.cargo_lock_sha256.clone(),
            timing_elf: leaf.timing_elf_sha256.clone(),
            source_build: leaf.source_build_receipt_sha256.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Qg6ResidualPhysicalArmProvenance {
    instance: String,
    backing: String,
    ranked_cache: String,
    path: String,
}

impl From<&Qg6ResidualLeafObservation> for Qg6ResidualPhysicalArmProvenance {
    fn from(leaf: &Qg6ResidualLeafObservation) -> Self {
        Self {
            instance: leaf.instance_receipt_sha256.clone(),
            backing: leaf.backing_instance_receipt_sha256.clone(),
            ranked_cache: leaf.ranked_cache_receipt_sha256.clone(),
            path: leaf.path_receipt_sha256.clone(),
        }
    }
}

/// Construct the canonical six-sweep Williams design for one meta-block.
///
/// The base row is `[0, 1, 5, 2, 4, 3]`; each following row applies the
/// required cyclic role mapping. The resulting schedule has exactly 36 leaves,
/// each role occupies each ordinal once, every ordered pair is before/after
/// three times, and every directed immediate predecessor occurs once.
///
/// # Errors
///
/// Rejects a meta-block identity whose leaf-ID range would overflow.
pub fn qg6_residual_williams_schedule(
    meta_block_id: u64,
) -> Result<Vec<Qg6ResidualWilliamsLeaf>, Qg6ResidualValidationError> {
    let schedule = canonical_residual_williams_schedule(meta_block_id)?;
    validate_qg6_residual_williams_schedule(&schedule)?;
    Ok(schedule)
}

/// Verify a six-role residual schedule before any physical leaf is observed.
///
/// One role mapping and row/execution randomization are allowed;
/// arbitrary per-row column permutations and prefixes are rejected.
///
/// # Errors
///
/// Rejects noncanonical rows, duplicate/missing leaves, unbalanced order, or
/// an incomplete 36-leaf design.
pub fn validate_qg6_residual_williams_schedule(
    schedule: &[Qg6ResidualWilliamsLeaf],
) -> Result<(), Qg6ResidualValidationError> {
    if schedule.len() != QG6_R1_RESIDUAL_LEAF_COUNT {
        return Err(Qg6ResidualValidationError::InvalidSchedule {
            reason: "Williams meta-block must contain exactly 36 leaves".to_owned(),
        });
    }
    let meta_block_id = schedule[0].meta_block_id;
    let mut cells: [Option<&Qg6ResidualWilliamsLeaf>; QG6_R1_RESIDUAL_LEAF_COUNT] =
        std::array::from_fn(|_| None);
    let mut leaf_ids = BTreeSet::new();
    for leaf in schedule {
        if leaf.meta_block_id != meta_block_id
            || leaf.sweep >= QG6_R1_RESIDUAL_ROLE_COUNT_U8
            || leaf.ordinal >= QG6_R1_RESIDUAL_ROLE_COUNT_U8
            || !leaf_ids.insert(leaf.leaf_id)
        {
            return Err(Qg6ResidualValidationError::InvalidSchedule {
                reason: "Williams schedule has mixed metadata, an out-of-range cell, or duplicate leaf ID"
                    .to_owned(),
            });
        }
        let cell_index =
            usize::from(leaf.sweep) * QG6_R1_RESIDUAL_ROLE_COUNT + usize::from(leaf.ordinal);
        if cells[cell_index].replace(leaf).is_some() {
            return Err(Qg6ResidualValidationError::InvalidSchedule {
                reason: "Williams schedule contains a duplicate sweep/ordinal cell".to_owned(),
            });
        }
    }
    let cells = cells.map(|cell| {
        cell.ok_or_else(|| Qg6ResidualValidationError::InvalidSchedule {
            reason: "Williams schedule omits a sweep/ordinal cell".to_owned(),
        })
    });
    let cells: [&Qg6ResidualWilliamsLeaf; QG6_R1_RESIDUAL_LEAF_COUNT] = cells
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| Qg6ResidualValidationError::InvalidSchedule {
            reason: "Williams schedule cell materialization failed".to_owned(),
        })?;

    let mut role_mapping: [Option<Qg6ResidualArmRole>; QG6_R1_RESIDUAL_ROLE_COUNT] =
        std::array::from_fn(|_| None);
    for ordinal in 0..QG6_R1_RESIDUAL_ROLE_COUNT {
        let canonical_role = usize::from(QG6_R1_RESIDUAL_WILLIAMS_BASE_ROW[ordinal]);
        let observed_role = cells[ordinal].role;
        if role_mapping
            .iter()
            .flatten()
            .any(|role| *role == observed_role)
        {
            return Err(Qg6ResidualValidationError::InvalidSchedule {
                reason: "Williams role randomization is not one bijection for the meta-block"
                    .to_owned(),
            });
        }
        role_mapping[canonical_role] = Some(observed_role);
    }
    let role_mapping = role_mapping.map(|role| {
        role.ok_or_else(|| Qg6ResidualValidationError::InvalidSchedule {
            reason: "Williams role randomization omits a canonical role".to_owned(),
        })
    });
    let role_mapping: [Qg6ResidualArmRole; QG6_R1_RESIDUAL_ROLE_COUNT] = role_mapping
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| Qg6ResidualValidationError::InvalidSchedule {
            reason: "Williams role mapping materialization failed".to_owned(),
        })?;

    for sweep in 0_u8..QG6_R1_RESIDUAL_ROLE_COUNT_U8 {
        for ordinal in 0..QG6_R1_RESIDUAL_ROLE_COUNT {
            let cell = cells[usize::from(sweep) * QG6_R1_RESIDUAL_ROLE_COUNT + ordinal];
            let canonical_role = usize::from(
                (QG6_R1_RESIDUAL_WILLIAMS_BASE_ROW[ordinal] + sweep)
                    % QG6_R1_RESIDUAL_ROLE_COUNT_U8,
            );
            if cell.role != role_mapping[canonical_role] {
                return Err(Qg6ResidualValidationError::InvalidSchedule {
                    reason: "Williams leaves do not preserve one randomized role mapping"
                        .to_owned(),
                });
            }
        }
    }

    let design_order = cells.into_iter().cloned().collect::<Vec<_>>();
    validate_residual_williams_balance(&design_order)
}

/// Construct a domain-separated source/lockfile/ELF consistency digest.
///
/// This is an integrity check for fields already observed by the runner; it
/// does not prove that a self-reported ELF came from a source revision. The
/// later artifact receipt must independently prove that source-build relation.
///
/// # Errors
///
/// Rejects non-lowercase-SHA256 inputs.
pub fn qg6_residual_source_elf_consistency_sha256(
    source_sha256: &str,
    cargo_lock_sha256: &str,
    timing_elf_sha256: &str,
) -> Result<String, Qg6ResidualValidationError> {
    if ![source_sha256, cargo_lock_sha256, timing_elf_sha256]
        .into_iter()
        .all(is_lower_hex_sha256)
    {
        return Err(Qg6ResidualValidationError::ProvenanceMismatch {
            reason: "source, Cargo.lock, and timing ELF identities must be lowercase SHA-256"
                .to_owned(),
        });
    }
    let mut hasher = Sha256::new();
    hash_len_prefixed(
        &mut hasher,
        QG6_R1_RESIDUAL_SOURCE_ELF_CONSISTENCY_VERSION.as_bytes(),
    );
    hash_len_prefixed(&mut hasher, source_sha256.as_bytes());
    hash_len_prefixed(&mut hasher, cargo_lock_sha256.as_bytes());
    hash_len_prefixed(&mut hasher, timing_elf_sha256.as_bytes());
    Ok(lower_hex(hasher.finalize()))
}

/// Construct the backend-neutral workload identity for one residual leaf.
///
/// Each leaf executes one predeclared query contract exactly once. The digest
/// binds that standardized workload before backend dispatch, so it cannot be
/// substituted by an engine-specific counter.
///
/// # Errors
///
/// Rejects a malformed query identifier or query-contract digest.
pub fn qg6_residual_standardized_workload_sha256(
    query_id: &str,
    query_contract_sha256: &str,
) -> Result<String, Qg6ResidualValidationError> {
    if !is_valid_residual_query_id(query_id) || !is_lower_hex_sha256(query_contract_sha256) {
        return Err(Qg6ResidualValidationError::InvalidLeaf {
            reason: "residual standardized workload requires a bounded query ID and SHA-256 query contract"
                .to_owned(),
        });
    }
    let mut hasher = Sha256::new();
    hash_len_prefixed(
        &mut hasher,
        QG6_R1_RESIDUAL_STANDARDIZED_WORKLOAD_VERSION.as_bytes(),
    );
    hash_len_prefixed(&mut hasher, query_id.as_bytes());
    hash_len_prefixed(&mut hasher, query_contract_sha256.as_bytes());
    hasher.update(1_u8.to_le_bytes());
    Ok(lower_hex(hasher.finalize()))
}

fn residual_ranked_miss_semantics_sha256(
    query_id: &str,
    parsed_ast_sha256: &str,
    raw_query_length_bytes: u16,
    ranked_cache_nonce_sha256: &str,
    ranked_cache_key_sha256: &str,
    result_envelope_sha256: &str,
    work_units: u64,
    fuel_budget: u64,
    fuel_consumed: u64,
    cancelled: bool,
    outcome_code: &str,
) -> Result<String, Qg6ResidualValidationError> {
    if !is_valid_residual_query_id(query_id)
        || raw_query_length_bytes == 0
        || usize::from(raw_query_length_bytes) > MAX_QUERY_TEXT_BYTES
        || ![
            parsed_ast_sha256,
            ranked_cache_nonce_sha256,
            ranked_cache_key_sha256,
            result_envelope_sha256,
        ]
        .into_iter()
        .all(is_lower_hex_sha256)
        || outcome_code.is_empty()
        || outcome_code.len() > MAX_UNSUPPORTED_REASON_CODE_BYTES
        || !outcome_code
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || matches!(byte, b'-' | b'_'))
    {
        return Err(Qg6ResidualValidationError::InvalidLeaf {
            reason: "residual ranked-miss semantics binding is malformed".to_owned(),
        });
    }
    let mut hasher = Sha256::new();
    hash_len_prefixed(
        &mut hasher,
        QG6_R1_RESIDUAL_RANKED_MISS_SEMANTICS_VERSION.as_bytes(),
    );
    hash_len_prefixed(&mut hasher, query_id.as_bytes());
    hash_len_prefixed(&mut hasher, parsed_ast_sha256.as_bytes());
    hasher.update(raw_query_length_bytes.to_le_bytes());
    hash_len_prefixed(&mut hasher, ranked_cache_nonce_sha256.as_bytes());
    hash_len_prefixed(&mut hasher, ranked_cache_key_sha256.as_bytes());
    hash_len_prefixed(&mut hasher, result_envelope_sha256.as_bytes());
    hasher.update(work_units.to_le_bytes());
    hasher.update(fuel_budget.to_le_bytes());
    hasher.update(fuel_consumed.to_le_bytes());
    hasher.update(u8::from(cancelled).to_le_bytes());
    hash_len_prefixed(&mut hasher, outcome_code.as_bytes());
    Ok(lower_hex(hasher.finalize()))
}

fn residual_boundary_effect_sha256(
    predecessor_leaf_id: Option<u64>,
    predecessor_instance_receipt_sha256: Option<&str>,
    boundary_effect_ns: u64,
    boundary_effect_limit_ns: u64,
) -> Result<String, Qg6ResidualValidationError> {
    if predecessor_leaf_id.is_some() != predecessor_instance_receipt_sha256.is_some()
        || predecessor_instance_receipt_sha256.is_some_and(|receipt| !is_lower_hex_sha256(receipt))
        || boundary_effect_limit_ns == 0
        || boundary_effect_ns > boundary_effect_limit_ns
    {
        return Err(Qg6ResidualValidationError::InvalidLeaf {
            reason: "residual boundary-effect binding is malformed or exceeds its bound".to_owned(),
        });
    }
    if predecessor_leaf_id.is_none() && boundary_effect_ns != 0 {
        return Err(Qg6ResidualValidationError::InvalidLeaf {
            reason: "residual first execution leaf cannot report a predecessor effect".to_owned(),
        });
    }
    let mut hasher = Sha256::new();
    hash_len_prefixed(
        &mut hasher,
        QG6_R1_RESIDUAL_BOUNDARY_EFFECT_VERSION.as_bytes(),
    );
    match predecessor_leaf_id {
        Some(leaf_id) => {
            hasher.update([1]);
            hasher.update(leaf_id.to_le_bytes());
            let receipt = predecessor_instance_receipt_sha256.ok_or_else(|| {
                Qg6ResidualValidationError::InvalidLeaf {
                    reason: "residual boundary-effect predecessor receipt is missing".to_owned(),
                }
            })?;
            hash_len_prefixed(&mut hasher, receipt.as_bytes());
        }
        None => hasher.update([0]),
    }
    hasher.update(boundary_effect_ns.to_le_bytes());
    hasher.update(boundary_effect_limit_ns.to_le_bytes());
    Ok(lower_hex(hasher.finalize()))
}

fn residual_invocation_receipt_sha256(
    leaf: &Qg6ResidualLeafObservation,
) -> Result<String, Qg6ResidualValidationError> {
    if ![
        &leaf.instance_receipt_sha256,
        &leaf.parsed_ast_sha256,
        &leaf.raw_query_sha256,
        &leaf.ranked_cache_receipt_sha256,
        &leaf.ranked_cache_key_sha256,
        &leaf.ranked_cache_nonce_sha256,
        &leaf.ranked_cache_lookup_receipt_sha256,
        &leaf.result_envelope_sha256,
        &leaf.host_receipt_sha256,
        &leaf.boot_receipt_sha256,
        &leaf.clock_receipt_sha256,
    ]
    .into_iter()
    .all(|value| is_lower_hex_sha256(value))
    {
        return Err(Qg6ResidualValidationError::InvalidLeaf {
            reason: "residual invocation receipt inputs are malformed".to_owned(),
        });
    }
    let cache_disposition = match leaf.cache_disposition {
        Qg6ResidualCacheDisposition::FirstTouch => 0_u8,
        Qg6ResidualCacheDisposition::RankedMiss => 1,
        Qg6ResidualCacheDisposition::RankedHit => 2,
        Qg6ResidualCacheDisposition::GenerationRebind => 3,
    };
    let mut hasher = Sha256::new();
    hash_len_prefixed(&mut hasher, QG6_R1_RESIDUAL_INVOCATION_VERSION.as_bytes());
    hash_len_prefixed(&mut hasher, leaf.instance_receipt_sha256.as_bytes());
    hash_len_prefixed(&mut hasher, leaf.query_id.as_bytes());
    hash_len_prefixed(&mut hasher, leaf.parsed_ast_sha256.as_bytes());
    hash_len_prefixed(&mut hasher, leaf.raw_query_sha256.as_bytes());
    hash_len_prefixed(&mut hasher, leaf.ranked_cache_receipt_sha256.as_bytes());
    hash_len_prefixed(&mut hasher, leaf.ranked_cache_key_sha256.as_bytes());
    hash_len_prefixed(&mut hasher, leaf.ranked_cache_nonce_sha256.as_bytes());
    hash_len_prefixed(
        &mut hasher,
        leaf.ranked_cache_lookup_receipt_sha256.as_bytes(),
    );
    hasher.update([cache_disposition]);
    hash_len_prefixed(&mut hasher, leaf.result_envelope_sha256.as_bytes());
    hasher.update(leaf.work_units.to_le_bytes());
    hasher.update(leaf.fuel_budget.to_le_bytes());
    hasher.update(leaf.fuel_consumed.to_le_bytes());
    hasher.update([u8::from(leaf.cancelled)]);
    hash_len_prefixed(&mut hasher, leaf.outcome_code.as_bytes());
    hash_len_prefixed(&mut hasher, leaf.host_receipt_sha256.as_bytes());
    hash_len_prefixed(&mut hasher, leaf.boot_receipt_sha256.as_bytes());
    hash_len_prefixed(&mut hasher, leaf.clock_receipt_sha256.as_bytes());
    hasher.update(leaf.started_ns.to_le_bytes());
    hasher.update(leaf.ended_ns.to_le_bytes());
    Ok(lower_hex(hasher.finalize()))
}

fn residual_rebind_transition_sha256(
    leaf: &Qg6ResidualLeafObservation,
    evidence: &Qg6ResidualGenerationRebindEvidence,
) -> Result<String, Qg6ResidualValidationError> {
    if ![
        &evidence.pre_backing_sha256,
        &evidence.mutation_plan_sha256,
        &evidence.mutation_receipt_sha256,
        &evidence.parity_receipt_sha256,
        &leaf.backing_sha256,
        &leaf.result_envelope_sha256,
    ]
    .into_iter()
    .all(|value| is_lower_hex_sha256(value))
    {
        return Err(Qg6ResidualValidationError::InvalidLeaf {
            reason: "residual generation-rebind transition inputs are malformed".to_owned(),
        });
    }
    let mut hasher = Sha256::new();
    hash_len_prefixed(
        &mut hasher,
        QG6_R1_RESIDUAL_REBIND_TRANSITION_VERSION.as_bytes(),
    );
    hasher.update(evidence.pre_generation.to_le_bytes());
    hasher.update(leaf.generation.to_le_bytes());
    hash_len_prefixed(&mut hasher, evidence.pre_backing_sha256.as_bytes());
    hash_len_prefixed(&mut hasher, leaf.backing_sha256.as_bytes());
    hash_len_prefixed(&mut hasher, evidence.mutation_plan_sha256.as_bytes());
    hash_len_prefixed(&mut hasher, evidence.mutation_receipt_sha256.as_bytes());
    hash_len_prefixed(&mut hasher, evidence.parity_receipt_sha256.as_bytes());
    hash_len_prefixed(&mut hasher, leaf.result_envelope_sha256.as_bytes());
    Ok(lower_hex(hasher.finalize()))
}

/// Admit schedule-level prerequisites for exactly one complete residual
/// meta-block and derive its joint contrast vector.
///
/// The later hierarchical bootstrap must resample these vectors as one unit.
/// No external caller can presently mint the required producer authority. This
/// public entry therefore fails closed until the real residual runner binds its
/// source/build, parser, cache, clock, and lifecycle receipts. The internal
/// authority-bearing admission core still cannot admit evidence for a
/// performance claim because it does not run the later joint bootstrap gate.
///
/// # Errors
///
/// Rejects incomplete, forged, duplicate, overlapped, malformed, mixed-
/// untrusted observations. A future producer must use the internal capability
/// path; caller-supplied strings are not an authority.
pub fn admit_qg6_residual_schedule_meta_block(
    _observations: Vec<Qg6ResidualLeafObservation>,
) -> Result<Qg6ResidualScheduleAdmission, Qg6ResidualValidationError> {
    Err(Qg6ResidualValidationError::ProvenanceMismatch {
        reason: "residual admission has no bound trusted producer authority in this phase"
            .to_owned(),
    })
}

fn admit_qg6_residual_schedule_meta_block_with_authority(
    observations: Vec<Qg6ResidualLeafObservation>,
    authority: &Qg6ResidualProducerAuthority,
) -> Result<Qg6ResidualScheduleAdmission, Qg6ResidualValidationError> {
    if observations.len() != QG6_R1_RESIDUAL_LEAF_COUNT {
        return Err(Qg6ResidualValidationError::InvalidLeaf {
            reason: "residual meta-block must retain exactly 36 observations".to_owned(),
        });
    }
    let meta_block_id = observations[0].meta_block_id;
    let mut leaves_by_id = BTreeMap::new();
    for observation in observations {
        validate_residual_leaf_shape(&observation)?;
        if observation.meta_block_id != meta_block_id {
            return Err(Qg6ResidualValidationError::InvalidLeaf {
                reason: "residual meta-block mixes meta-block identities".to_owned(),
            });
        }
        if leaves_by_id
            .insert(observation.leaf_id, observation)
            .is_some()
        {
            return Err(Qg6ResidualValidationError::InvalidLeaf {
                reason: "residual meta-block contains a duplicate leaf ID".to_owned(),
            });
        }
    }

    let mut schedule = leaves_by_id
        .values()
        .map(|leaf| Qg6ResidualWilliamsLeaf {
            leaf_id: leaf.leaf_id,
            meta_block_id: leaf.meta_block_id,
            sweep: leaf.sweep,
            ordinal: leaf.ordinal,
            role: leaf.role,
        })
        .collect::<Vec<_>>();
    schedule.sort_unstable_by_key(|leaf| (leaf.sweep, leaf.ordinal));
    validate_qg6_residual_williams_schedule(&schedule)?;

    let mut leaves_by_execution_ordinal = BTreeMap::new();
    for observation in leaves_by_id.into_values() {
        if leaves_by_execution_ordinal
            .insert(observation.execution_ordinal, observation)
            .is_some()
        {
            return Err(Qg6ResidualValidationError::InvalidLeaf {
                reason: "residual meta-block contains a duplicate execution ordinal".to_owned(),
            });
        }
    }
    let mut ordered = Vec::new();
    ordered
        .try_reserve_exact(QG6_R1_RESIDUAL_LEAF_COUNT)
        .map_err(|_| Qg6ResidualValidationError::InvalidLeaf {
            reason: "residual observation ordering allocation failed".to_owned(),
        })?;
    for execution_ordinal in 0_u8..QG6_R1_RESIDUAL_LEAF_COUNT_U8 {
        ordered.push(
            leaves_by_execution_ordinal
                .remove(&execution_ordinal)
                .ok_or_else(|| Qg6ResidualValidationError::InvalidLeaf {
                    reason: "residual meta-block omits an execution ordinal".to_owned(),
                })?,
        );
    }
    validate_residual_execution_williams_order(&ordered)?;
    validate_residual_leaf_intervals(&ordered)?;
    validate_residual_shared_scope(&ordered)?;
    validate_residual_boundary_effects(&ordered)?;
    validate_residual_ranked_miss_semantics(&ordered)?;
    validate_residual_role_build_provenance(&ordered)?;
    validate_residual_physical_provenance(&ordered)?;
    validate_residual_backing_content(&ordered)?;
    validate_residual_generation_rebind_evidence(&ordered)?;
    authority.validate(&ordered)?;
    Ok(Qg6ResidualScheduleAdmission {
        schedule,
        joint_contrasts: residual_joint_contrast_vector(&ordered),
        leaves: ordered,
    })
}

// Keep the authority-bearing core type-checked in ordinary builds even while
// the public entry point deliberately fails closed. This is a signature pin,
// not a production authority constructor: only the test module can mint the
// capability until the real residual runner owns the required observations.
const _: fn(
    Vec<Qg6ResidualLeafObservation>,
    &Qg6ResidualProducerAuthority,
) -> Result<Qg6ResidualScheduleAdmission, Qg6ResidualValidationError> =
    admit_qg6_residual_schedule_meta_block_with_authority;

fn canonical_residual_williams_schedule(
    meta_block_id: u64,
) -> Result<Vec<Qg6ResidualWilliamsLeaf>, Qg6ResidualValidationError> {
    let first_leaf_id = meta_block_id
        .checked_mul(QG6_R1_RESIDUAL_LEAF_COUNT_U64)
        .ok_or_else(|| Qg6ResidualValidationError::InvalidSchedule {
            reason: "meta-block leaf-ID range overflows u64".to_owned(),
        })?;
    let mut expected = Vec::new();
    expected
        .try_reserve_exact(QG6_R1_RESIDUAL_LEAF_COUNT)
        .map_err(|_| Qg6ResidualValidationError::InvalidSchedule {
            reason: "canonical Williams schedule allocation failed".to_owned(),
        })?;
    for sweep in 0_u8..QG6_R1_RESIDUAL_ROLE_COUNT_U8 {
        for ordinal in 0_u8..QG6_R1_RESIDUAL_ROLE_COUNT_U8 {
            let role_index = (QG6_R1_RESIDUAL_WILLIAMS_BASE_ROW[usize::from(ordinal)] + sweep)
                % QG6_R1_RESIDUAL_ROLE_COUNT_U8;
            let leaf_offset =
                u64::from(sweep) * u64::from(QG6_R1_RESIDUAL_ROLE_COUNT_U8) + u64::from(ordinal);
            let role = *Qg6ResidualArmRole::ALL
                .get(usize::from(role_index))
                .ok_or_else(|| Qg6ResidualValidationError::InvalidSchedule {
                    reason: "canonical Williams role index is out of bounds".to_owned(),
                })?;
            expected.push(Qg6ResidualWilliamsLeaf {
                leaf_id: first_leaf_id.checked_add(leaf_offset).ok_or_else(|| {
                    Qg6ResidualValidationError::InvalidSchedule {
                        reason: "meta-block leaf ID overflows u64".to_owned(),
                    }
                })?,
                meta_block_id,
                sweep,
                ordinal,
                role,
            });
        }
    }
    Ok(expected)
}

fn validate_residual_williams_balance(
    schedule: &[Qg6ResidualWilliamsLeaf],
) -> Result<(), Qg6ResidualValidationError> {
    let mut role_ordinal_counts = [[0_u8; QG6_R1_RESIDUAL_ROLE_COUNT]; QG6_R1_RESIDUAL_ROLE_COUNT];
    let mut before_counts = [[0_u8; QG6_R1_RESIDUAL_ROLE_COUNT]; QG6_R1_RESIDUAL_ROLE_COUNT];
    let mut predecessor_counts = [[0_u8; QG6_R1_RESIDUAL_ROLE_COUNT]; QG6_R1_RESIDUAL_ROLE_COUNT];
    // `as_chunks` carries the row width in the type rather than in a runtime
    // argument, so a row cannot be observed short. The trailing remainder is
    // discarded exactly as `chunks_exact` discarded it; this is a typing
    // change, not a validation change.
    let (rows, _) = schedule.as_chunks::<QG6_R1_RESIDUAL_ROLE_COUNT>();
    for row in rows {
        for leaf in row {
            role_ordinal_counts[leaf.role.index()][usize::from(leaf.ordinal)] += 1;
        }
        for left in 0..QG6_R1_RESIDUAL_ROLE_COUNT {
            for right in left + 1..QG6_R1_RESIDUAL_ROLE_COUNT {
                before_counts[row[left].role.index()][row[right].role.index()] += 1;
            }
        }
        for pair in row.windows(2) {
            predecessor_counts[pair[0].role.index()][pair[1].role.index()] += 1;
        }
    }
    for role in Qg6ResidualArmRole::ALL {
        if role_ordinal_counts[role.index()]
            .iter()
            .any(|count| *count != 1)
        {
            return Err(Qg6ResidualValidationError::InvalidSchedule {
                reason: "each residual role must occupy every ordinal exactly once".to_owned(),
            });
        }
        for other in Qg6ResidualArmRole::ALL {
            if role == other {
                continue;
            }
            if before_counts[role.index()][other.index()] != 3
                || before_counts[other.index()][role.index()] != 3
            {
                return Err(Qg6ResidualValidationError::InvalidSchedule {
                    reason: "each ordered residual role pair must be before/after 3/3".to_owned(),
                });
            }
            if predecessor_counts[role.index()][other.index()] != 1 {
                return Err(Qg6ResidualValidationError::InvalidSchedule {
                    reason: "each directed residual immediate predecessor must occur once"
                        .to_owned(),
                });
            }
        }
    }
    Ok(())
}

fn is_valid_residual_query_id(query_id: &str) -> bool {
    !query_id.is_empty()
        && query_id.len() <= MAX_QUERY_ID_BYTES
        && query_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

fn validate_residual_leaf_shape(
    leaf: &Qg6ResidualLeafObservation,
) -> Result<(), Qg6ResidualValidationError> {
    if !is_valid_residual_query_id(&leaf.query_id) {
        return Err(Qg6ResidualValidationError::InvalidLeaf {
            reason: "residual query ID must be a bounded non-sensitive identifier".to_owned(),
        });
    }
    if ![
        &leaf.instance_receipt_sha256,
        &leaf.backing_sha256,
        &leaf.backing_instance_receipt_sha256,
        &leaf.ranked_cache_receipt_sha256,
        &leaf.path_receipt_sha256,
        &leaf.source_sha256,
        &leaf.cargo_lock_sha256,
        &leaf.timing_elf_sha256,
        &leaf.source_elf_consistency_sha256,
        &leaf.source_build_receipt_sha256,
        &leaf.fixture_sha256,
        &leaf.query_contract_sha256,
        &leaf.parsed_ast_sha256,
        &leaf.standardized_workload_sha256,
        &leaf.result_envelope_sha256,
        &leaf.ranked_cache_key_sha256,
        &leaf.ranked_cache_nonce_sha256,
        &leaf.raw_query_sha256,
        &leaf.ranked_miss_semantics_sha256,
        &leaf.ranked_cache_lookup_receipt_sha256,
        &leaf.host_receipt_sha256,
        &leaf.boot_receipt_sha256,
        &leaf.clock_receipt_sha256,
        &leaf.invocation_receipt_sha256,
        &leaf.boundary_effect_receipt_sha256,
    ]
    .into_iter()
    .all(|value| is_lower_hex_sha256(value))
    {
        return Err(Qg6ResidualValidationError::InvalidLeaf {
            reason: "residual identity fields must be lowercase SHA-256 values".to_owned(),
        });
    }
    let expected_consistency = qg6_residual_source_elf_consistency_sha256(
        &leaf.source_sha256,
        &leaf.cargo_lock_sha256,
        &leaf.timing_elf_sha256,
    )?;
    if leaf.source_elf_consistency_sha256 != expected_consistency {
        return Err(Qg6ResidualValidationError::ProvenanceMismatch {
            reason: "residual source/Cargo.lock/timing-ELF consistency digest does not verify"
                .to_owned(),
        });
    }
    let expected_workload =
        qg6_residual_standardized_workload_sha256(&leaf.query_id, &leaf.query_contract_sha256)?;
    if leaf.standardized_workload_sha256 != expected_workload {
        return Err(Qg6ResidualValidationError::InvalidLeaf {
            reason: "residual standardized workload digest does not verify".to_owned(),
        });
    }
    if leaf.ended_ns <= leaf.started_ns
        || leaf.latency_ns == 0
        || leaf.ended_ns - leaf.started_ns != leaf.latency_ns
    {
        return Err(Qg6ResidualValidationError::InvalidLeaf {
            reason: "residual monotonic interval and latency arithmetic disagree".to_owned(),
        });
    }
    if Duration::from_nanos(leaf.latency_ns) > QG6_R1_RESIDUAL_MAX_LATENCY {
        return Err(Qg6ResidualValidationError::InvalidLeaf {
            reason: "residual latency exceeds the bounded estimator domain".to_owned(),
        });
    }
    if leaf.cache_disposition != leaf.stratum.expected_cache_disposition() {
        return Err(Qg6ResidualValidationError::InvalidLeaf {
            reason: "residual cache disposition is incompatible with its stratum".to_owned(),
        });
    }
    if leaf.execution_ordinal >= QG6_R1_RESIDUAL_LEAF_COUNT_U8
        || leaf.work_units == 0
        || leaf.fuel_consumed > leaf.fuel_budget
        || leaf.cancelled
        || leaf.outcome_code != QG6_R1_RESIDUAL_COMPLETED_OUTCOME_CODE
    {
        return Err(Qg6ResidualValidationError::InvalidLeaf {
            reason: "residual execution order, work, fuel/cancel, or typed outcome is invalid"
                .to_owned(),
        });
    }
    let expected_ranked_miss_semantics = residual_ranked_miss_semantics_sha256(
        &leaf.query_id,
        &leaf.parsed_ast_sha256,
        leaf.raw_query_length_bytes,
        &leaf.ranked_cache_nonce_sha256,
        &leaf.ranked_cache_key_sha256,
        &leaf.result_envelope_sha256,
        leaf.work_units,
        leaf.fuel_budget,
        leaf.fuel_consumed,
        leaf.cancelled,
        &leaf.outcome_code,
    )?;
    if leaf.ranked_miss_semantics_sha256 != expected_ranked_miss_semantics {
        return Err(Qg6ResidualValidationError::InvalidLeaf {
            reason: "residual ranked-cache nonce/key semantics binding does not verify".to_owned(),
        });
    }
    let expected_invocation_receipt = residual_invocation_receipt_sha256(leaf)?;
    if leaf.invocation_receipt_sha256 != expected_invocation_receipt {
        return Err(Qg6ResidualValidationError::InvalidLeaf {
            reason: "residual invocation binding does not verify".to_owned(),
        });
    }
    match (leaf.stratum, &leaf.generation_rebind_evidence) {
        (Qg6ResidualStratum::GenerationRebind, Some(evidence)) => {
            if evidence.pre_generation >= leaf.generation
                || evidence.pre_backing_sha256 == leaf.backing_sha256
                || ![
                    &evidence.pre_backing_sha256,
                    &evidence.mutation_plan_sha256,
                    &evidence.mutation_receipt_sha256,
                    &evidence.parity_receipt_sha256,
                    &evidence.transition_receipt_sha256,
                ]
                .into_iter()
                .all(|value| is_lower_hex_sha256(value))
                || evidence.transition_receipt_sha256
                    != residual_rebind_transition_sha256(leaf, evidence)?
            {
                return Err(Qg6ResidualValidationError::InvalidLeaf {
                    reason: "residual generation-rebind transition evidence is malformed"
                        .to_owned(),
                });
            }
        }
        (Qg6ResidualStratum::GenerationRebind, None)
        | (Qg6ResidualStratum::FirstTouch | Qg6ResidualStratum::SteadyRankedCacheMiss, Some(_)) => {
            return Err(Qg6ResidualValidationError::InvalidLeaf {
                reason:
                    "residual generation-rebind evidence is missing or appears in the wrong stratum"
                        .to_owned(),
            });
        }
        (Qg6ResidualStratum::FirstTouch | Qg6ResidualStratum::SteadyRankedCacheMiss, None) => {}
    }
    let expected_boundary_effect = residual_boundary_effect_sha256(
        leaf.boundary_predecessor_leaf_id,
        leaf.boundary_predecessor_instance_receipt_sha256.as_deref(),
        leaf.boundary_effect_ns,
        leaf.boundary_effect_limit_ns,
    )?;
    if leaf.boundary_effect_receipt_sha256 != expected_boundary_effect {
        return Err(Qg6ResidualValidationError::InvalidLeaf {
            reason: "residual boundary-effect binding does not verify".to_owned(),
        });
    }
    Ok(())
}

fn validate_residual_execution_williams_order(
    ordered: &[Qg6ResidualLeafObservation],
) -> Result<(), Qg6ResidualValidationError> {
    let mut observed_sweeps = BTreeSet::new();
    // Same typing change as the schedule scan above: the width moves into the
    // type and the discarded remainder keeps `chunks_exact` behaviour.
    let (rows, _) = ordered.as_chunks::<QG6_R1_RESIDUAL_ROLE_COUNT>();
    for row in rows {
        let sweep = row[0].sweep;
        if !observed_sweeps.insert(sweep)
            || row
                .iter()
                .enumerate()
                .any(|(ordinal, leaf)| leaf.sweep != sweep || usize::from(leaf.ordinal) != ordinal)
        {
            return Err(Qg6ResidualValidationError::InvalidLeaf {
                reason: "residual execution order is not a permutation of complete Williams rows"
                    .to_owned(),
            });
        }
    }
    if observed_sweeps.len() != QG6_R1_RESIDUAL_ROLE_COUNT {
        return Err(Qg6ResidualValidationError::InvalidLeaf {
            reason: "residual execution order omits a Williams row".to_owned(),
        });
    }
    Ok(())
}

fn validate_residual_leaf_intervals(
    ordered: &[Qg6ResidualLeafObservation],
) -> Result<(), Qg6ResidualValidationError> {
    for pair in ordered.windows(2) {
        if pair[1].started_ns < pair[0].ended_ns {
            return Err(Qg6ResidualValidationError::InvalidLeaf {
                reason: "residual timed leaf intervals overlap or violate execution order"
                    .to_owned(),
            });
        }
    }
    Ok(())
}

fn validate_residual_shared_scope(
    ordered: &[Qg6ResidualLeafObservation],
) -> Result<(), Qg6ResidualValidationError> {
    let reference = &ordered[0];
    for leaf in &ordered[1..] {
        if leaf.stratum != reference.stratum
            || leaf.query_class != reference.query_class
            || leaf.query_id != reference.query_id
            || leaf.fixture_sha256 != reference.fixture_sha256
            || leaf.query_contract_sha256 != reference.query_contract_sha256
            || leaf.parsed_ast_sha256 != reference.parsed_ast_sha256
            || leaf.standardized_workload_sha256 != reference.standardized_workload_sha256
            || leaf.result_envelope_sha256 != reference.result_envelope_sha256
            || leaf.generation != reference.generation
            || leaf.cache_disposition != reference.cache_disposition
            || leaf.work_units != reference.work_units
            || leaf.fuel_budget != reference.fuel_budget
            || leaf.fuel_consumed != reference.fuel_consumed
            || leaf.cancelled != reference.cancelled
            || leaf.outcome_code != reference.outcome_code
            || leaf.host_receipt_sha256 != reference.host_receipt_sha256
            || leaf.boot_receipt_sha256 != reference.boot_receipt_sha256
            || leaf.clock_receipt_sha256 != reference.clock_receipt_sha256
            || leaf.boundary_effect_limit_ns != reference.boundary_effect_limit_ns
        {
            return Err(Qg6ResidualValidationError::InvalidLeaf {
                reason:
                    "residual meta-block mixes query, semantic, result, lifecycle, work, host/boot, clock, or stratum scope"
                        .to_owned(),
            });
        }
    }
    Ok(())
}

fn validate_residual_role_build_provenance(
    ordered: &[Qg6ResidualLeafObservation],
) -> Result<(), Qg6ResidualValidationError> {
    let mut by_role: [Option<Qg6ResidualRoleBuildProvenance>; QG6_R1_RESIDUAL_ROLE_COUNT] =
        std::array::from_fn(|_| None);
    for leaf in ordered {
        let candidate = Qg6ResidualRoleBuildProvenance::from(leaf);
        let slot = &mut by_role[leaf.role.index()];
        if let Some(existing) = slot {
            if existing != &candidate {
                return Err(Qg6ResidualValidationError::ProvenanceMismatch {
                    reason:
                        "one residual role changed its source/build provenance within a meta-block"
                            .to_owned(),
                });
            }
        } else {
            *slot = Some(candidate);
        }
    }
    if by_role.iter().any(Option::is_none) {
        return Err(Qg6ResidualValidationError::ProvenanceMismatch {
            reason: "residual meta-block omitted a source/build role provenance".to_owned(),
        });
    }
    let provenances: [Qg6ResidualRoleBuildProvenance; QG6_R1_RESIDUAL_ROLE_COUNT] = by_role
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| Qg6ResidualValidationError::ProvenanceMismatch {
            reason: "residual source/build role materialization failed".to_owned(),
        })?
        .try_into()
        .map_err(|_| Qg6ResidualValidationError::ProvenanceMismatch {
            reason: "residual source/build role materialization failed".to_owned(),
        })?;
    let old_a = &provenances[Qg6ResidualArmRole::OldA.index()];
    let old_b = &provenances[Qg6ResidualArmRole::OldB.index()];
    let current_a = &provenances[Qg6ResidualArmRole::CurrentA.index()];
    let current_b = &provenances[Qg6ResidualArmRole::CurrentB.index()];
    let tantivy_a = &provenances[Qg6ResidualArmRole::TantivyA.index()];
    let tantivy_b = &provenances[Qg6ResidualArmRole::TantivyB.index()];
    if old_a.source != old_b.source
        || current_a.source != current_b.source
        || tantivy_a.source != tantivy_b.source
        || old_a.source == current_a.source
        || old_a.source == tantivy_a.source
        || current_a.source == tantivy_a.source
        || provenances
            .iter()
            .any(|provenance| provenance.cargo_lock != old_a.cargo_lock)
        || provenances
            .iter()
            .any(|provenance| provenance.timing_elf != old_a.timing_elf)
        || provenances
            .iter()
            .map(|provenance| &provenance.source_build)
            .collect::<BTreeSet<_>>()
            .len()
            != QG6_R1_RESIDUAL_ROLE_COUNT
    {
        return Err(Qg6ResidualValidationError::ProvenanceMismatch {
            reason: "residual source/build topology lacks independent A/A and family bindings"
                .to_owned(),
        });
    }
    Ok(())
}

fn validate_residual_physical_provenance(
    ordered: &[Qg6ResidualLeafObservation],
) -> Result<(), Qg6ResidualValidationError> {
    let first_touch = ordered[0].stratum == Qg6ResidualStratum::FirstTouch;
    let mut instances = BTreeSet::<String>::new();
    let mut backing_instances = BTreeSet::<String>::new();
    let mut ranked_caches = BTreeSet::<String>::new();
    let mut paths = BTreeSet::<String>::new();
    let mut by_role: [Option<Qg6ResidualPhysicalArmProvenance>; QG6_R1_RESIDUAL_ROLE_COUNT] =
        std::array::from_fn(|_| None);
    for leaf in ordered {
        let candidate = Qg6ResidualPhysicalArmProvenance::from(leaf);
        if first_touch {
            if !instances.insert(candidate.instance.clone())
                || !backing_instances.insert(candidate.backing.clone())
                || !ranked_caches.insert(candidate.ranked_cache.clone())
                || !paths.insert(candidate.path.clone())
            {
                return Err(Qg6ResidualValidationError::ProvenanceMismatch {
                    reason: "residual first-touch leaves reuse an instance, backing identity, ranked-cache, or path receipt"
                        .to_owned(),
                });
            }
        } else {
            let slot = &mut by_role[leaf.role.index()];
            if let Some(existing) = slot {
                if existing != &candidate {
                    return Err(Qg6ResidualValidationError::ProvenanceMismatch {
                        reason: "one steady/rebind residual role changed physical provenance within a meta-block"
                            .to_owned(),
                    });
                }
            } else {
                *slot = Some(candidate);
            }
        }
    }
    if !first_touch {
        for provenance in by_role.iter().flatten() {
            if !instances.insert(provenance.instance.clone())
                || !backing_instances.insert(provenance.backing.clone())
                || !ranked_caches.insert(provenance.ranked_cache.clone())
                || !paths.insert(provenance.path.clone())
            {
                return Err(Qg6ResidualValidationError::ProvenanceMismatch {
                    reason: "residual physical roles share an instance, backing identity, ranked-cache, or path receipt"
                        .to_owned(),
                });
            }
        }
    }
    let expected_cardinality = if first_touch {
        QG6_R1_RESIDUAL_LEAF_COUNT
    } else {
        QG6_R1_RESIDUAL_ROLE_COUNT
    };
    if instances.len() != expected_cardinality
        || backing_instances.len() != expected_cardinality
        || ranked_caches.len() != expected_cardinality
        || paths.len() != expected_cardinality
    {
        return Err(Qg6ResidualValidationError::ProvenanceMismatch {
            reason:
                "residual physical-arm receipt cardinality does not match its lifecycle stratum"
                    .to_owned(),
        });
    }
    if !first_touch && by_role.iter().any(Option::is_none) {
        return Err(Qg6ResidualValidationError::ProvenanceMismatch {
            reason: "residual steady/rebind meta-block omitted a physical role provenance"
                .to_owned(),
        });
    }
    Ok(())
}

fn validate_residual_backing_content(
    ordered: &[Qg6ResidualLeafObservation],
) -> Result<(), Qg6ResidualValidationError> {
    let mut backing_by_role: [Option<&str>; QG6_R1_RESIDUAL_ROLE_COUNT] =
        std::array::from_fn(|_| None);
    for leaf in ordered {
        let slot = &mut backing_by_role[leaf.role.index()];
        if let Some(existing) = slot {
            if *existing != leaf.backing_sha256 {
                return Err(Qg6ResidualValidationError::ProvenanceMismatch {
                    reason: "one residual role changed backing byte content within a meta-block"
                        .to_owned(),
                });
            }
        } else {
            *slot = Some(&leaf.backing_sha256);
        }
    }
    let backing_by_role = backing_by_role.map(|backing| {
        backing.ok_or_else(|| Qg6ResidualValidationError::ProvenanceMismatch {
            reason: "residual meta-block omitted backing byte content".to_owned(),
        })
    });
    let backing_by_role: [&str; QG6_R1_RESIDUAL_ROLE_COUNT] = backing_by_role
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| Qg6ResidualValidationError::ProvenanceMismatch {
            reason: "residual backing byte-content materialization failed".to_owned(),
        })?;
    for (left, right) in [
        (Qg6ResidualArmRole::OldA, Qg6ResidualArmRole::OldB),
        (Qg6ResidualArmRole::CurrentA, Qg6ResidualArmRole::CurrentB),
        (Qg6ResidualArmRole::TantivyA, Qg6ResidualArmRole::TantivyB),
    ] {
        if backing_by_role[left.index()] != backing_by_role[right.index()] {
            return Err(Qg6ResidualValidationError::ProvenanceMismatch {
                reason:
                    "residual A/A backing byte content differs despite distinct physical identities"
                        .to_owned(),
            });
        }
    }
    Ok(())
}

fn validate_residual_ranked_miss_semantics(
    ordered: &[Qg6ResidualLeafObservation],
) -> Result<(), Qg6ResidualValidationError> {
    if ordered[0].stratum != Qg6ResidualStratum::SteadyRankedCacheMiss {
        return Ok(());
    }
    let raw_query_length_bytes = ordered[0].raw_query_length_bytes;
    let mut nonces = BTreeSet::new();
    let mut cache_keys = BTreeSet::new();
    for leaf in ordered {
        if leaf.raw_query_length_bytes != raw_query_length_bytes
            || !nonces.insert(leaf.ranked_cache_nonce_sha256.as_str())
            || !cache_keys.insert(leaf.ranked_cache_key_sha256.as_str())
        {
            return Err(Qg6ResidualValidationError::InvalidLeaf {
                reason: "residual steady ranked-miss leaves lack unique constant-length nonce/cache-key semantics"
                    .to_owned(),
            });
        }
    }
    Ok(())
}

fn validate_residual_generation_rebind_evidence(
    ordered: &[Qg6ResidualLeafObservation],
) -> Result<(), Qg6ResidualValidationError> {
    if ordered[0].stratum != Qg6ResidualStratum::GenerationRebind {
        return Ok(());
    }
    let expected_plan = ordered[0]
        .generation_rebind_evidence
        .as_ref()
        .ok_or_else(|| Qg6ResidualValidationError::InvalidLeaf {
            reason: "residual rebind meta-block lacks transition evidence".to_owned(),
        })?
        .mutation_plan_sha256
        .as_str();
    let mut by_role: [Option<&Qg6ResidualGenerationRebindEvidence>; QG6_R1_RESIDUAL_ROLE_COUNT] =
        std::array::from_fn(|_| None);
    let mut mutation_receipts = BTreeSet::new();
    let mut parity_receipts = BTreeSet::new();
    for leaf in ordered {
        let evidence = leaf.generation_rebind_evidence.as_ref().ok_or_else(|| {
            Qg6ResidualValidationError::InvalidLeaf {
                reason: "residual rebind leaf lacks transition evidence".to_owned(),
            }
        })?;
        if evidence.mutation_plan_sha256 != expected_plan {
            return Err(Qg6ResidualValidationError::InvalidLeaf {
                reason: "residual rebind leaves disagree on the matched mutation plan".to_owned(),
            });
        }
        let slot = &mut by_role[leaf.role.index()];
        if let Some(existing) = slot {
            if *existing != evidence {
                return Err(Qg6ResidualValidationError::InvalidLeaf {
                    reason:
                        "one residual role changed rebind transition evidence within a meta-block"
                            .to_owned(),
                });
            }
        } else {
            *slot = Some(evidence);
            mutation_receipts.insert(evidence.mutation_receipt_sha256.as_str());
            parity_receipts.insert(evidence.parity_receipt_sha256.as_str());
        }
    }
    if by_role.iter().any(Option::is_none)
        || mutation_receipts.len() != QG6_R1_RESIDUAL_ROLE_COUNT
        || parity_receipts.len() != QG6_R1_RESIDUAL_ROLE_COUNT
    {
        return Err(Qg6ResidualValidationError::InvalidLeaf {
            reason: "residual rebind receipts are not independent for all six physical arms"
                .to_owned(),
        });
    }
    Ok(())
}

fn validate_residual_boundary_effects(
    ordered: &[Qg6ResidualLeafObservation],
) -> Result<(), Qg6ResidualValidationError> {
    let expected_boundary_effect_limit_ns = ordered[0].boundary_effect_limit_ns;
    if expected_boundary_effect_limit_ns == 0 {
        return Err(Qg6ResidualValidationError::InvalidLeaf {
            reason: "residual boundary-effect authority has no positive predeclared bound"
                .to_owned(),
        });
    }
    let first = &ordered[0];
    if first.boundary_predecessor_leaf_id.is_some()
        || first.boundary_predecessor_instance_receipt_sha256.is_some()
        || first.boundary_effect_limit_ns != expected_boundary_effect_limit_ns
    {
        return Err(Qg6ResidualValidationError::InvalidLeaf {
            reason: "residual first execution leaf has a forged boundary predecessor".to_owned(),
        });
    }
    for pair in ordered.windows(2) {
        let previous = &pair[0];
        let leaf = &pair[1];
        if leaf.boundary_predecessor_leaf_id != Some(previous.leaf_id)
            || leaf.boundary_predecessor_instance_receipt_sha256.as_deref()
                != Some(previous.instance_receipt_sha256.as_str())
            || leaf.boundary_effect_limit_ns != expected_boundary_effect_limit_ns
            || leaf.boundary_effect_ns
                != leaf
                    .started_ns
                    .checked_sub(previous.ended_ns)
                    .ok_or_else(|| Qg6ResidualValidationError::InvalidLeaf {
                        reason: "residual boundary interval underflowed".to_owned(),
                    })?
        {
            return Err(Qg6ResidualValidationError::InvalidLeaf {
                reason: "residual boundary receipt does not bind the actual bounded execution gap"
                    .to_owned(),
            });
        }
    }
    Ok(())
}

fn residual_joint_contrast_vector(
    ordered: &[Qg6ResidualLeafObservation],
) -> Qg6ResidualJointContrastVector {
    let means: [f64; QG6_R1_RESIDUAL_ROLE_COUNT] = std::array::from_fn(|index| {
        let role = Qg6ResidualArmRole::ALL[index];
        let total = ordered
            .iter()
            .filter(|leaf| leaf.role == role)
            .map(|leaf| Duration::from_nanos(leaf.latency_ns).as_secs_f64().ln())
            .sum::<f64>();
        total / QG6_R1_RESIDUAL_ROLE_COUNT_F64
    });
    let old_mean = f64::midpoint(
        means[Qg6ResidualArmRole::OldA.index()],
        means[Qg6ResidualArmRole::OldB.index()],
    );
    let current_mean = f64::midpoint(
        means[Qg6ResidualArmRole::CurrentA.index()],
        means[Qg6ResidualArmRole::CurrentB.index()],
    );
    let tantivy_mean = f64::midpoint(
        means[Qg6ResidualArmRole::TantivyA.index()],
        means[Qg6ResidualArmRole::TantivyB.index()],
    );
    Qg6ResidualJointContrastVector {
        old_b_minus_old_a: means[Qg6ResidualArmRole::OldB.index()]
            - means[Qg6ResidualArmRole::OldA.index()],
        current_b_minus_current_a: means[Qg6ResidualArmRole::CurrentB.index()]
            - means[Qg6ResidualArmRole::CurrentA.index()],
        tantivy_b_minus_tantivy_a: means[Qg6ResidualArmRole::TantivyB.index()]
            - means[Qg6ResidualArmRole::TantivyA.index()],
        current_mean_minus_old_mean: current_mean - old_mean,
        current_mean_minus_tantivy_mean: current_mean - tantivy_mean,
    }
}

fn pair_roles(
    comparison: Qg6Comparison,
    control_first: bool,
) -> (Qg6Comparison, Qg6ArmRole, Qg6ArmRole) {
    let (control, treatment) = match comparison {
        Qg6Comparison::TantivyNull => (Qg6ArmRole::TantivyNullLeft, Qg6ArmRole::TantivyNullRight),
        Qg6Comparison::QuillNull => (Qg6ArmRole::QuillNullLeft, Qg6ArmRole::QuillNullRight),
        Qg6Comparison::Effect => (Qg6ArmRole::EffectControl, Qg6ArmRole::EffectTreatment),
    };
    if control_first {
        (comparison, control, treatment)
    } else {
        (comparison, treatment, control)
    }
}

fn build_one<A, F>(
    role: Qg6ArmRole,
    identity: &Qg6ExperimentIdentity,
    lifecycle: &mut Qg6LifecycleReceipt,
    build: &mut F,
) -> Result<A, Qg6HarnessError>
where
    F: FnMut(Qg6ArmRole, &Qg6ExperimentIdentity, &mut Qg6SetupRecorder<'_>) -> Result<A, String>,
{
    let counters = lifecycle.arm_mut(role);
    counters.build_calls = counters.build_calls.saturating_add(1);
    let mut recorder = Qg6SetupRecorder { counters };
    let arm = build(role, identity, &mut recorder)
        .map_err(|error| adapter_failure(Qg6Phase::Prepare, role, "<setup>", &error))?;
    validate_setup(role, identity, recorder.counters)?;
    Ok(arm)
}

fn validate_setup(
    role: Qg6ArmRole,
    identity: &Qg6ExperimentIdentity,
    counters: &Qg6ArmLifecycle,
) -> Result<(), Qg6HarnessError> {
    if counters.build_calls != 1 {
        return Err(invalid_setup(
            role,
            "engine must be constructed exactly once",
        ));
    }
    if counters.populate_calls == 0 {
        return Err(invalid_setup(
            role,
            "engine must receive at least one population batch",
        ));
    }
    if counters.populated_documents != identity.document_count {
        return Err(invalid_setup(
            role,
            "populated document count must equal the frozen corpus count",
        ));
    }
    if counters.commit_calls != 1 {
        return Err(invalid_setup(
            role,
            "engine must commit exactly once before preflight",
        ));
    }
    Ok(())
}

fn invalid_setup(role: Qg6ArmRole, reason: &str) -> Qg6HarnessError {
    Qg6HarnessError::InvalidSetup {
        arm: role,
        reason: reason.to_owned(),
    }
}

struct ObservedResult {
    ordered_doc_ids: Vec<String>,
    receipt: Qg6ResultReceipt,
}

fn invoke_search<A, R, F, N>(
    arms: &Qg6SixArms<A>,
    query: &Qg6QuerySpec,
    k: usize,
    document_count: u64,
    role: Qg6ArmRole,
    phase: Qg6Phase,
    search: &mut F,
    normalize: &mut N,
) -> Result<ObservedResult, Qg6HarnessError>
where
    F: FnMut(&A, &Qg6QuerySpec, usize) -> Result<R, String>,
    N: FnMut(R) -> Qg6SearchResult,
{
    let result = search(arms.get(role), black_box(query), black_box(k))
        .map_err(|error| adapter_failure(phase, role, query.id(), &error))?;
    observe_result(
        normalize(result),
        k,
        document_count,
        phase,
        role,
        query.id(),
    )
}

fn invoke_phased_search<A, R, F, N>(
    arms: &Qg6SixArms<A>,
    query: &Qg6QuerySpec,
    k: usize,
    document_count: u64,
    role: Qg6ArmRole,
    phase: Qg6Phase,
    search: &mut F,
    normalize: &mut N,
) -> Result<ObservedResult, Qg6HarnessError>
where
    F: FnMut(&A, &Qg6QuerySpec, usize, Qg6Phase) -> Result<R, String>,
    N: FnMut(R) -> Qg6SearchResult,
{
    let result = search(arms.get(role), black_box(query), black_box(k), phase)
        .map_err(|error| adapter_failure(phase, role, query.id(), &error))?;
    observe_result(
        normalize(result),
        k,
        document_count,
        phase,
        role,
        query.id(),
    )
}

fn invoke_search_borrowed<A, R, F, N>(
    arms: &Qg6SixArms<A>,
    query: &Qg6QuerySpec,
    k: usize,
    document_count: u64,
    role: Qg6ArmRole,
    search: &mut F,
    normalize: &mut N,
) -> Result<(R, ObservedResult), Qg6HarnessError>
where
    F: FnMut(&A, &Qg6QuerySpec, usize) -> Result<R, String>,
    N: FnMut(&R) -> Qg6SearchResult,
{
    let native = search(arms.get(role), black_box(query), black_box(k))
        .map_err(|error| adapter_failure(Qg6Phase::Preflight, role, query.id(), &error))?;
    let observed = observe_result(
        normalize(&native),
        k,
        document_count,
        Qg6Phase::Preflight,
        role,
        query.id(),
    )?;
    Ok((native, observed))
}

fn build_semantic_contract(
    identity: &Qg6ExperimentIdentity,
    queries: &[Qg6QuerySpec],
    expected_results: &[Qg6SixArmResultReceipts],
) -> Result<Qg6SemanticContract, Qg6HarnessError> {
    Qg6SemanticContract::from_receipts(identity, queries, expected_results)
}

fn observe_result(
    result: Qg6SearchResult,
    k: usize,
    expected_doc_count: u64,
    phase: Qg6Phase,
    role: Qg6ArmRole,
    query_id: &str,
) -> Result<ObservedResult, Qg6HarnessError> {
    if result.ordered_hits.len() > k {
        return Err(Qg6HarnessError::InvalidResult {
            phase,
            arm: role,
            query_id: query_id.to_owned(),
            reason: "returned hit count exceeds the declared k".to_owned(),
        });
    }
    if result
        .ordered_hits
        .iter()
        .any(|hit| hit.document_id.is_empty())
    {
        return Err(Qg6HarnessError::InvalidResult {
            phase,
            arm: role,
            query_id: query_id.to_owned(),
            reason: "returned document ID is empty".to_owned(),
        });
    }
    if result
        .ordered_hits
        .iter()
        .any(|hit| hit.document_id.len() > MAX_DOC_ID_BYTES)
    {
        return Err(Qg6HarnessError::InvalidResult {
            phase,
            arm: role,
            query_id: query_id.to_owned(),
            reason: "returned document ID exceeds 4096 bytes".to_owned(),
        });
    }
    if result
        .ordered_hits
        .iter()
        .any(|hit| !f32::from_bits(hit.score_bits).is_finite())
    {
        return Err(Qg6HarnessError::InvalidResult {
            phase,
            arm: role,
            query_id: query_id.to_owned(),
            reason: "returned score is not finite".to_owned(),
        });
    }
    let mut unique_document_ids = BTreeSet::new();
    if result
        .ordered_hits
        .iter()
        .any(|hit| !unique_document_ids.insert(hit.document_id.as_str()))
    {
        return Err(Qg6HarnessError::InvalidResult {
            phase,
            arm: role,
            query_id: query_id.to_owned(),
            reason: "returned document IDs are not unique".to_owned(),
        });
    }
    let doc_count = result.doc_count.unwrap_or(expected_doc_count);
    let returned_count =
        u64::try_from(result.ordered_hits.len()).map_err(|_| Qg6HarnessError::InvalidResult {
            phase,
            arm: role,
            query_id: query_id.to_owned(),
            reason: "returned hit count does not fit u64".to_owned(),
        })?;
    if result.total_count > doc_count || doc_count != expected_doc_count {
        return Err(Qg6HarnessError::InvalidResult {
            phase,
            arm: role,
            query_id: query_id.to_owned(),
            reason: "result total/live document cardinalities are inconsistent".to_owned(),
        });
    }
    let requested_count = u64::try_from(k).map_err(|_| Qg6HarnessError::InvalidResult {
        phase,
        arm: role,
        query_id: query_id.to_owned(),
        reason: "requested result count does not fit u64".to_owned(),
    })?;
    if returned_count != result.total_count.min(requested_count) {
        return Err(Qg6HarnessError::InvalidResult {
            phase,
            arm: role,
            query_id: query_id.to_owned(),
            reason: "returned hit count is not exactly min(k, total_count)".to_owned(),
        });
    }
    let ordered_hits = result
        .ordered_hits
        .iter()
        .map(|hit| Qg6RankedHitReceipt {
            document_id_sha256: sha256_hex(hit.document_id.as_bytes()),
            score_bits: hit.score_bits,
        })
        .collect::<Vec<_>>();
    let mut receipt = Qg6ResultReceipt {
        returned_count: result.ordered_hits.len(),
        ordered_hits,
        total_count: result.total_count,
        doc_count,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = receipt.canonical_sha256();
    if let Some(claimed_sha256) = result.claimed_sha256 {
        if claimed_sha256 != receipt.receipt_sha256 {
            return Err(Qg6HarnessError::ResultDigestMismatch {
                phase,
                arm: role,
                query_id: query_id.to_owned(),
                claimed_sha256,
                computed_sha256: receipt.receipt_sha256,
            });
        }
    }
    receipt.verify(k, expected_doc_count)?;
    Ok(ObservedResult {
        receipt,
        ordered_doc_ids: result
            .ordered_hits
            .into_iter()
            .map(|hit| hit.document_id)
            .collect(),
    })
}

fn compare_exact(
    query_id: &str,
    expected_arm: Qg6ArmRole,
    expected: &ObservedResult,
    observed_arm: Qg6ArmRole,
    observed: &ObservedResult,
) -> Result<(), Qg6HarnessError> {
    if expected.receipt.returned_count != observed.receipt.returned_count {
        return Err(Qg6HarnessError::HitCountMismatch {
            query_id: query_id.to_owned(),
            expected_arm,
            observed_arm,
            expected_count: expected.receipt.returned_count,
            observed_count: observed.receipt.returned_count,
        });
    }
    if expected.ordered_doc_ids != observed.ordered_doc_ids {
        let rank = expected
            .ordered_doc_ids
            .iter()
            .zip(&observed.ordered_doc_ids)
            .position(|(left, right)| left != right)
            .expect("equal counts and unequal vectors have a differing rank");
        return Err(Qg6HarnessError::OrderedDocIdsMismatch {
            query_id: query_id.to_owned(),
            expected_arm,
            observed_arm,
            first_differing_rank: rank,
            expected_doc_sha256: sha256_hex(expected.ordered_doc_ids[rank].as_bytes()),
            observed_doc_sha256: sha256_hex(observed.ordered_doc_ids[rank].as_bytes()),
        });
    }
    if expected.receipt.receipt_sha256 != observed.receipt.receipt_sha256 {
        return Err(Qg6HarnessError::ResultDigestMismatch {
            phase: Qg6Phase::Preflight,
            arm: observed_arm,
            query_id: query_id.to_owned(),
            claimed_sha256: observed.receipt.receipt_sha256.clone(),
            computed_sha256: expected.receipt.receipt_sha256.clone(),
        });
    }
    Ok(())
}

fn semantic_parity_failure(
    query_id: &str,
    expected_arm: Qg6ArmRole,
    observed_arm: Qg6ArmRole,
    error: &str,
) -> Qg6HarnessError {
    Qg6HarnessError::SemanticParityFailure {
        query_id: query_id.to_owned(),
        expected_arm,
        observed_arm,
        error_sha256: sha256_hex(error.as_bytes()),
        error_bytes: error.len(),
    }
}

fn ensure_stable(
    phase: Qg6Phase,
    role: Qg6ArmRole,
    query_id: &str,
    expected: &Qg6ResultReceipt,
    observed: &Qg6ResultReceipt,
) -> Result<(), Qg6HarnessError> {
    if expected != observed {
        return Err(Qg6HarnessError::ResultDrift {
            phase,
            arm: role,
            query_id: query_id.to_owned(),
            expected_count: expected.returned_count,
            observed_count: observed.returned_count,
            expected_sha256: expected.receipt_sha256.clone(),
            observed_sha256: observed.receipt_sha256.clone(),
        });
    }
    Ok(())
}

fn verify_lifecycle(
    lifecycle: &Qg6LifecycleReceipt,
    document_count: u64,
    query_count: usize,
    warmup_rounds: usize,
    rounds_per_query: usize,
    searches_per_sample: usize,
) -> Result<(), Qg6HarnessError> {
    let expected_preflight = usize_to_u64(query_count)?;
    let expected_postflight = expected_preflight;
    let expected_warmups =
        usize_to_u64(query_count.checked_mul(warmup_rounds).ok_or_else(|| {
            Qg6HarnessError::LifecycleViolation {
                reason: "warmup call count overflow".to_owned(),
            }
        })?)?;
    let expected_timed = usize_to_u64(
        query_count
            .checked_mul(rounds_per_query)
            .and_then(|count| count.checked_mul(searches_per_sample))
            .ok_or_else(|| Qg6HarnessError::LifecycleViolation {
                reason: "timed call count overflow".to_owned(),
            })?,
    )?;
    for role in Qg6ArmRole::ALL {
        let arm = lifecycle.arm(role);
        if arm.build_calls != 1
            || arm.populate_calls == 0
            || arm.populated_documents != document_count
            || arm.commit_calls != 1
            || arm.preflight_search_calls != expected_preflight
            || arm.warmup_search_calls != expected_warmups
            || arm.timed_search_calls != expected_timed
            || arm.postflight_search_calls != expected_postflight
            || arm.timed_setup_calls != 0
        {
            return Err(Qg6HarnessError::LifecycleViolation {
                reason: format!(
                    "arm {role:?} counts differ from build=1, populated_documents={document_count}, \
                     commit=1, preflight={expected_preflight}, warmup={expected_warmups}, \
                     timed={expected_timed}, postflight={expected_postflight}, timed_setup=0"
                ),
            });
        }
    }
    Ok(())
}

fn median_sorted_u64(values: &[u64]) -> u64 {
    debug_assert!(!values.is_empty());
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        let low = values[middle - 1];
        low + (values[middle] - low) / 2
    }
}

#[allow(clippy::cast_precision_loss)]
fn nanoseconds_to_millis(nanoseconds: u64) -> f64 {
    nanoseconds as f64 / 1_000_000.0
}

fn normalize_query_text(text: &str) -> String {
    let nfc = text.nfc().collect::<String>();
    nfc.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn parsed_ast_sha256(
    normalized_text: &str,
    query_id: &str,
) -> Result<(String, frankensearch_quill::ParsedQuery), Qg6HarnessError> {
    let parser =
        DefaultQueryParser::new(DEFAULT_SCHEMA).map_err(|error| Qg6HarnessError::InvalidSpec {
            reason: format!(
                "QG-6 parser contract is unavailable for query {query_id:?}: sha256={}",
                sha256_hex(error.to_string().as_bytes())
            ),
        })?;
    let mut parsed = parser.parse_lenient(normalized_text);
    if parsed.was_truncated || !parsed.diagnostics.is_empty() || parsed.query.is_empty() {
        let mut diagnostic_hasher = Sha256::new();
        diagnostic_hasher.update([u8::from(parsed.was_truncated)]);
        for diagnostic in &parsed.diagnostics {
            hash_len_prefixed(
                &mut diagnostic_hasher,
                format!("{:?}", diagnostic.kind).as_bytes(),
            );
        }
        return Err(Qg6HarnessError::InvalidSpec {
            reason: format!(
                "query {query_id:?} has unsupported or recovered syntax; diagnostic_sha256={}",
                lower_hex(diagnostic_hasher.finalize())
            ),
        });
    }
    let _report = canonicalize_query(&mut parsed.query);
    let mut hasher = Sha256::new();
    hasher.update(b"frankensearch/qg6/canonical-query-ast/v1\0");
    hash_query_ast(&mut hasher, &parsed.query);
    Ok((lower_hex(hasher.finalize()), parsed))
}

fn validate_query_shape(
    class: PerfQueryClass,
    query: &Query,
    query_id: &str,
) -> Result<(), Qg6HarnessError> {
    let accepted = match class {
        PerfQueryClass::Identifier => {
            !query_contains_explicit_boolean(query) && !query_contains_must_not(query)
        }
        PerfQueryClass::ShortKeyword => {
            matches!(query, Query::Term { .. } | Query::Glob { .. })
                || is_implicit_short_keyword_default_field_expansion(query)
        }
        PerfQueryClass::NaturalLanguage => {
            matches!(query, Query::Boolean { operator: None, .. })
                && !query_contains_must_not(query)
                && !query_contains_explicit_boolean(query)
        }
        PerfQueryClass::Phrase => {
            query_contains_phrase(query)
                && !query_contains_explicit_boolean(query)
                && !query_contains_must_not(query)
        }
        PerfQueryClass::Boolean => {
            matches!(query, Query::Boolean { .. })
                && (query_contains_explicit_boolean(query) || query_contains_must_not(query))
        }
    };
    if accepted {
        Ok(())
    } else {
        Err(Qg6HarnessError::InvalidSpec {
            reason: format!(
                "query {query_id:?} parsed to a shape outside class {}",
                class_slug(class)
            ),
        })
    }
}

fn is_implicit_short_keyword_default_field_expansion(query: &Query) -> bool {
    let Query::Boolean {
        clauses,
        operator: None,
    } = query
    else {
        return false;
    };
    if clauses.len() != 2 {
        return false;
    }

    let Some(content_field_id) = DEFAULT_SCHEMA
        .fields
        .iter()
        .find(|field| field.name == "content")
        .map(|field| field.id)
    else {
        return false;
    };
    let Some(title_field_id) = DEFAULT_SCHEMA
        .fields
        .iter()
        .find(|field| field.name == "title")
        .map(|field| field.id)
    else {
        return false;
    };
    if content_field_id == title_field_id {
        return false;
    }

    let mut normalized_text = None;
    let mut saw_content = false;
    let mut saw_title = false;
    for clause in clauses {
        if clause.occur != Occur::Should {
            return false;
        }
        let Query::Term { fields, text } = &clause.query else {
            return false;
        };
        let [field] = fields.as_slice() else {
            return false;
        };
        if text.is_empty()
            || text.split_whitespace().count() != 1
            || normalize_query_text(text) != *text
        {
            return false;
        }
        if let Some(expected_text) = normalized_text {
            if text != expected_text {
                return false;
            }
        } else {
            normalized_text = Some(text.as_str());
        }

        if field.field_id == content_field_id && field.boost.to_bits() == 1.0_f32.to_bits() {
            if saw_content {
                return false;
            }
            saw_content = true;
        } else if field.field_id == title_field_id && field.boost.to_bits() == 2.0_f32.to_bits() {
            if saw_title {
                return false;
            }
            saw_title = true;
        } else {
            return false;
        }
    }

    saw_content && saw_title
}

fn query_contains_phrase(query: &Query) -> bool {
    match query {
        Query::Phrase { .. } => true,
        Query::Boolean { clauses, .. } => clauses
            .iter()
            .any(|clause| query_contains_phrase(&clause.query)),
        Query::Boost { query, .. } => query_contains_phrase(query),
        Query::Empty
        | Query::All
        | Query::Term { .. }
        | Query::Range { .. }
        | Query::Set { .. }
        | Query::Glob { .. } => false,
    }
}

fn query_contains_must_not(query: &Query) -> bool {
    match query {
        Query::Boolean { clauses, .. } => clauses
            .iter()
            .any(|clause| clause.occur == Occur::MustNot || query_contains_must_not(&clause.query)),
        Query::Boost { query, .. } => query_contains_must_not(query),
        Query::Empty
        | Query::All
        | Query::Term { .. }
        | Query::Phrase { .. }
        | Query::Range { .. }
        | Query::Set { .. }
        | Query::Glob { .. } => false,
    }
}

fn query_contains_explicit_boolean(query: &Query) -> bool {
    match query {
        Query::Boolean { clauses, operator } => {
            operator.is_some()
                || clauses
                    .iter()
                    .any(|clause| query_contains_explicit_boolean(&clause.query))
        }
        Query::Boost { query, .. } => query_contains_explicit_boolean(query),
        Query::Empty
        | Query::All
        | Query::Term { .. }
        | Query::Phrase { .. }
        | Query::Range { .. }
        | Query::Set { .. }
        | Query::Glob { .. } => false,
    }
}

fn hash_query_ast(hasher: &mut Sha256, query: &Query) {
    match query {
        Query::Empty => hasher.update([0]),
        Query::All => hasher.update([1]),
        Query::Term { fields, text } => {
            hasher.update([2]);
            hash_query_fields(hasher, fields);
            hash_len_prefixed(hasher, text.as_bytes());
        }
        Query::Phrase {
            fields,
            terms,
            slop,
            prefix,
        } => {
            hasher.update([3]);
            hash_query_fields(hasher, fields);
            hasher.update(usize_to_u64_infallible(terms.len()).to_le_bytes());
            for term in terms {
                hasher.update(term.position.to_le_bytes());
                hash_len_prefixed(hasher, term.text.as_bytes());
            }
            hasher.update(slop.to_le_bytes());
            hasher.update([u8::from(*prefix)]);
        }
        Query::Boolean { clauses, operator } => {
            hasher.update([4]);
            hasher.update([match operator {
                None => 0,
                Some(BooleanOperator::And) => 1,
                Some(BooleanOperator::Or) => 2,
            }]);
            hasher.update(usize_to_u64_infallible(clauses.len()).to_le_bytes());
            for clause in clauses {
                hasher.update([match clause.occur {
                    Occur::Must => 0,
                    Occur::Should => 1,
                    Occur::MustNot => 2,
                }]);
                hash_query_ast(hasher, &clause.query);
            }
        }
        Query::Range {
            field_id,
            lower,
            upper,
        } => {
            hasher.update([5]);
            hasher.update(field_id.to_le_bytes());
            hash_query_bound(hasher, lower);
            hash_query_bound(hasher, upper);
        }
        Query::Set { field_id, values } => {
            hasher.update([6]);
            hasher.update(field_id.to_le_bytes());
            hasher.update(usize_to_u64_infallible(values.len()).to_le_bytes());
            for value in values {
                hash_query_value(hasher, value);
            }
        }
        Query::Glob { field_ids, pattern } => {
            hasher.update([7]);
            hasher.update(usize_to_u64_infallible(field_ids.len()).to_le_bytes());
            for field_id in field_ids {
                hasher.update(field_id.to_le_bytes());
            }
            hash_len_prefixed(hasher, pattern.as_bytes());
        }
        Query::Boost { query, factor } => {
            hasher.update([8]);
            hasher.update(factor.to_bits().to_le_bytes());
            hash_query_ast(hasher, query);
        }
    }
}

fn hash_query_fields(hasher: &mut Sha256, fields: &[frankensearch_quill::QueryField]) {
    hasher.update(usize_to_u64_infallible(fields.len()).to_le_bytes());
    for field in fields {
        hasher.update(field.field_id.to_le_bytes());
        hasher.update(field.boost.to_bits().to_le_bytes());
    }
}

fn hash_query_bound(hasher: &mut Sha256, bound: &Bound<QueryValue>) {
    match bound {
        Bound::Included(value) => {
            hasher.update([0]);
            hash_query_value(hasher, value);
        }
        Bound::Excluded(value) => {
            hasher.update([1]);
            hash_query_value(hasher, value);
        }
        Bound::Unbounded => hasher.update([2]),
    }
}

fn hash_query_value(hasher: &mut Sha256, value: &QueryValue) {
    match value {
        QueryValue::I64(value) => {
            hasher.update([0]);
            hasher.update(value.to_le_bytes());
        }
        QueryValue::U64(value) => {
            hasher.update([1]);
            hasher.update(value.to_le_bytes());
        }
        QueryValue::Str(value) => {
            hasher.update([2]);
            hash_len_prefixed(hasher, value.as_bytes());
        }
    }
}

fn validate_experiment_inputs(
    document_count: u64,
    k: usize,
    queries: &[Qg6QuerySpec],
) -> Result<(), Qg6HarnessError> {
    if document_count == 0 {
        return Err(Qg6HarnessError::InvalidSpec {
            reason: "QG-6 corpus must contain at least one document".to_owned(),
        });
    }
    if k == 0 || k > MAX_K {
        return Err(Qg6HarnessError::InvalidSpec {
            reason: "QG-6 result limit is outside 1..=100000".to_owned(),
        });
    }
    if queries.is_empty() || queries.len() > MAX_QUERY_COUNT {
        return Err(Qg6HarnessError::InvalidSpec {
            reason: "QG-6 query count is outside 1..=4096".to_owned(),
        });
    }
    for query in queries {
        query.validate_entry()?;
        if !query.supported_k.contains(&k) {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: format!("query {:?} does not admit k={k}", query.id),
            });
        }
    }
    let ids = queries
        .iter()
        .map(|query| query.id.as_str())
        .collect::<BTreeSet<_>>();
    let normalized = queries
        .iter()
        .map(|query| query.normalized_text_sha256.as_str())
        .collect::<BTreeSet<_>>();
    let asts = queries
        .iter()
        .map(|query| query.parsed_ast_sha256.as_str())
        .collect::<BTreeSet<_>>();
    if ids.len() != queries.len()
        || normalized.len() != queries.len()
        || asts.len() != queries.len()
    {
        return Err(Qg6HarnessError::InvalidSpec {
            reason: "QG-6 query IDs, normalized texts, and parsed ASTs must be unique".to_owned(),
        });
    }
    Ok(())
}

fn adapter_failure(
    phase: Qg6Phase,
    arm: Qg6ArmRole,
    query_id: &str,
    error: &str,
) -> Qg6HarnessError {
    Qg6HarnessError::AdapterFailure {
        phase,
        arm,
        query_id: query_id.to_owned(),
        error_sha256: sha256_hex(error.as_bytes()),
        error_bytes: error.len(),
    }
}

pub fn query_manifest_sha256(queries: &[Qg6QuerySpec]) -> String {
    let receipts = queries
        .iter()
        .map(Qg6QueryIdentityReceipt::from_query)
        .collect::<Vec<_>>();
    query_identity_manifest_sha256(receipts.iter())
}

pub fn query_identity_manifest_sha256<'a>(
    queries: impl IntoIterator<Item = &'a Qg6QueryIdentityReceipt>,
) -> String {
    let mut hasher = Sha256::new();
    hash_len_prefixed(&mut hasher, QG6_QUERY_MANIFEST_VERSION.as_bytes());
    hash_len_prefixed(&mut hasher, QG6_SAMPLING_FRAME.as_bytes());
    let mut ordered = queries.into_iter().collect::<Vec<_>>();
    ordered.sort_unstable_by(|left, right| left.query_id.cmp(&right.query_id));
    hasher.update(usize_to_u64_infallible(ordered.len()).to_le_bytes());
    for query in ordered {
        hash_query_identity_receipt(&mut hasher, query);
    }
    lower_hex(hasher.finalize())
}

fn hash_query_identity_receipt(hasher: &mut Sha256, query: &Qg6QueryIdentityReceipt) {
    hash_len_prefixed(hasher, query.query_id.as_bytes());
    hasher.update([query_class_tag(query.class)]);
    hash_len_prefixed(hasher, query.normalized_text_sha256.as_bytes());
    hash_len_prefixed(hasher, query.parsed_ast_sha256.as_bytes());
    hasher.update([query.coverage_row, query.coverage_column]);
    query.support_divergence.hash_into(hasher);
    for k in query.supported_k {
        hasher.update(usize_to_u64_infallible(k).to_le_bytes());
    }
    hash_len_prefixed(hasher, query.query_generator_revision.as_bytes());
    hash_len_prefixed(hasher, query.corpus_generator_revision.as_bytes());
}

const fn query_class_tag(class: PerfQueryClass) -> u8 {
    match class {
        PerfQueryClass::Identifier => 0,
        PerfQueryClass::ShortKeyword => 1,
        PerfQueryClass::NaturalLanguage => 2,
        PerfQueryClass::Phrase => 3,
        PerfQueryClass::Boolean => 4,
    }
}

fn hash_len_prefixed(hasher: &mut Sha256, value: &[u8]) {
    hasher.update(usize_to_u64_infallible(value.len()).to_le_bytes());
    hasher.update(value);
}

fn sha256_hex(bytes: &[u8]) -> String {
    lower_hex(Sha256::digest(bytes))
}

fn lower_hex(bytes: impl AsRef<[u8]>) -> String {
    let bytes = bytes.as_ref();
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn balanced_bools(count: usize, seed: u64) -> Result<Vec<bool>, Qg6HarnessError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Qg6HarnessError::InvalidSpec {
            reason: "QG-6 balanced schedule allocation failed".to_owned(),
        })?;
    values.extend((0..count).map(|index| index < count / 2));
    shuffle(&mut values, seed);
    Ok(values)
}

fn shuffle<T>(values: &mut [T], mut seed: u64) {
    for index in (1..values.len()).rev() {
        seed = splitmix64(seed);
        let modulus = usize_to_u64_infallible(index + 1);
        let swap_index = usize::try_from(seed % modulus).expect("shuffle index fits usize");
        values.swap(index, swap_index);
    }
}

const fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    state = (state ^ (state >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    state = (state ^ (state >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    state ^ (state >> 31)
}

fn monotonic_ns(origin: Instant) -> u64 {
    u64::try_from(origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn usize_to_u64(value: usize) -> Result<u64, Qg6HarnessError> {
    u64::try_from(value).map_err(|_| Qg6HarnessError::InvalidSpec {
        reason: "QG-6 count does not fit u64".to_owned(),
    })
}

fn usize_to_u64_infallible(value: usize) -> u64 {
    u64::try_from(value).expect("bounded QG-6 length fits u64")
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;

    #[derive(Debug)]
    struct FakeArm {
        role: Qg6ArmRole,
    }

    fn queries() -> Vec<Qg6QuerySpec> {
        vec![
            Qg6QuerySpec::new("identifier-0", "term00042").expect("query"),
            Qg6QuerySpec::new("identifier-1", "term00137").expect("query"),
            Qg6QuerySpec::new("identifier-2", "term00256").expect("query"),
            Qg6QuerySpec::new("identifier-3", "term00301").expect("query"),
        ]
    }

    fn prepare() -> Qg6PreparedExperiment<FakeArm> {
        Qg6PreparedExperiment::prepare_with(
            "a".repeat(64),
            "b".repeat(64),
            100_000,
            10,
            queries(),
            |role, identity, setup| {
                setup.record_population_batch(identity.document_count / 2);
                setup
                    .record_population_batch(identity.document_count - identity.document_count / 2);
                setup.record_commit();
                Ok(FakeArm { role })
            },
        )
        .expect("prepared experiment")
    }

    fn normative_manifest() -> Vec<Qg6QuerySpec> {
        build_normative_query_manifest().expect("frozen QG-6 manifest")
    }

    fn canonical_result(query: &Qg6QuerySpec) -> Vec<String> {
        vec![
            format!("{}-doc-0", query.id()),
            format!("{}-doc-1", query.id()),
        ]
    }

    fn ranked_result(
        query: &Qg6QuerySpec,
        score_bits: u32,
        total_count: u64,
        k: usize,
    ) -> Qg6SearchResult {
        let returned_count = usize::try_from(total_count)
            .expect("test total count")
            .min(k);
        Qg6SearchResult::from_ranked_hits(
            (0..returned_count)
                .map(|index| {
                    Qg6SearchHit::new(format!("{}-ranked-doc-{index}", query.id()), score_bits)
                })
                .collect(),
            total_count,
            100_000,
        )
    }

    fn residual_digest(value: u64) -> String {
        format!("{value:064x}")
    }

    fn residual_observation(
        scheduled: &Qg6ResidualWilliamsLeaf,
        started_ns: u64,
    ) -> Qg6ResidualLeafObservation {
        let family = u64::try_from(scheduled.role.index() / 2).expect("family fits u64");
        let source_sha256 = residual_digest(10 + family);
        let cargo_lock_sha256 = residual_digest(20);
        let timing_elf_sha256 = residual_digest(30);
        let query_id = "identifier-0".to_owned();
        let query_contract_sha256 = residual_digest(71);
        let parsed_ast_sha256 = residual_digest(73);
        let role_index = u64::try_from(scheduled.role.index()).expect("role fits u64");
        let execution_ordinal = scheduled.sweep * QG6_R1_RESIDUAL_ROLE_COUNT_U8 + scheduled.ordinal;
        let ranked_cache_nonce_sha256 = residual_digest(300 + u64::from(execution_ordinal));
        let ranked_cache_key_sha256 = residual_digest(400 + u64::from(execution_ordinal));
        let result_envelope_sha256 = residual_digest(72);
        let boundary_effect_limit_ns = 10;
        let mut observation = Qg6ResidualLeafObservation {
            leaf_id: scheduled.leaf_id,
            meta_block_id: scheduled.meta_block_id,
            sweep: scheduled.sweep,
            ordinal: scheduled.ordinal,
            role: scheduled.role,
            stratum: Qg6ResidualStratum::SteadyRankedCacheMiss,
            query_class: PerfQueryClass::Identifier,
            query_id: query_id.clone(),
            instance_receipt_sha256: residual_digest(40 + role_index),
            backing_sha256: residual_digest(50 + family),
            backing_instance_receipt_sha256: residual_digest(53 + role_index),
            ranked_cache_receipt_sha256: residual_digest(55 + role_index),
            path_receipt_sha256: residual_digest(60 + role_index),
            source_elf_consistency_sha256: qg6_residual_source_elf_consistency_sha256(
                &source_sha256,
                &cargo_lock_sha256,
                &timing_elf_sha256,
            )
            .expect("valid source/ELF binding"),
            source_build_receipt_sha256: residual_digest(80 + role_index),
            source_sha256,
            cargo_lock_sha256,
            timing_elf_sha256,
            fixture_sha256: residual_digest(70),
            standardized_workload_sha256: qg6_residual_standardized_workload_sha256(
                &query_id,
                &query_contract_sha256,
            )
            .expect("valid standardized workload"),
            query_contract_sha256,
            parsed_ast_sha256: parsed_ast_sha256.clone(),
            result_envelope_sha256: result_envelope_sha256.clone(),
            ranked_cache_key_sha256: ranked_cache_key_sha256.clone(),
            ranked_cache_nonce_sha256: ranked_cache_nonce_sha256.clone(),
            raw_query_sha256: residual_digest(500 + u64::from(execution_ordinal)),
            raw_query_length_bytes: 32,
            ranked_miss_semantics_sha256: residual_ranked_miss_semantics_sha256(
                &query_id,
                &parsed_ast_sha256,
                32,
                &ranked_cache_nonce_sha256,
                &ranked_cache_key_sha256,
                &result_envelope_sha256,
                1,
                100,
                1,
                false,
                QG6_R1_RESIDUAL_COMPLETED_OUTCOME_CODE,
            )
            .expect("valid ranked-miss semantics"),
            cache_disposition: Qg6ResidualCacheDisposition::RankedMiss,
            generation: 3,
            work_units: 1,
            fuel_budget: 100,
            fuel_consumed: 1,
            cancelled: false,
            outcome_code: QG6_R1_RESIDUAL_COMPLETED_OUTCOME_CODE.to_owned(),
            ranked_cache_lookup_receipt_sha256: residual_digest(600 + u64::from(execution_ordinal)),
            host_receipt_sha256: residual_digest(90),
            boot_receipt_sha256: residual_digest(92),
            clock_receipt_sha256: residual_digest(91),
            invocation_receipt_sha256: residual_digest(0),
            generation_rebind_evidence: None,
            boundary_predecessor_leaf_id: None,
            boundary_predecessor_instance_receipt_sha256: None,
            boundary_effect_ns: 0,
            boundary_effect_limit_ns,
            boundary_effect_receipt_sha256: residual_boundary_effect_sha256(
                None,
                None,
                0,
                boundary_effect_limit_ns,
            )
            .expect("valid initial boundary receipt"),
            execution_ordinal,
            started_ns,
            ended_ns: started_ns
                + 100
                + u64::try_from(scheduled.role.index()).expect("role fits u64"),
            latency_ns: 100 + u64::try_from(scheduled.role.index()).expect("role fits u64"),
        };
        observation.invocation_receipt_sha256 =
            residual_invocation_receipt_sha256(&observation).expect("valid invocation receipt");
        observation
    }

    fn finalize_residual_execution_order(observations: &mut [Qg6ResidualLeafObservation]) {
        observations.sort_unstable_by_key(|leaf| leaf.execution_ordinal);
        let mut next_started_ns = 0_u64;
        let mut predecessor = None;
        for leaf in observations {
            leaf.started_ns = next_started_ns;
            leaf.ended_ns = leaf.started_ns + leaf.latency_ns;
            if let Some((predecessor_leaf_id, predecessor_instance_receipt_sha256)) = predecessor {
                leaf.boundary_predecessor_leaf_id = Some(predecessor_leaf_id);
                leaf.boundary_predecessor_instance_receipt_sha256 =
                    Some(predecessor_instance_receipt_sha256);
                leaf.boundary_effect_ns = 1;
            } else {
                leaf.boundary_predecessor_leaf_id = None;
                leaf.boundary_predecessor_instance_receipt_sha256 = None;
                leaf.boundary_effect_ns = 0;
            }
            leaf.boundary_effect_receipt_sha256 = residual_boundary_effect_sha256(
                leaf.boundary_predecessor_leaf_id,
                leaf.boundary_predecessor_instance_receipt_sha256.as_deref(),
                leaf.boundary_effect_ns,
                leaf.boundary_effect_limit_ns,
            )
            .expect("valid boundary receipt");
            leaf.invocation_receipt_sha256 =
                residual_invocation_receipt_sha256(leaf).expect("valid invocation receipt");
            predecessor = Some((leaf.leaf_id, leaf.instance_receipt_sha256.clone()));
            next_started_ns = leaf.ended_ns + 1;
        }
    }

    fn admit_residual_with_authority(
        observations: Vec<Qg6ResidualLeafObservation>,
        trusted_observations: &[Qg6ResidualLeafObservation],
    ) -> Result<Qg6ResidualScheduleAdmission, Qg6ResidualValidationError> {
        let authority = Qg6ResidualProducerAuthority::test_fixture(trusted_observations)?;
        admit_qg6_residual_schedule_meta_block_with_authority(observations, &authority)
    }

    fn admit_residual(
        observations: Vec<Qg6ResidualLeafObservation>,
    ) -> Result<Qg6ResidualScheduleAdmission, Qg6ResidualValidationError> {
        let trusted_observations = observations.clone();
        admit_residual_with_authority(observations, &trusted_observations)
    }

    fn valid_residual_meta_block() -> Vec<Qg6ResidualLeafObservation> {
        let mut observations = qg6_residual_williams_schedule(7)
            .expect("Williams schedule")
            .iter()
            .enumerate()
            .map(|(index, scheduled)| {
                residual_observation(
                    scheduled,
                    u64::try_from(index).expect("bounded test index") * 1_000,
                )
            })
            .collect::<Vec<_>>();
        finalize_residual_execution_order(&mut observations);
        observations
    }

    fn refresh_ranked_miss_semantics(leaf: &mut Qg6ResidualLeafObservation) {
        leaf.ranked_miss_semantics_sha256 = residual_ranked_miss_semantics_sha256(
            &leaf.query_id,
            &leaf.parsed_ast_sha256,
            leaf.raw_query_length_bytes,
            &leaf.ranked_cache_nonce_sha256,
            &leaf.ranked_cache_key_sha256,
            &leaf.result_envelope_sha256,
            leaf.work_units,
            leaf.fuel_budget,
            leaf.fuel_consumed,
            leaf.cancelled,
            &leaf.outcome_code,
        )
        .expect("valid ranked-miss semantics");
    }

    fn valid_first_touch_meta_block() -> Vec<Qg6ResidualLeafObservation> {
        let mut observations = valid_residual_meta_block();
        for leaf in &mut observations {
            let execution_ordinal = u64::from(leaf.execution_ordinal);
            leaf.stratum = Qg6ResidualStratum::FirstTouch;
            leaf.cache_disposition = Qg6ResidualCacheDisposition::FirstTouch;
            leaf.instance_receipt_sha256 = residual_digest(500 + execution_ordinal);
            leaf.backing_instance_receipt_sha256 = residual_digest(600 + execution_ordinal);
            leaf.ranked_cache_receipt_sha256 = residual_digest(700 + execution_ordinal);
            leaf.path_receipt_sha256 = residual_digest(800 + execution_ordinal);
        }
        finalize_residual_execution_order(&mut observations);
        observations
    }

    fn set_residual_role_latencies(
        observations: &mut [Qg6ResidualLeafObservation],
        latency_ns_by_role: [u64; QG6_R1_RESIDUAL_ROLE_COUNT],
    ) {
        for leaf in observations.iter_mut() {
            leaf.latency_ns = latency_ns_by_role[leaf.role.index()];
        }
        finalize_residual_execution_order(observations);
    }

    #[test]
    fn schedule_is_deterministic_balanced_and_interleaves_three_comparisons() {
        let first = seeded_interleaved_six_arm_schedule(16, 2, 0x5155_494c).expect("schedule");
        let second = seeded_interleaved_six_arm_schedule(16, 2, 0x5155_494c).expect("schedule");
        assert_eq!(first, second);
        assert_eq!(first.len(), 16 * 2 * 3);

        let mut query_comparison_counts = BTreeMap::new();
        let mut first_counts = BTreeMap::new();
        let mut permutation_counts = BTreeMap::new();
        let mut unit_ids = BTreeSet::new();
        let (units, remainder) = first.as_chunks::<3>();
        assert!(remainder.is_empty());
        for unit in units {
            assert!(unit_ids.insert(unit[0].unit_id));
            assert!(unit.iter().all(|block| {
                block.unit_id == unit[0].unit_id && block.query_index == unit[0].query_index
            }));
            assert_eq!(
                unit.iter()
                    .map(|block| block.comparison)
                    .collect::<BTreeSet<_>>(),
                BTreeSet::from([
                    Qg6Comparison::TantivyNull,
                    Qg6Comparison::QuillNull,
                    Qg6Comparison::Effect,
                ])
            );
            assert_eq!(
                unit.iter()
                    .flat_map(|block| [block.first, block.second])
                    .collect::<BTreeSet<_>>(),
                BTreeSet::from(Qg6ArmRole::ALL)
            );
            *permutation_counts
                .entry([unit[0].comparison, unit[1].comparison, unit[2].comparison])
                .or_insert(0_usize) += 1;
            for block in unit {
                *query_comparison_counts
                    .entry((block.query_index, block.comparison))
                    .or_insert(0_usize) += 1;
                *first_counts.entry(block.first).or_insert(0_usize) += 1;
            }
        }
        assert_eq!(unit_ids.len(), 32);
        for query_index in 0..16 {
            for comparison in [
                Qg6Comparison::TantivyNull,
                Qg6Comparison::QuillNull,
                Qg6Comparison::Effect,
            ] {
                assert_eq!(query_comparison_counts[&(query_index, comparison)], 2);
            }
        }
        assert!(
            first_counts
                .values()
                .max()
                .expect("first counts")
                .abs_diff(*first_counts.values().min().expect("first counts"))
                <= 1
        );
        assert!(
            permutation_counts
                .values()
                .max()
                .expect("permutation counts")
                .abs_diff(
                    *permutation_counts
                        .values()
                        .min()
                        .expect("permutation counts")
                )
                <= 1
        );
    }

    #[test]
    fn preparation_builds_all_six_roles_independently_once() {
        let mut built_roles = Vec::new();
        let prepared = Qg6PreparedExperiment::prepare_with(
            "a".repeat(64),
            "b".repeat(64),
            100_000,
            10,
            queries(),
            |role, identity, setup| {
                built_roles.push(role);
                setup.record_population_batch(identity.document_count);
                setup.record_commit();
                Ok(FakeArm { role })
            },
        )
        .expect("six independent prepared roles");

        assert_eq!(built_roles, Qg6ArmRole::ALL);
        assert_eq!(
            built_roles.iter().copied().collect::<BTreeSet<_>>().len(),
            6
        );
        for role in Qg6ArmRole::ALL {
            assert_eq!(prepared.arms.get(role).role, role);
            assert_eq!(prepared.lifecycle.arm(role).build_calls, 1);
        }
    }

    #[test]
    fn schedule_authority_rejects_resealed_mutation_and_role_relabeling() {
        let mut preflight = |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize| {
            black_box(arm.role);
            Ok(Qg6SearchResult::from(canonical_result(query)))
        };
        let validated = prepare()
            .validate_exact_parity(&mut preflight)
            .expect("exact parity");
        let authority = validated
            .schedule_authority(2, 3, 0x5155_494c)
            .expect("pre-timing authority");
        assert_eq!(authority.schedule.len(), 4 * 2 * 3);
        authority.verify().expect("canonical authority");
        assert_eq!(
            Qg6ScheduleAuthority::for_experiment(
                authority.identity.clone(),
                authority.query_count,
                authority.rounds_per_query,
                authority.searches_per_sample,
                authority.schedule_seed,
            )
            .expect("external parent authority"),
            authority
        );

        let mut mutated = authority.clone();
        mutated.schedule[0].unit_id += 1;
        mutated.authority_sha256 = mutated.recomputed_sha256().expect("reseal mutation");
        assert!(mutated.verify().is_err());

        let mut relabeled = authority;
        relabeled.schedule[0].first = match relabeled.schedule[0].first {
            Qg6ArmRole::EffectTreatment => Qg6ArmRole::EffectControl,
            _ => Qg6ArmRole::EffectTreatment,
        };
        relabeled.authority_sha256 = relabeled.recomputed_sha256().expect("reseal relabeling");
        assert!(relabeled.verify().is_err());
    }

    #[test]
    fn residual_williams_schedule_and_complete_leaf_store_admit_one_joint_vector() {
        let schedule = qg6_residual_williams_schedule(7).expect("six-role schedule");
        validate_qg6_residual_williams_schedule(&schedule).expect("balanced Williams design");
        assert_eq!(schedule.len(), 36);

        let mut role_ordinal_counts = BTreeMap::new();
        let mut before_counts = BTreeMap::new();
        let mut predecessor_counts = BTreeMap::new();
        let (rows, _) = schedule.as_chunks::<QG6_R1_RESIDUAL_ROLE_COUNT>();
        for row in rows {
            for leaf in row {
                *role_ordinal_counts
                    .entry((leaf.role, leaf.ordinal))
                    .or_insert(0_usize) += 1;
            }
            for left in 0..6 {
                for right in left + 1..6 {
                    *before_counts
                        .entry((row[left].role, row[right].role))
                        .or_insert(0_usize) += 1;
                }
            }
            for pair in row.windows(2) {
                *predecessor_counts
                    .entry((pair[0].role, pair[1].role))
                    .or_insert(0_usize) += 1;
            }
        }
        for role in Qg6ResidualArmRole::ALL {
            for ordinal in 0_u8..6 {
                assert_eq!(role_ordinal_counts[&(role, ordinal)], 1);
            }
            for other in Qg6ResidualArmRole::ALL {
                if role == other {
                    continue;
                }
                assert_eq!(before_counts[&(role, other)], 3);
                assert_eq!(before_counts[&(other, role)], 3);
                assert_eq!(predecessor_counts[&(role, other)], 1);
            }
        }

        let admitted = admit_residual(valid_residual_meta_block())
            .expect("complete independent physical observations admit");
        assert_eq!(admitted.leaves.len(), 36);
        assert_eq!(
            admitted
                .leaves
                .iter()
                .map(|leaf| leaf.leaf_id)
                .collect::<BTreeSet<_>>()
                .len(),
            36
        );
        let log_latency = |latency_ns| Duration::from_nanos(latency_ns).as_secs_f64().ln();
        // Bit equality throughout: these assert that the estimator reproduced
        // the exact contrast, so an approximate comparison would admit a drift
        // the receipt seals would reject.
        assert_eq!(
            admitted.joint_contrasts.old_b_minus_old_a.to_bits(),
            (log_latency(101) - log_latency(100)).to_bits()
        );
        assert_eq!(
            admitted.joint_contrasts.current_b_minus_current_a.to_bits(),
            (log_latency(103) - log_latency(102)).to_bits()
        );
        assert_eq!(
            admitted.joint_contrasts.tantivy_b_minus_tantivy_a.to_bits(),
            (log_latency(105) - log_latency(104)).to_bits()
        );
        assert_eq!(
            admitted
                .joint_contrasts
                .current_mean_minus_old_mean
                .to_bits(),
            (f64::midpoint(log_latency(102), log_latency(103))
                - f64::midpoint(log_latency(100), log_latency(101)))
            .to_bits()
        );
        assert_eq!(
            admitted
                .joint_contrasts
                .current_mean_minus_tantivy_mean
                .to_bits(),
            (f64::midpoint(log_latency(102), log_latency(103))
                - f64::midpoint(log_latency(104), log_latency(105)))
            .to_bits()
        );
    }

    #[test]
    fn residual_estimator_recovers_exact_known_log_effect_and_nulls() {
        let mut observations = valid_residual_meta_block();
        set_residual_role_latencies(&mut observations, [100, 100, 80, 80, 160, 160]);

        let admitted = admit_residual(observations).expect("known effect admits");
        assert_eq!(
            admitted.joint_contrasts.old_b_minus_old_a.to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            admitted.joint_contrasts.current_b_minus_current_a.to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            admitted.joint_contrasts.tantivy_b_minus_tantivy_a.to_bits(),
            0.0_f64.to_bits()
        );
        let expected_current_old = Duration::from_nanos(80).as_secs_f64().ln()
            - Duration::from_nanos(100).as_secs_f64().ln();
        let expected_current_tantivy = Duration::from_nanos(80).as_secs_f64().ln()
            - Duration::from_nanos(160).as_secs_f64().ln();
        assert_eq!(
            admitted
                .joint_contrasts
                .current_mean_minus_old_mean
                .to_bits(),
            expected_current_old.to_bits()
        );
        assert_eq!(
            admitted
                .joint_contrasts
                .current_mean_minus_tantivy_mean
                .to_bits(),
            expected_current_tantivy.to_bits()
        );
        assert!(
            (admitted.joint_contrasts.current_mean_minus_old_mean - 0.8_f64.ln()).abs() < 1e-12
        );
        assert!(
            (admitted.joint_contrasts.current_mean_minus_tantivy_mean - 0.5_f64.ln()).abs() < 1e-12
        );
    }

    #[test]
    fn residual_williams_validator_accepts_one_role_mapping_and_row_randomization() {
        let mut schedule = qg6_residual_williams_schedule(7).expect("schedule");
        let role_mapping = [
            Qg6ResidualArmRole::CurrentA,
            Qg6ResidualArmRole::TantivyA,
            Qg6ResidualArmRole::OldB,
            Qg6ResidualArmRole::CurrentB,
            Qg6ResidualArmRole::TantivyB,
            Qg6ResidualArmRole::OldA,
        ];
        for leaf in &mut schedule {
            leaf.role = role_mapping[leaf.role.index()];
        }
        schedule.rotate_left(QG6_R1_RESIDUAL_ROLE_COUNT);
        validate_qg6_residual_williams_schedule(&schedule)
            .expect("one role mapping and arbitrary row order remain a Williams design");
    }

    #[test]
    fn residual_admission_accepts_randomized_execution_row_order() {
        let mut observations = valid_residual_meta_block();
        for leaf in &mut observations {
            leaf.execution_ordinal = (QG6_R1_RESIDUAL_ROLE_COUNT_U8 - 1 - leaf.sweep)
                * QG6_R1_RESIDUAL_ROLE_COUNT_U8
                + leaf.ordinal;
        }
        finalize_residual_execution_order(&mut observations);
        admit_residual(observations)
            .expect("execution row randomization remains estimable under the Williams design");
    }

    #[test]
    fn residual_admission_rejects_arbitrary_execution_leaf_permutation() {
        let mut observations = valid_residual_meta_block();
        observations.swap(0, 1);
        observations[0].execution_ordinal = 0;
        observations[1].execution_ordinal = 1;
        finalize_residual_execution_order(&mut observations);
        assert!(matches!(
            admit_residual(observations),
            Err(Qg6ResidualValidationError::InvalidLeaf { .. })
        ));
    }

    #[test]
    fn residual_williams_validator_rejects_forged_role_permutation() {
        let mut forged = qg6_residual_williams_schedule(7).expect("schedule");
        forged[0].role = forged[1].role;
        assert!(matches!(
            validate_qg6_residual_williams_schedule(&forged),
            Err(Qg6ResidualValidationError::InvalidSchedule { .. })
        ));
    }

    #[test]
    fn residual_admission_rejects_truncated_leaf_store() {
        let mut observations = valid_residual_meta_block();
        observations.pop();
        assert!(matches!(
            admit_residual(observations),
            Err(Qg6ResidualValidationError::InvalidLeaf { .. })
        ));
    }

    #[test]
    fn residual_admission_rejects_duplicate_leaf_id() {
        let mut observations = valid_residual_meta_block();
        observations[1].leaf_id = observations[0].leaf_id;
        assert!(matches!(
            admit_residual(observations),
            Err(Qg6ResidualValidationError::InvalidLeaf { .. })
        ));
    }

    #[test]
    fn residual_admission_rejects_shared_ranked_cache_receipt() {
        let mut observations = valid_residual_meta_block();
        let old_a_cache_receipt = observations
            .iter()
            .find(|leaf| leaf.role == Qg6ResidualArmRole::OldA)
            .expect("OldA observation")
            .ranked_cache_receipt_sha256
            .clone();
        for leaf in observations
            .iter_mut()
            .filter(|leaf| leaf.role == Qg6ResidualArmRole::CurrentA)
        {
            leaf.ranked_cache_receipt_sha256 = old_a_cache_receipt.clone();
        }
        finalize_residual_execution_order(&mut observations);
        assert!(matches!(
            admit_residual(observations),
            Err(Qg6ResidualValidationError::ProvenanceMismatch { .. })
        ));
    }

    #[test]
    fn residual_first_touch_requires_a_fresh_six_arm_set_for_every_sweep() {
        admit_residual(valid_first_touch_meta_block())
            .expect("fresh first-touch arms may vary across Williams sweeps");

        let mut reused = valid_first_touch_meta_block();
        let reused_instance = reused[0].instance_receipt_sha256.clone();
        reused[QG6_R1_RESIDUAL_ROLE_COUNT].instance_receipt_sha256 = reused_instance;
        finalize_residual_execution_order(&mut reused);
        assert!(matches!(
            admit_residual(reused),
            Err(Qg6ResidualValidationError::ProvenanceMismatch { .. })
        ));
    }

    #[test]
    fn residual_admission_separates_backing_bytes_from_physical_backing_identity() {
        admit_residual(valid_residual_meta_block())
            .expect("A/A backing bytes may match when physical backing receipts differ");

        let mut shared_backing_identity = valid_residual_meta_block();
        let old_a_backing_identity = shared_backing_identity
            .iter()
            .find(|leaf| leaf.role == Qg6ResidualArmRole::OldA)
            .expect("OldA observation")
            .backing_instance_receipt_sha256
            .clone();
        for leaf in shared_backing_identity
            .iter_mut()
            .filter(|leaf| leaf.role == Qg6ResidualArmRole::CurrentA)
        {
            leaf.backing_instance_receipt_sha256 = old_a_backing_identity.clone();
        }
        assert!(matches!(
            admit_residual(shared_backing_identity),
            Err(Qg6ResidualValidationError::ProvenanceMismatch { .. })
        ));
    }

    #[test]
    fn residual_steady_ranked_miss_requires_unique_constant_length_nonce_cache_keys() {
        let mut duplicate_nonce = valid_residual_meta_block();
        let nonce = duplicate_nonce[0].ranked_cache_nonce_sha256.clone();
        duplicate_nonce[1].ranked_cache_nonce_sha256 = nonce;
        refresh_ranked_miss_semantics(&mut duplicate_nonce[1]);
        assert!(matches!(
            admit_residual(duplicate_nonce),
            Err(Qg6ResidualValidationError::InvalidLeaf { .. })
        ));

        let mut variable_length = valid_residual_meta_block();
        variable_length[1].raw_query_length_bytes = 31;
        refresh_ranked_miss_semantics(&mut variable_length[1]);
        assert!(matches!(
            admit_residual(variable_length),
            Err(Qg6ResidualValidationError::InvalidLeaf { .. })
        ));
    }

    #[test]
    fn residual_admission_rejects_coordinated_source_lock_elf_substitution() {
        let trusted_observations = valid_residual_meta_block();
        let mut observations = trusted_observations.clone();
        for leaf in &mut observations {
            let family = u64::try_from(leaf.role.index() / 2).expect("family fits u64");
            leaf.source_sha256 = residual_digest(110 + family);
            leaf.cargo_lock_sha256 = residual_digest(120);
            leaf.timing_elf_sha256 = residual_digest(130);
            leaf.source_build_receipt_sha256 =
                residual_digest(140 + u64::try_from(leaf.role.index()).expect("role fits u64"));
            leaf.source_elf_consistency_sha256 = qg6_residual_source_elf_consistency_sha256(
                &leaf.source_sha256,
                &leaf.cargo_lock_sha256,
                &leaf.timing_elf_sha256,
            )
            .expect("coordinated leaf consistency");
        }
        assert!(matches!(
            admit_residual_with_authority(observations, &trusted_observations),
            Err(Qg6ResidualValidationError::ProvenanceMismatch { .. })
        ));
    }

    #[test]
    fn residual_admission_rejects_rebind_skew_and_forged_boundary_receipts() {
        let mut generation_skew = valid_residual_meta_block();
        for leaf in &mut generation_skew {
            leaf.stratum = Qg6ResidualStratum::GenerationRebind;
            leaf.cache_disposition = Qg6ResidualCacheDisposition::GenerationRebind;
        }
        generation_skew[1].generation += 1;
        assert!(matches!(
            admit_residual(generation_skew),
            Err(Qg6ResidualValidationError::InvalidLeaf { .. })
        ));

        let mut forged_boundary = valid_residual_meta_block();
        forged_boundary[1].boundary_predecessor_leaf_id = Some(forged_boundary[0].leaf_id + 1);
        forged_boundary[1].boundary_effect_receipt_sha256 = residual_boundary_effect_sha256(
            forged_boundary[1].boundary_predecessor_leaf_id,
            forged_boundary[1]
                .boundary_predecessor_instance_receipt_sha256
                .as_deref(),
            forged_boundary[1].boundary_effect_ns,
            forged_boundary[1].boundary_effect_limit_ns,
        )
        .expect("self-consistent forged boundary receipt");
        assert!(matches!(
            admit_residual(forged_boundary),
            Err(Qg6ResidualValidationError::InvalidLeaf { .. })
        ));
    }

    #[test]
    fn residual_admission_rejects_work_outcome_and_environment_binding_gaps() {
        let mut zero_work = valid_residual_meta_block();
        zero_work[0].work_units = 0;
        refresh_ranked_miss_semantics(&mut zero_work[0]);
        assert!(matches!(
            admit_residual(zero_work),
            Err(Qg6ResidualValidationError::InvalidLeaf { .. })
        ));

        let mut cancelled = valid_residual_meta_block();
        cancelled[0].cancelled = true;
        refresh_ranked_miss_semantics(&mut cancelled[0]);
        assert!(matches!(
            admit_residual(cancelled),
            Err(Qg6ResidualValidationError::InvalidLeaf { .. })
        ));

        let mut changed_host = valid_residual_meta_block();
        changed_host[0].host_receipt_sha256 = residual_digest(92);
        assert!(matches!(
            admit_residual(changed_host),
            Err(Qg6ResidualValidationError::InvalidLeaf { .. })
        ));
    }

    #[test]
    fn residual_admission_rejects_overlapped_intervals() {
        let mut observations = valid_residual_meta_block();
        observations[1].started_ns = observations[0].ended_ns - 1;
        observations[1].ended_ns = observations[1].started_ns + observations[1].latency_ns;
        assert!(matches!(
            admit_residual(observations),
            Err(Qg6ResidualValidationError::InvalidLeaf { .. })
        ));
    }

    #[test]
    fn residual_admission_rejects_unbounded_latency_before_estimation() {
        let mut observations = valid_residual_meta_block();
        observations[0].started_ns = 0;
        observations[0].ended_ns = u64::MAX;
        observations[0].latency_ns = u64::MAX;
        assert!(matches!(
            admit_residual(observations),
            Err(Qg6ResidualValidationError::InvalidLeaf { .. })
        ));
    }

    #[test]
    fn residual_admission_rejects_standardized_workload_mutation() {
        let mut observations = valid_residual_meta_block();
        observations[0].standardized_workload_sha256 = residual_digest(97);
        assert!(matches!(
            admit_residual(observations),
            Err(Qg6ResidualValidationError::InvalidLeaf { .. })
        ));
    }

    #[test]
    fn residual_admission_rejects_source_elf_consistency_forgery() {
        let mut observations = valid_residual_meta_block();
        let current_a = observations
            .iter()
            .position(|leaf| leaf.role == Qg6ResidualArmRole::CurrentA)
            .expect("CurrentA leaf");
        observations[current_a].timing_elf_sha256 = residual_digest(99);
        assert!(matches!(
            admit_residual(observations),
            Err(Qg6ResidualValidationError::ProvenanceMismatch { .. })
        ));
    }

    #[test]
    fn residual_admission_rejects_mismatched_aa_causal_source() {
        let mut observations = valid_residual_meta_block();
        for leaf in observations
            .iter_mut()
            .filter(|leaf| leaf.role == Qg6ResidualArmRole::CurrentA)
        {
            leaf.source_sha256 = residual_digest(96);
            leaf.source_elf_consistency_sha256 = qg6_residual_source_elf_consistency_sha256(
                &leaf.source_sha256,
                &leaf.cargo_lock_sha256,
                &leaf.timing_elf_sha256,
            )
            .expect("valid consistency digest");
        }
        assert!(matches!(
            admit_residual(observations),
            Err(Qg6ResidualValidationError::ProvenanceMismatch { .. })
        ));
    }

    #[test]
    fn residual_admission_rejects_per_role_elf_even_when_consistency_recomputes() {
        let mut observations = valid_residual_meta_block();
        for leaf in observations
            .iter_mut()
            .filter(|leaf| leaf.role == Qg6ResidualArmRole::CurrentA)
        {
            leaf.timing_elf_sha256 = residual_digest(99);
            leaf.source_elf_consistency_sha256 = qg6_residual_source_elf_consistency_sha256(
                &leaf.source_sha256,
                &leaf.cargo_lock_sha256,
                &leaf.timing_elf_sha256,
            )
            .expect("valid consistency digest");
        }
        assert!(matches!(
            admit_residual(observations),
            Err(Qg6ResidualValidationError::ProvenanceMismatch { .. })
        ));
    }

    #[test]
    fn residual_admission_rejects_per_role_lockfile_even_when_consistency_recomputes() {
        let mut observations = valid_residual_meta_block();
        for leaf in observations
            .iter_mut()
            .filter(|leaf| leaf.role == Qg6ResidualArmRole::CurrentA)
        {
            leaf.cargo_lock_sha256 = residual_digest(98);
            leaf.source_elf_consistency_sha256 = qg6_residual_source_elf_consistency_sha256(
                &leaf.source_sha256,
                &leaf.cargo_lock_sha256,
                &leaf.timing_elf_sha256,
            )
            .expect("valid consistency digest");
        }
        assert!(matches!(
            admit_residual(observations),
            Err(Qg6ResidualValidationError::ProvenanceMismatch { .. })
        ));
    }

    #[test]
    fn frozen_manifest_has_eighty_unique_queries_and_twenty_equal_weight_cells() {
        let manifest = normative_manifest();
        validate_complete_query_manifest(&manifest).expect("complete frozen manifest");
        assert_eq!(manifest.len(), 5 * 16);
        assert_eq!(QG6_TOTAL_QUERY_COUNT, 80);
        assert!(Qg6QuerySpec::sampling_frame().contains("equal weight"));
        assert!(Qg6QuerySpec::sampling_frame().contains("never independent queries"));

        let mut ids = BTreeSet::new();
        let mut normalized = BTreeSet::new();
        let mut asts = BTreeSet::new();
        for class in PerfQueryClass::ALL {
            let class_queries = manifest
                .iter()
                .filter(|query| query.class() == class)
                .collect::<Vec<_>>();
            assert_eq!(class_queries.len(), 16);
            assert_eq!(
                class_queries
                    .iter()
                    .map(|query| (query.coverage_row(), query.coverage_column()))
                    .collect::<BTreeSet<_>>(),
                (0_u8..4)
                    .flat_map(|row| (0_u8..4).map(move |column| (row, column)))
                    .collect()
            );
            for query in class_queries {
                assert!(ids.insert(query.id()));
                assert!(normalized.insert(query.normalized_text_sha256.as_str()));
                assert!(asts.insert(query.parsed_ast_sha256.as_str()));
            }
        }

        let forward_hash = query_manifest_sha256(&manifest);
        assert_eq!(forward_hash, QG6_FROZEN_MANIFEST_SHA256);
        let mut reversed = manifest;
        reversed.reverse();
        assert_eq!(
            forward_hash,
            query_manifest_sha256(&reversed),
            "manifest identity must not depend on load order"
        );
        assert_eq!(
            forward_hash,
            Qg6QuerySpec::normative_manifest_sha256().expect("normative hash")
        );
    }

    #[test]
    fn frozen_short_keywords_accept_only_the_implicit_default_field_expansion() {
        for (index, seed) in SHORT_KEYWORD_QUERY_SEEDS.iter().enumerate() {
            let normalized = normalize_query_text(seed.text);
            let (_, parsed) = parsed_ast_sha256(&normalized, &format!("short_keyword-{index:02}"))
                .expect("frozen short keyword parses without recovery");
            assert!(
                is_implicit_short_keyword_default_field_expansion(&parsed.query),
                "frozen short keyword {index} did not use the exact default-field expansion"
            );
            validate_query_shape(
                PerfQueryClass::ShortKeyword,
                &parsed.query,
                &format!("short_keyword-{index:02}"),
            )
            .expect("frozen short keyword shape");
        }
    }

    #[test]
    fn short_keyword_shape_rejects_multi_token_explicit_mixed_and_duplicate_fields() {
        let parse = |query_id: &str, text: &str| {
            parsed_ast_sha256(&normalize_query_text(text), query_id)
                .expect("hostile shape parses without recovery")
                .1
                .query
        };
        let rejects = |query_id: &str, query: &Query| {
            assert!(
                validate_query_shape(PerfQueryClass::ShortKeyword, query, query_id).is_err(),
                "hostile short-keyword shape {query_id:?} was accepted"
            );
        };

        rejects(
            "short_keyword-multi-token",
            &parse("short_keyword-multi-token", "term00001 term00002"),
        );
        rejects(
            "short_keyword-explicit",
            &parse("short_keyword-explicit", "term00001 OR term00002"),
        );

        let mut mixed = parse("short_keyword-mixed", "term00001");
        let Query::Boolean { clauses, .. } = &mut mixed else {
            panic!("shipping one-token query must use implicit default-field expansion");
        };
        clauses[1].query = Query::All;
        rejects("short_keyword-mixed", &mixed);

        let mut duplicate_field = parse("short_keyword-duplicate-field", "term00001");
        let Query::Boolean { clauses, .. } = &mut duplicate_field else {
            panic!("shipping one-token query must use implicit default-field expansion");
        };
        let Query::Term {
            fields: first_fields,
            ..
        } = &clauses[0].query
        else {
            panic!("first default-field branch must be a term");
        };
        let first_fields = first_fields.clone();
        let Query::Term { fields, .. } = &mut clauses[1].query else {
            panic!("second default-field branch must be a term");
        };
        *fields = first_fields;
        rejects("short_keyword-duplicate-field", &duplicate_field);
    }

    #[test]
    fn frozen_manifest_reloads_and_independent_corpus_builds_are_deterministic() {
        let manifest = normative_manifest();
        let json = serde_json::to_vec(&manifest).expect("serialize manifest");
        let first: Vec<Qg6QuerySpec> = serde_json::from_slice(&json).expect("first fresh load");
        let second: Vec<Qg6QuerySpec> = serde_json::from_slice(&json).expect("second fresh load");
        validate_complete_query_manifest(&first).expect("first fresh load validates");
        validate_complete_query_manifest(&second).expect("second fresh load validates");
        assert_eq!(
            query_manifest_sha256(&first),
            query_manifest_sha256(&second)
        );

        let corpus_spec = crate::SyntheticCorpusSpec {
            seed: 0x5155_494c_4c50_4552,
            document_count: 256,
            vocabulary_size: 8_192,
            zipf_exponent: crate::ZipfExponent::S11,
            max_document_bytes: 4_096,
        };
        let first_corpus =
            crate::SyntheticCorpus::new(corpus_spec.clone()).expect("first corpus build");
        let second_corpus = crate::SyntheticCorpus::new(corpus_spec).expect("second corpus build");
        assert_eq!(
            first_corpus
                .manifest()
                .expect("first manifest")
                .content_sha256,
            second_corpus
                .manifest()
                .expect("second manifest")
                .content_sha256
        );
    }

    #[test]
    fn frozen_manifest_rejects_fifteen_or_seventeen_queries_per_class() {
        let mut fifteen = normative_manifest();
        fifteen.remove(0);
        assert!(validate_complete_query_manifest(&fifteen).is_err());

        let mut seventeen = normative_manifest();
        let mut extra = seventeen[0].clone();
        extra.id = "identifier-16".to_owned();
        extra.text = "qg6seventeenthidentifier".to_owned();
        let normalized = normalize_query_text(&extra.text);
        extra.normalized_text_sha256 = sha256_hex(normalized.as_bytes());
        extra.parsed_ast_sha256 = parsed_ast_sha256(&normalized, &extra.id)
            .expect("extra AST")
            .0;
        seventeen.push(extra);
        assert!(validate_complete_query_manifest(&seventeen).is_err());
    }

    #[test]
    fn frozen_manifest_rejects_alias_reclassification_hash_drift_and_unsupported() {
        let mut normalized_alias = normative_manifest();
        normalized_alias[1].text = normalized_alias[0].text.clone();
        normalized_alias[1].normalized_text_sha256 =
            normalized_alias[0].normalized_text_sha256.clone();
        normalized_alias[1].parsed_ast_sha256 = normalized_alias[0].parsed_ast_sha256.clone();
        assert!(validate_complete_query_manifest(&normalized_alias).is_err());

        let mut ast_alias = normative_manifest();
        ast_alias[1].text = ast_alias[0].text.to_ascii_uppercase();
        let normalized = normalize_query_text(&ast_alias[1].text);
        ast_alias[1].normalized_text_sha256 = sha256_hex(normalized.as_bytes());
        ast_alias[1].parsed_ast_sha256 = parsed_ast_sha256(&normalized, ast_alias[1].id())
            .expect("case-alias AST")
            .0;
        assert_ne!(
            ast_alias[0].normalized_text_sha256,
            ast_alias[1].normalized_text_sha256
        );
        assert_eq!(
            ast_alias[0].parsed_ast_sha256,
            ast_alias[1].parsed_ast_sha256
        );
        assert!(validate_complete_query_manifest(&ast_alias).is_err());

        let mut reclassified = normative_manifest();
        let reclassified_index = reclassified
            .iter()
            .position(|query| query.class != PerfQueryClass::Boolean)
            .expect("manifest contains a non-Boolean query");
        reclassified[reclassified_index].class = PerfQueryClass::Boolean;
        assert!(validate_complete_query_manifest(&reclassified).is_err());

        let mut drifted = normative_manifest();
        drifted[0].normalized_text_sha256 = "f".repeat(64);
        assert!(validate_complete_query_manifest(&drifted).is_err());

        let mut unsupported = normative_manifest();
        unsupported[0].support_divergence = Qg6SupportDivergence::Unsupported {
            reason_code: "unsupported_test_syntax".to_owned(),
        };
        let error = validate_complete_query_manifest(&unsupported)
            .expect_err("unsupported syntax fails before timing");
        assert!(error.to_string().contains("refusing silent skip"));
    }

    #[test]
    fn unsupported_reason_codes_are_bounded_stable_tokens() {
        for invalid in [
            "",
            "UPPERCASE",
            "contains space",
            "contains/slash",
            "reason_code_that_is_deliberately_longer_than_the_sixty_four_byte_contract_limit",
        ] {
            let mut query = normative_manifest()[0].clone();
            query.support_divergence = Qg6SupportDivergence::Unsupported {
                reason_code: invalid.to_owned(),
            };
            assert!(
                query.validate_entry().is_err(),
                "invalid reason token {invalid:?} was accepted"
            );
        }

        let mut bounded = normative_manifest()[0].clone();
        bounded.support_divergence = Qg6SupportDivergence::Unsupported {
            reason_code: "known_parser_gap.v1".to_owned(),
        };
        let error = bounded
            .validate_entry()
            .expect_err("explicitly unsupported queries always fail closed");
        let message = error.to_string();
        assert!(message.contains("refusing silent skip"));
        assert!(message.contains("reason_bytes=19"));
    }

    #[test]
    fn frozen_manifest_rejects_unknown_fields_and_raw_text_never_enters_log_identity() {
        let manifest = normative_manifest();
        let raw_text = manifest[0].text().to_owned();
        let debug = format!("{:?}", manifest[0]);
        let log_json =
            serde_json::to_string(&manifest[0].log_identity()).expect("serialize log identity");
        assert!(!debug.contains(&raw_text));
        assert!(!log_json.contains(&raw_text));

        let mut json = serde_json::to_value(&manifest[0]).expect("query JSON");
        json.as_object_mut()
            .expect("query object")
            .insert("raw_text_log".to_owned(), serde_json::json!(raw_text));
        assert!(
            serde_json::from_value::<Qg6QuerySpec>(json).is_err(),
            "unknown/raw-text logging fields must fail closed"
        );
    }

    #[test]
    fn unsupported_parser_syntax_and_unknown_field_syntax_fail_before_preparation() {
        let recovered =
            Qg6QuerySpec::new("phrase-hostile", "\"unterminated").expect_err("parser recovery");
        assert!(
            recovered
                .to_string()
                .contains("unsupported or recovered syntax")
        );

        let unknown_field = Qg6QuerySpec::new("identifier-hostile", "unknown_field:value")
            .expect_err("unknown field");
        assert!(
            unknown_field
                .to_string()
                .contains("unsupported or recovered syntax")
        );
    }

    #[test]
    fn normative_class_slice_has_sixteen_queries_and_rejects_silent_skip() {
        let class =
            Qg6QuerySpec::normative_for_class(PerfQueryClass::Boolean).expect("boolean slice");
        assert_eq!(class.len(), QG6_QUERY_GROUPS);
        let mut skipped = class;
        skipped.pop();
        assert_eq!(skipped.len(), 15);
        assert_ne!(
            query_manifest_sha256(&skipped),
            query_manifest_sha256(
                &Qg6QuerySpec::normative_for_class(PerfQueryClass::Boolean)
                    .expect("complete boolean slice")
            )
        );
    }

    #[test]
    fn integer_median_is_exact_for_odd_and_even_subsamples() {
        assert_eq!(median_sorted_u64(&[1, 3, 9]), 3);
        assert_eq!(median_sorted_u64(&[2, 4]), 3);
        assert_eq!(median_sorted_u64(&[u64::MAX - 1, u64::MAX]), u64::MAX - 1);
    }

    #[test]
    fn result_receipt_self_seal_binds_every_semantic_field() {
        let receipt = Qg6ResultReceipt::from_redacted_hits(
            vec![Qg6RankedHitReceipt {
                document_id_sha256: "a".repeat(64),
                score_bits: 1.0_f32.to_bits(),
            }],
            1,
            100_000,
            10,
        )
        .expect("sealed result receipt");
        let ranked_hit = &receipt.ordered_hits[0];
        let ranked_hit_json = serde_json::to_value(ranked_hit).expect("compact ranked-hit JSON");
        assert_eq!(
            ranked_hit_json,
            serde_json::json!([ranked_hit.document_id_sha256, ranked_hit.score_bits]),
            "ranked-hit receipts must remain compact ordered tuples"
        );
        assert_eq!(
            serde_json::from_value::<Qg6RankedHitReceipt>(ranked_hit_json)
                .expect("compact ranked-hit round trip"),
            *ranked_hit,
        );

        let mut mutations = Vec::new();
        let mut returned_count = receipt.clone();
        returned_count.returned_count += 1;
        mutations.push(returned_count);
        let mut document_id = receipt.clone();
        document_id.ordered_hits[0].document_id_sha256 = "b".repeat(64);
        mutations.push(document_id);
        let mut score_bits = receipt.clone();
        score_bits.ordered_hits[0].score_bits = 2.0_f32.to_bits();
        mutations.push(score_bits);
        let mut total_count = receipt.clone();
        total_count.total_count += 1;
        mutations.push(total_count);
        let mut doc_count = receipt.clone();
        doc_count.doc_count += 1;
        mutations.push(doc_count);
        let mut self_seal = receipt.clone();
        self_seal.receipt_sha256 = "c".repeat(64);
        mutations.push(self_seal);

        for mutation in mutations {
            assert!(mutation.verify(10, 100_000).is_err());
        }
        assert_ne!(
            qg6_result_sequence_sha256(&receipt, 1).expect("one result"),
            qg6_result_sequence_sha256(&receipt, 2).expect("two results")
        );
    }

    #[test]
    fn result_receipt_and_native_observation_reject_underfilled_top_k() {
        let redacted = Qg6ResultReceipt::from_redacted_hits(
            vec![Qg6RankedHitReceipt {
                document_id_sha256: "a".repeat(64),
                score_bits: 1.0_f32.to_bits(),
            }],
            7,
            100_000,
            10,
        )
        .expect_err("sealed receipts must reject underfilled top-k results");
        assert!(
            redacted
                .to_string()
                .contains("malformed or has an invalid self-seal")
        );

        let native = Qg6SearchResult::from_ranked_hits(
            vec![Qg6SearchHit::new("doc-0", 1.0_f32.to_bits())],
            7,
            100_000,
        );
        let observed = observe_result(
            native,
            10,
            100_000,
            Qg6Phase::Preflight,
            Qg6ArmRole::TantivyNullLeft,
            "identifier-underfill",
        )
        .err()
        .expect("native observations must reject underfilled top-k results");
        assert!(
            observed
                .to_string()
                .contains("not exactly min(k, total_count)")
        );
    }

    #[test]
    fn result_receipt_rejects_resealed_invalid_ranked_hits() {
        let receipt = Qg6ResultReceipt::from_redacted_hits(
            vec![
                Qg6RankedHitReceipt {
                    document_id_sha256: "a".repeat(64),
                    score_bits: 1.0_f32.to_bits(),
                },
                Qg6RankedHitReceipt {
                    document_id_sha256: "b".repeat(64),
                    score_bits: 0.5_f32.to_bits(),
                },
            ],
            2,
            100_000,
            10,
        )
        .expect("valid ranked receipt");

        let mut empty_id = receipt.clone();
        empty_id.ordered_hits[0].document_id_sha256 = EMPTY_DOCUMENT_ID_SHA256.to_owned();
        empty_id.receipt_sha256 = empty_id.canonical_sha256();

        let mut duplicate_id = receipt.clone();
        duplicate_id.ordered_hits[1].document_id_sha256 =
            duplicate_id.ordered_hits[0].document_id_sha256.clone();
        duplicate_id.receipt_sha256 = duplicate_id.canonical_sha256();

        let mut non_finite_score = receipt;
        non_finite_score.ordered_hits[0].score_bits = f32::NAN.to_bits();
        non_finite_score.receipt_sha256 = non_finite_score.canonical_sha256();

        for invalid in [empty_id, duplicate_id, non_finite_score] {
            assert!(
                invalid.verify(10, 100_000).is_err(),
                "fully resealed invalid ranked receipt escaped verification"
            );
        }
    }

    #[test]
    fn native_observation_rejects_empty_duplicate_and_non_finite_hits() {
        let cases = [
            (
                vec![Qg6SearchHit::new("", 1.0_f32.to_bits())],
                "returned document ID is empty",
            ),
            (
                vec![
                    Qg6SearchHit::new("doc-0", 1.0_f32.to_bits()),
                    Qg6SearchHit::new("doc-0", 0.5_f32.to_bits()),
                ],
                "returned document IDs are not unique",
            ),
            (
                vec![Qg6SearchHit::new("doc-0", f32::INFINITY.to_bits())],
                "returned score is not finite",
            ),
        ];

        for (hits, expected_reason) in cases {
            let total_count = u64::try_from(hits.len()).expect("test hit count");
            let error = observe_result(
                Qg6SearchResult::from_ranked_hits(hits, total_count, 100_000),
                10,
                100_000,
                Qg6Phase::Preflight,
                Qg6ArmRole::TantivyNullLeft,
                "identifier-invalid-hit",
            )
            .err()
            .expect("invalid native hit must fail before timing");
            assert!(
                error.to_string().contains(expected_reason),
                "unexpected error for {expected_reason}: {error}"
            );
        }
    }

    #[test]
    fn semantic_contract_canonicalizes_reversed_query_and_receipt_input() {
        let forward_queries = queries();
        let forward_receipts = forward_queries
            .iter()
            .enumerate()
            .map(|(index, _)| {
                let receipt = Qg6ResultReceipt::from_redacted_hits(
                    vec![Qg6RankedHitReceipt {
                        document_id_sha256: format!("{index:064x}"),
                        score_bits: u32::try_from(index).expect("score bits"),
                    }],
                    1,
                    100_000,
                    10,
                )
                .expect("result receipt");
                Qg6SixArmResultReceipts {
                    tantivy_null_left: receipt.clone(),
                    tantivy_null_right: receipt.clone(),
                    quill_null_left: receipt.clone(),
                    quill_null_right: receipt.clone(),
                    effect_control: receipt.clone(),
                    effect_treatment: receipt,
                }
            })
            .collect::<Vec<_>>();
        let identity = Qg6ExperimentIdentity {
            corpus_sha256: "a".repeat(64),
            query_manifest_sha256: query_manifest_sha256(&forward_queries),
            config_contract_sha256: "b".repeat(64),
            document_count: 100_000,
            k: 10,
        };
        let forward =
            Qg6SemanticContract::from_receipts(&identity, &forward_queries, &forward_receipts)
                .expect("forward contract");
        let mut reversed_queries = forward_queries;
        reversed_queries.reverse();
        let mut reversed_receipts = forward_receipts;
        reversed_receipts.reverse();
        let reversed =
            Qg6SemanticContract::from_receipts(&identity, &reversed_queries, &reversed_receipts)
                .expect("reversed contract");

        assert_eq!(forward, reversed);
        assert!(
            forward
                .groups
                .windows(2)
                .all(|pair| pair[0].query.query_id < pair[1].query.query_id)
        );
    }

    #[test]
    fn exact_parity_and_measurement_produce_complete_lifecycle_receipt() {
        let mut preflight = |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize| {
            black_box(arm.role);
            Ok(Qg6SearchResult::from(canonical_result(query)))
        };
        let validated = prepare()
            .validate_exact_parity(&mut preflight)
            .expect("exact parity");
        let authority = validated
            .schedule_authority(10, 1, 0x5eed)
            .expect("pre-timing schedule authority");
        let mut search = |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize, _phase: Qg6Phase| {
            black_box(arm.role);
            Ok(Qg6SearchResult::from(canonical_result(query)))
        };
        let measurement = validated
            .measure(2, &authority, &mut search)
            .expect("measurement");

        assert_eq!(measurement.samples.len(), 4 * 10 * 6);
        assert_eq!(measurement.schedule_authority, authority);
        measurement
            .verify_against_schedule_authority(&authority)
            .expect("external authority remains binding");
        assert_eq!(
            measurement
                .samples
                .iter()
                .map(|sample| sample.sample_id)
                .collect::<BTreeSet<_>>()
                .len(),
            measurement.samples.len()
        );
        for role in Qg6ArmRole::ALL {
            let lifecycle = measurement.lifecycle.arm(role);
            assert_eq!(lifecycle.build_calls, 1);
            assert_eq!(lifecycle.populate_calls, 2);
            assert_eq!(lifecycle.populated_documents, 100_000);
            assert_eq!(lifecycle.commit_calls, 1);
            assert_eq!(lifecycle.preflight_search_calls, 4);
            assert_eq!(lifecycle.warmup_search_calls, 8);
            assert_eq!(lifecycle.timed_search_calls, 40);
            assert_eq!(lifecycle.postflight_search_calls, 4);
            assert_eq!(lifecycle.timed_setup_calls, 0);
        }
        measurement
            .semantic_contract
            .verify()
            .expect("sealed semantic contract");

        let mut relabeled = measurement.clone();
        let replacement = match relabeled.samples[0].arm {
            Qg6ArmRole::EffectTreatment => Qg6ArmRole::EffectControl,
            _ => Qg6ArmRole::EffectTreatment,
        };
        relabeled.samples[0].arm = replacement;
        relabeled.samples[0].timing_leaves_sha256 = relabeled.samples[0]
            .recomputed_timing_leaves_sha256()
            .expect("reseal relabeled sample");
        relabeled.samples[0]
            .verify_timing_leaves()
            .expect("self-consistent relabeling");
        assert!(
            relabeled
                .verify_against_schedule_authority(&authority)
                .is_err(),
            "externally retained authority must reject self-consistent relabeling"
        );

        let mut mutated_id = measurement;
        mutated_id.samples[0].sample_id += 10_000;
        mutated_id.samples[0].timing_leaves_sha256 = mutated_id.samples[0]
            .recomputed_timing_leaves_sha256()
            .expect("reseal mutated sample");
        mutated_id.samples[0]
            .verify_timing_leaves()
            .expect("self-consistent sample-ID mutation");
        assert!(
            mutated_id
                .verify_against_schedule_authority(&authority)
                .is_err(),
            "externally retained authority must reject self-consistent sample-ID mutation"
        );
    }

    #[test]
    fn p50_subsamples_count_every_search_and_retain_one_sample_per_arm() {
        let mut preflight = |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize| {
            black_box(arm.role);
            Ok(Qg6SearchResult::from(canonical_result(query)))
        };
        let validated = prepare()
            .validate_exact_parity(&mut preflight)
            .expect("exact parity");
        let authority = validated
            .schedule_authority(2, 3, 0x5eed)
            .expect("pre-timing schedule authority");
        let mut search = |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize, _phase: Qg6Phase| {
            black_box(arm.role);
            Ok(Qg6SearchResult::from(canonical_result(query)))
        };
        let measurement = validated
            .measure_query_p50_with_normalizer(1, &authority, &mut search, &mut |result| result)
            .expect("p50 subsample measurement");

        assert_eq!(measurement.searches_per_sample, 3);
        assert_eq!(measurement.samples.len(), 4 * 2 * 6);
        assert!(measurement.samples.iter().all(|sample| {
            sample.subsample_count == 3
                && sample.observed_latency_ns > 0
                && sample.verify_timing_leaves().is_ok()
                && sample.timing_leaves.len() == 3
                && sample.timing_leaves.iter().all(|leaf| {
                    leaf.started_ns < leaf.ended_ns
                        && leaf.observed_latency_ms().is_finite()
                        && leaf.observed_latency_ms() > 0.0
                })
        }));
        let leaf = &measurement.samples[0].timing_leaves[0];
        assert_eq!(
            serde_json::to_value(leaf).expect("compact timing leaf JSON"),
            serde_json::json!([leaf.started_ns, leaf.ended_ns]),
            "timing leaves must remain compact numeric tuples without repeated parent facts"
        );
        let sample = &measurement.samples[0];
        let sample_json = serde_json::to_value(sample).expect("compact timed-sample JSON");
        let expected_leaf_wire = sample
            .timing_leaves
            .iter()
            .map(|leaf| {
                format!(
                    "{}:{}",
                    leaf.started_ns - sample.started_ns,
                    leaf.observed_latency_ns()
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(
            sample_json["timing_leaves"], expected_leaf_wire,
            "timed samples must encode parent-relative leaf starts and durations"
        );
        assert_eq!(
            serde_json::from_value::<Qg6TimedSample>(sample_json)
                .expect("compact timed-sample round trip"),
            *sample,
        );
        for role in Qg6ArmRole::ALL {
            assert_eq!(
                measurement.lifecycle.arm(role).timed_search_calls,
                4 * 2 * 3
            );
            assert_eq!(measurement.lifecycle.arm(role).postflight_search_calls, 4);
        }
    }

    #[test]
    fn timed_sample_relative_leaf_wire_round_trips_gaps_and_rejects_absolute_legacy_tuples() {
        let mut preflight = |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize| {
            black_box(arm.role);
            Ok(Qg6SearchResult::from(canonical_result(query)))
        };
        let validated = prepare()
            .validate_exact_parity(&mut preflight)
            .expect("exact parity");
        let authority = validated
            .schedule_authority(2, 3, 0x5eed)
            .expect("pre-timing schedule authority");
        let mut search = |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize, _phase: Qg6Phase| {
            black_box(arm.role);
            Ok(Qg6SearchResult::from(canonical_result(query)))
        };
        let measurement = validated
            .measure_query_p50_with_normalizer(1, &authority, &mut search, &mut |result| result)
            .expect("p50 subsample measurement");
        let mut sample = measurement.samples[0].clone();
        sample.started_ns = 1_000;
        sample.ended_ns = 1_050;
        sample.observed_latency_ns = 5;
        sample.timing_leaves = vec![
            Qg6SearchTimingLeafReceipt::from_interval(1_005, 1_008).expect("first leaf"),
            Qg6SearchTimingLeafReceipt::from_interval(1_020, 1_025).expect("second leaf"),
            Qg6SearchTimingLeafReceipt::from_interval(1_040, 1_047).expect("third leaf"),
        ];
        sample.timing_leaves_sha256 = sample
            .recomputed_timing_leaves_sha256()
            .expect("reseal gapped sample");
        sample
            .verify_timing_leaves()
            .expect("gapped sample remains valid");

        let compact = serde_json::to_value(&sample).expect("serialize relative leaf wire");
        assert_eq!(compact["timing_leaves"], serde_json::json!("5:3,20:5,40:7"));
        assert_eq!(
            serde_json::from_value::<Qg6TimedSample>(compact.clone())
                .expect("gapped relative leaf round trip"),
            sample,
        );

        let mut legacy_absolute = compact;
        legacy_absolute["timing_leaves"] = serde_json::json!("1005:1008,20:5,40:7");
        assert!(
            serde_json::from_value::<Qg6TimedSample>(legacy_absolute).is_err(),
            "absolute start/end tuples from the superseded wire must fail closed"
        );
        for malformed_wire in [
            "05:3,20:5,40:7",
            "5:03,20:5,40:7",
            "5:3,20:5,40:7,",
            "5::3,20:5,40:7",
            "18446744073709551616:3,20:5,40:7",
        ] {
            let mut malformed = serde_json::to_value(&sample).expect("serialize valid sample");
            malformed["timing_leaves"] = serde_json::json!(malformed_wire);
            assert!(
                serde_json::from_value::<Qg6TimedSample>(malformed).is_err(),
                "malformed compact timing wire escaped: {malformed_wire}"
            );
        }
    }

    #[test]
    fn timing_leaves_fail_closed_on_cardinality_order_interval_and_result_mutations() {
        let mut preflight = |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize| {
            black_box(arm.role);
            Ok(Qg6SearchResult::from(canonical_result(query)))
        };
        let validated = prepare()
            .validate_exact_parity(&mut preflight)
            .expect("exact parity");
        let authority = validated
            .schedule_authority(2, 3, 0x5eed)
            .expect("pre-timing schedule authority");
        let mut search = |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize, _phase: Qg6Phase| {
            black_box(arm.role);
            Ok(Qg6SearchResult::from(canonical_result(query)))
        };
        let measurement = validated
            .measure_query_p50_with_normalizer(1, &authority, &mut search, &mut |result| result)
            .expect("timing leaves");
        let sample = measurement.samples[0].clone();
        sample
            .verify_timing_leaves()
            .expect("honest timing leaves verify");

        let mut missing = sample.clone();
        missing.timing_leaves.pop();
        missing.timing_leaves_sha256 = missing
            .recomputed_timing_leaves_sha256()
            .expect("reseal missing leaves");
        assert!(missing.verify_timing_leaves().is_err());

        let mut reordered = sample.clone();
        reordered.timing_leaves.swap(0, 1);
        reordered.timing_leaves_sha256 = reordered
            .recomputed_timing_leaves_sha256()
            .expect("reseal reordered leaves");
        assert!(reordered.verify_timing_leaves().is_err());

        let mut invalid_interval = sample.clone();
        invalid_interval.timing_leaves[0].ended_ns = invalid_interval.timing_leaves[0].started_ns;
        invalid_interval.timing_leaves_sha256 = invalid_interval
            .recomputed_timing_leaves_sha256()
            .expect("reseal invalid interval leaves");
        assert!(invalid_interval.verify_timing_leaves().is_err());

        let mut changed_result = sample;
        changed_result.result_receipt_sha256 = "f".repeat(64);
        changed_result.timing_leaves_sha256 = changed_result
            .recomputed_timing_leaves_sha256()
            .expect("reseal changed result leaves");
        assert!(changed_result.verify_timing_leaves().is_err());
    }

    #[test]
    fn p50_subsamples_reject_zero_searches() {
        let mut preflight = |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize| {
            black_box(arm.role);
            Ok(Qg6SearchResult::from(canonical_result(query)))
        };
        let validated = prepare()
            .validate_exact_parity(&mut preflight)
            .expect("exact parity");
        let error = validated
            .schedule_authority(2, 0, 0x5eed)
            .expect_err("zero-sized p50 subsample");

        assert!(matches!(error, Qg6HarnessError::InvalidSpec { .. }));
    }

    #[test]
    fn p50_subsamples_reject_unbounded_timing_leaf_cardinality() {
        let mut preflight = |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize| {
            black_box(arm.role);
            Ok(Qg6SearchResult::from(canonical_result(query)))
        };
        let validated = prepare()
            .validate_exact_parity(&mut preflight)
            .expect("exact parity");
        let error = validated
            .schedule_authority(2, QG6_TIMED_SEARCHES_PER_SAMPLE + 1, 0x5eed)
            .expect_err("unbounded timing leaves must fail closed");

        assert!(matches!(error, Qg6HarnessError::InvalidSpec { .. }));
    }

    #[test]
    fn semantic_parity_retains_each_arms_native_result_receipt() {
        let native_result = |role: Qg6ArmRole, query: &Qg6QuerySpec| {
            let mut ids = canonical_result(query);
            if role == Qg6ArmRole::EffectTreatment {
                ids[1].push_str("-native-tie-choice");
            }
            (role, ids)
        };
        let mut preflight =
            |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize| Ok(native_result(arm.role, query));
        let mut normalize =
            |result: &(Qg6ArmRole, Vec<String>)| Qg6SearchResult::from(result.1.clone());
        let mut compare = |_query: &Qg6QuerySpec,
                           _expected_role: Qg6ArmRole,
                           expected: &(Qg6ArmRole, Vec<String>),
                           observed_role: Qg6ArmRole,
                           observed: &(Qg6ArmRole, Vec<String>)| {
            if observed_role == Qg6ArmRole::EffectTreatment || expected.1 == observed.1 {
                Ok(())
            } else {
                Err("non-treatment native result changed".to_owned())
            }
        };
        let validated = prepare()
            .validate_semantic_parity_with(&mut preflight, &mut normalize, &mut compare)
            .expect("semantic tie-envelope parity");
        let authority = validated
            .schedule_authority(2, 1, 0x5eed)
            .expect("pre-timing schedule authority");
        let mut timed_search =
            |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize, _phase: Qg6Phase| {
                Ok(Qg6SearchResult::from(native_result(arm.role, query).1))
            };
        let measurement = validated
            .measure(1, &authority, &mut timed_search)
            .expect("per-arm native receipts remain stable");

        assert!(
            measurement
                .samples
                .iter()
                .any(|sample| sample.arm == Qg6ArmRole::EffectTreatment)
        );
    }

    #[test]
    fn semantic_parity_keeps_tantivy_and_quill_nulls_independently_exact() {
        for drift_role in [Qg6ArmRole::TantivyNullRight, Qg6ArmRole::QuillNullRight] {
            let mut preflight = |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize| {
                let mut ids = canonical_result(query);
                if arm.role == drift_role {
                    ids.swap(0, 1);
                }
                Ok((arm.role, ids))
            };
            let mut normalize =
                |result: &(Qg6ArmRole, Vec<String>)| Qg6SearchResult::from(result.1.clone());
            let mut permissive_compare =
                |_query: &Qg6QuerySpec,
                 _expected_role: Qg6ArmRole,
                 _expected: &(Qg6ArmRole, Vec<String>),
                 _observed_role: Qg6ArmRole,
                 _observed: &(Qg6ArmRole, Vec<String>)| { Ok(()) };

            let error = prepare()
                .validate_semantic_parity_with(
                    &mut preflight,
                    &mut normalize,
                    &mut permissive_compare,
                )
                .err()
                .expect("same-engine null drift must fail before timing");
            assert!(matches!(
                error,
                Qg6HarnessError::OrderedDocIdsMismatch { observed_arm, .. }
                    if observed_arm == drift_role
            ));
        }
    }

    #[test]
    fn semantic_parity_failure_hashes_adapter_diagnostic() {
        let canary = "SECRET-SEMANTIC-PARITY-DIAGNOSTIC";
        let mut preflight = |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize| {
            Ok((arm.role, canonical_result(query)))
        };
        let mut normalize =
            |result: &(Qg6ArmRole, Vec<String>)| Qg6SearchResult::from(result.1.clone());
        let mut compare = |_query: &Qg6QuerySpec,
                           _expected_role: Qg6ArmRole,
                           _expected: &(Qg6ArmRole, Vec<String>),
                           observed_role: Qg6ArmRole,
                           _observed: &(Qg6ArmRole, Vec<String>)| {
            if observed_role == Qg6ArmRole::EffectTreatment {
                Err(canary.to_owned())
            } else {
                Ok(())
            }
        };
        let error = prepare()
            .validate_semantic_parity_with(&mut preflight, &mut normalize, &mut compare)
            .err()
            .expect("semantic parity rejection");
        assert!(matches!(
            error,
            Qg6HarnessError::SemanticParityFailure {
                observed_arm: Qg6ArmRole::EffectTreatment,
                ..
            }
        ));
        assert!(!error.to_string().contains(canary));
    }

    #[test]
    fn preflight_rejects_hit_count_mismatch_before_timing() {
        let mut search = |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize| {
            let mut result = canonical_result(query);
            if arm.role == Qg6ArmRole::EffectTreatment {
                result.pop();
            }
            Ok(Qg6SearchResult::from(result))
        };
        let error = prepare()
            .validate_exact_parity(&mut search)
            .err()
            .expect("count mismatch");
        assert!(matches!(
            error,
            Qg6HarnessError::HitCountMismatch {
                observed_arm: Qg6ArmRole::EffectTreatment,
                ..
            }
        ));
    }

    #[test]
    fn preflight_rejects_order_mismatch_without_exposing_doc_ids() {
        let canary = "SECRET-CANARY-DOC-ID";
        let mut search = |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize| {
            let mut result = vec![canary.to_owned(), format!("{}-other", query.id())];
            if arm.role == Qg6ArmRole::TantivyNullRight {
                result.swap(0, 1);
            }
            Ok(Qg6SearchResult::from(result))
        };
        let error = prepare()
            .validate_exact_parity(&mut search)
            .err()
            .expect("order mismatch");
        assert!(matches!(
            error,
            Qg6HarnessError::OrderedDocIdsMismatch {
                observed_arm: Qg6ArmRole::TantivyNullRight,
                first_differing_rank: 0,
                ..
            }
        ));
        assert!(!error.to_string().contains(canary));
    }

    #[test]
    fn preflight_rejects_claimed_digest_mismatch() {
        let mut search = |_arm: &FakeArm, query: &Qg6QuerySpec, _k: usize| {
            Ok(Qg6SearchResult::from(canonical_result(query)).with_claimed_sha256("f".repeat(64)))
        };
        let error = prepare()
            .validate_exact_parity(&mut search)
            .err()
            .expect("digest mismatch");
        assert!(matches!(
            error,
            Qg6HarnessError::ResultDigestMismatch {
                phase: Qg6Phase::Preflight,
                arm: Qg6ArmRole::TantivyNullLeft,
                ..
            }
        ));
    }

    #[test]
    fn measurement_rejects_result_drift_after_preflight() {
        let mut preflight = |_arm: &FakeArm, query: &Qg6QuerySpec, _k: usize| {
            Ok(Qg6SearchResult::from(canonical_result(query)))
        };
        let validated = prepare()
            .validate_exact_parity(&mut preflight)
            .expect("preflight");
        let authority = validated
            .schedule_authority(2, 1, 7)
            .expect("pre-timing schedule authority");
        let mut search = |_arm: &FakeArm, query: &Qg6QuerySpec, _k: usize, phase: Qg6Phase| {
            let mut result = canonical_result(query);
            if phase == Qg6Phase::Warmup {
                result[0].push_str("-drift");
            }
            Ok(Qg6SearchResult::from(result))
        };
        let error = validated
            .measure(1, &authority, &mut search)
            .expect_err("warmup drift");
        assert!(matches!(
            error,
            Qg6HarnessError::ResultDrift {
                phase: Qg6Phase::Warmup,
                ..
            }
        ));
    }

    #[test]
    fn postflight_rejects_score_and_total_count_drift_after_all_timing() {
        let mut preflight = |_arm: &FakeArm, query: &Qg6QuerySpec, k: usize| {
            Ok(ranked_result(query, 1.0_f32.to_bits(), 7, k))
        };
        let validated = prepare()
            .validate_exact_parity(&mut preflight)
            .expect("preflight");
        let authority = validated
            .schedule_authority(2, 1, 7)
            .expect("pre-timing schedule authority");
        let mut search = |_arm: &FakeArm, query: &Qg6QuerySpec, k: usize, phase: Qg6Phase| {
            if phase == Qg6Phase::Postflight {
                Ok(ranked_result(query, 2.0_f32.to_bits(), 8, k))
            } else {
                Ok(ranked_result(query, 1.0_f32.to_bits(), 7, k))
            }
        };
        let error = validated
            .measure(1, &authority, &mut search)
            .expect_err("postflight result drift");
        assert!(matches!(
            error,
            Qg6HarnessError::ResultDrift {
                phase: Qg6Phase::Postflight,
                ..
            }
        ));
    }

    #[test]
    fn setup_requires_exact_population_and_one_commit() {
        let error = Qg6PreparedExperiment::prepare_with(
            "a".repeat(64),
            "b".repeat(64),
            100,
            10,
            queries(),
            |role, _identity, setup| {
                setup.record_population_batch(99);
                setup.record_commit();
                Ok(FakeArm { role })
            },
        )
        .err()
        .expect("population mismatch");
        assert!(matches!(
            error,
            Qg6HarnessError::InvalidSetup {
                arm: Qg6ArmRole::TantivyNullLeft,
                ..
            }
        ));
    }

    #[test]
    fn filtered_fixture_is_always_no_claim() {
        assert_eq!(
            Qg6SelectionScope::from_cell_counts(1, 20).expect("filtered"),
            Qg6SelectionScope::FilteredPreAdmission
        );
        assert_eq!(
            Qg6SelectionScope::from_cell_counts(20, 20).expect("complete"),
            Qg6SelectionScope::CompleteGate
        );
        assert_eq!(
            Qg6SelectionScope::FilteredPreAdmission.claim(),
            Qg6SelectionClaim::NoClaim
        );
        assert_eq!(
            Qg6SelectionScope::CompleteGate.claim(),
            Qg6SelectionClaim::EligibleForGateValidation
        );
    }

    #[test]
    fn error_diagnostics_hash_adapter_text() {
        let canary = "SECRET-ADAPTER-FAILURE";
        let mut search = |_arm: &FakeArm, _query: &Qg6QuerySpec, _k: usize| Err(canary.to_owned());
        let error = prepare()
            .validate_exact_parity(&mut search)
            .err()
            .expect("adapter error");
        assert!(matches!(error, Qg6HarnessError::AdapterFailure { .. }));
        assert!(!error.to_string().contains(canary));
    }
}
