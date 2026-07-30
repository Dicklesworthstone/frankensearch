//! Prepared, parity-gated four-arm execution for the QG-6 query benchmark.
//!
//! The generic runner deliberately owns the lifecycle boundary: engines are
//! constructed and populated through [`Qg6PreparedExperiment::prepare_with`],
//! validated for exact or explicitly proven semantic result parity, warmed
//! equally, and only then exposed to the timed schedule. This keeps corpus
//! construction, commits, configuration, warmup, and parity checks outside
//! every timed interval.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::hint::black_box;
use std::ops::Bound;
use std::time::Instant;

use frankensearch_quill::{
    BooleanOperator, DEFAULT_SCHEMA, DefaultQueryParser, Occur, Query, QueryValue,
    canonicalize_query,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use unicode_normalization::UnicodeNormalization;

use crate::perf::{PerfQueryClass, QG6_QUERY_GROUPS};

const QG6_QUERY_MANIFEST_VERSION: &str = "frankensearch-qg6-query-manifest-v3";
const QG6_RESULT_RECEIPT_VERSION: &str = "frankensearch-qg6-result-receipt-v1";
const QG6_RESULT_SEQUENCE_VERSION: &str = "frankensearch-qg6-result-sequence-v1";
const QG6_SEMANTIC_CONTRACT_VERSION: &str = "frankensearch-qg6-semantic-contract-v1";
const QG6_QUERY_IDENTITY_VERSION: &str = "frankensearch-qg6-query-identity-v1";
const QG6_QUERY_GENERATOR_REVISION: &str = "frankensearch-qg6-frozen-80-query-generator-v2";
const QG6_CORPUS_GENERATOR_REVISION: &str =
    "frankensearch-quill-gauntlet/generator-v2;schema=2;zipf=s11;vocab=8192;max_doc=4096";
const QG6_FROZEN_MANIFEST_SHA256: &str =
    "4e9ed3dc59538a8f4fb8d100420fdb90e15cc64d4d7a6c8d5de7a3db1eaac1ca";
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

/// The four independent logical indexes in the QG-6 admission experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg6ArmRole {
    /// Left side of the Tantivy/Tantivy null comparison.
    NullLeft,
    /// Right side of the Tantivy/Tantivy null comparison.
    NullRight,
    /// Tantivy side of the Quill/Tantivy effect comparison.
    EffectControl,
    /// Quill side of the Quill/Tantivy effect comparison.
    EffectTreatment,
}

impl Qg6ArmRole {
    /// Stable order used by preparation, preflight, and lifecycle receipts.
    pub const ALL: [Self; 4] = [
        Self::NullLeft,
        Self::NullRight,
        Self::EffectControl,
        Self::EffectTreatment,
    ];

    const fn index(self) -> usize {
        match self {
            Self::NullLeft => 0,
            Self::NullRight => 1,
            Self::EffectControl => 2,
            Self::EffectTreatment => 3,
        }
    }
}

/// Which independently measured pair a scheduled block belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg6Comparison {
    /// Tantivy/Tantivy A/A null.
    Null,
    /// Tantivy/Quill A/B effect.
    Effect,
}

/// First or second position inside one paired timing block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qg6SampleOrder {
    /// First arm invoked in the block.
    First,
    /// Second arm invoked in the block.
    Second,
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg6RankedHitReceipt {
    /// SHA-256 of the external document ID.
    pub document_id_sha256: String,
    /// Exact IEEE-754 score bits returned by the native engine.
    pub score_bits: u32,
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

/// Named receipts for all four independent logical roles.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg6FourArmResultReceipts {
    /// Left Tantivy arm of the A/A null.
    pub null_left: Qg6ResultReceipt,
    /// Right Tantivy arm of the A/A null.
    pub null_right: Qg6ResultReceipt,
    /// Tantivy control arm of the A/B effect.
    pub effect_control: Qg6ResultReceipt,
    /// Quill treatment arm of the A/B effect.
    pub effect_treatment: Qg6ResultReceipt,
}

impl Qg6FourArmResultReceipts {
    /// Resolve a named logical role without relying on array position.
    #[must_use]
    pub const fn get(&self, role: Qg6ArmRole) -> &Qg6ResultReceipt {
        match role {
            Qg6ArmRole::NullLeft => &self.null_left,
            Qg6ArmRole::NullRight => &self.null_right,
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

/// Ordered query-to-group mapping and its complete four-role semantic receipts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qg6QueryGroupReceipt {
    /// Canonical zero-based query group.
    pub group_id: u64,
    /// Full recomputable redacted query identity.
    pub query: Qg6QueryIdentityReceipt,
    /// Full receipts for every native role.
    pub roles: Qg6FourArmResultReceipts,
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
    /// Canonically ordered query mapping and four-role results.
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
        expected_results: &[Qg6FourArmResultReceipts],
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
    if work_units == 0 || !is_lower_hex_sha256(&receipt.receipt_sha256) {
        return Err(Qg6HarnessError::InvalidSpec {
            reason: "QG-6 result sequence requires positive work and a valid receipt".to_owned(),
        });
    }
    let mut hasher = Sha256::new();
    hash_len_prefixed(&mut hasher, QG6_RESULT_SEQUENCE_VERSION.as_bytes());
    hasher.update(work_units.to_le_bytes());
    hash_len_prefixed(&mut hasher, receipt.receipt_sha256.as_bytes());
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
    pub arms: [Qg6ArmLifecycle; 4],
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

/// One paired block in the deterministic four-arm schedule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qg6PairBlock {
    /// Globally unique block identifier in this experiment.
    pub block_id: u64,
    /// Index into the frozen ordered query manifest.
    pub query_index: usize,
    /// Null or effect comparison.
    pub comparison: Qg6Comparison,
    /// Arm executed first.
    pub first: Qg6ArmRole,
    /// Arm executed second.
    pub second: Qg6ArmRole,
}

/// One directly observed timed sample.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qg6TimedSample {
    /// Paired block shared with exactly one other arm.
    pub block_id: u64,
    /// Unique sample identifier.
    pub sample_id: u64,
    /// Stable, non-sensitive query ID.
    pub query_id: String,
    /// Index into the frozen ordered query manifest.
    pub query_index: usize,
    /// Null or effect stream.
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
    /// Digest over every independently recomputed result receipt.
    pub result_sha256: String,
}

/// Output of one prepared four-arm measurement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qg6Measurement {
    /// Frozen identity shared by all four arms.
    pub identity: Qg6ExperimentIdentity,
    /// Seed that fully determines the timed schedule.
    pub schedule_seed: u64,
    /// Equal warmup count per arm and query.
    pub warmup_rounds: usize,
    /// Equal timed pair count per comparison and query.
    pub rounds_per_query: usize,
    /// Equal individually timed searches summarized by each arm sample.
    pub searches_per_sample: usize,
    /// Interleaved null/effect schedule.
    pub schedule: Vec<Qg6PairBlock>,
    /// Raw per-arm monotonic intervals.
    pub samples: Vec<Qg6TimedSample>,
    /// Lifecycle contamination proof.
    pub lifecycle: Qg6LifecycleReceipt,
    /// Sealed query mapping and full four-role semantic receipts.
    pub semantic_contract: Qg6SemanticContract,
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

struct Qg6FourArms<A> {
    null_left: A,
    null_right: A,
    effect_control: A,
    effect_treatment: A,
}

impl<A> Qg6FourArms<A> {
    fn get(&self, role: Qg6ArmRole) -> &A {
        match role {
            Qg6ArmRole::NullLeft => &self.null_left,
            Qg6ArmRole::NullRight => &self.null_right,
            Qg6ArmRole::EffectControl => &self.effect_control,
            Qg6ArmRole::EffectTreatment => &self.effect_treatment,
        }
    }
}

/// Four independently built arms before result parity has been established.
pub struct Qg6PreparedExperiment<A> {
    identity: Qg6ExperimentIdentity,
    queries: Vec<Qg6QuerySpec>,
    arms: Qg6FourArms<A>,
    lifecycle: Qg6LifecycleReceipt,
}

/// Prepared arms whose complete frozen query set passed result parity.
pub struct Qg6ValidatedExperiment<A> {
    prepared: Qg6PreparedExperiment<A>,
    expected_results: Vec<Qg6FourArmResultReceipts>,
}

impl<A> Qg6PreparedExperiment<A> {
    /// Build, populate, and commit four independent arms exactly once.
    ///
    /// The builder must record every population batch and the searchable
    /// commit through [`Qg6SetupRecorder`]. All four builders receive the same
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
        let null_left = build_one(Qg6ArmRole::NullLeft, &identity, &mut lifecycle, &mut build)?;
        let null_right = build_one(Qg6ArmRole::NullRight, &identity, &mut lifecycle, &mut build)?;
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
            arms: Qg6FourArms {
                null_left,
                null_right,
                effect_control,
                effect_treatment,
            },
            lifecycle,
        })
    }

    /// Validate every frozen query against all four arms before timing.
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
            let null_left = invoke_search(
                &self.arms,
                query,
                self.identity.k,
                self.identity.document_count,
                Qg6ArmRole::NullLeft,
                Qg6Phase::Preflight,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::NullLeft)
                .preflight_search_calls += 1;
            let null_right = invoke_search(
                &self.arms,
                query,
                self.identity.k,
                self.identity.document_count,
                Qg6ArmRole::NullRight,
                Qg6Phase::Preflight,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::NullRight)
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
                (Qg6ArmRole::NullRight, &null_right),
                (Qg6ArmRole::EffectControl, &effect_control),
                (Qg6ArmRole::EffectTreatment, &effect_treatment),
            ] {
                compare_exact(query.id(), Qg6ArmRole::NullLeft, &null_left, role, observed)?;
            }
            expected_results.push(Qg6FourArmResultReceipts {
                null_left: null_left.receipt,
                null_right: null_right.receipt,
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
    /// `compare` proves that each compared result is semantically equivalent
    /// to the baseline. This permits reviewed native tie-order differences
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
            let (null_left_native, null_left) = invoke_search_borrowed(
                &self.arms,
                query,
                self.identity.k,
                self.identity.document_count,
                Qg6ArmRole::NullLeft,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::NullLeft)
                .preflight_search_calls += 1;
            let (null_right_native, null_right) = invoke_search_borrowed(
                &self.arms,
                query,
                self.identity.k,
                self.identity.document_count,
                Qg6ArmRole::NullRight,
                search,
                normalize,
            )?;
            self.lifecycle
                .arm_mut(Qg6ArmRole::NullRight)
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
            for (role, observed) in [
                (Qg6ArmRole::NullRight, &null_right_native),
                (Qg6ArmRole::EffectControl, &effect_control_native),
                (Qg6ArmRole::EffectTreatment, &effect_treatment_native),
            ] {
                compare(
                    query,
                    Qg6ArmRole::NullLeft,
                    &null_left_native,
                    role,
                    observed,
                )
                .map_err(|error| {
                    semantic_parity_failure(query.id(), Qg6ArmRole::NullLeft, role, &error)
                })?;
            }
            expected_results.push(Qg6FourArmResultReceipts {
                null_left: null_left.receipt,
                null_right: null_right.receipt,
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
    /// Run equal warmups and an interleaved, balanced four-arm timing schedule.
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
        rounds_per_query: usize,
        schedule_seed: u64,
        search: &mut F,
    ) -> Result<Qg6Measurement, Qg6HarnessError>
    where
        F: FnMut(&A, &Qg6QuerySpec, usize, Qg6Phase) -> Result<Qg6SearchResult, String>,
    {
        self.measure_with_normalizer(
            warmup_rounds,
            rounds_per_query,
            schedule_seed,
            search,
            &mut |result| result,
        )
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
        rounds_per_query: usize,
        schedule_seed: u64,
        search: &mut F,
        normalize: &mut N,
    ) -> Result<Qg6Measurement, Qg6HarnessError>
    where
        F: FnMut(&A, &Qg6QuerySpec, usize, Qg6Phase) -> Result<R, String>,
        N: FnMut(R) -> Qg6SearchResult,
    {
        self.measure_query_p50_with_normalizer(
            warmup_rounds,
            rounds_per_query,
            1,
            schedule_seed,
            search,
            normalize,
        )
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
        rounds_per_query: usize,
        searches_per_sample: usize,
        schedule_seed: u64,
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
        if searches_per_sample == 0 {
            return Err(Qg6HarnessError::InvalidSpec {
                reason: "QG-6 prepared measurement requires at least one search per sample"
                    .to_owned(),
            });
        }
        let schedule = seeded_interleaved_four_arm_schedule(
            self.prepared.queries.len(),
            rounds_per_query,
            schedule_seed,
        )?;
        self.run_warmups(warmup_rounds, schedule_seed, search, normalize)?;

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
                let started_ns = monotonic_ns(origin);
                let mut latencies_ns = Vec::new();
                latencies_ns
                    .try_reserve_exact(searches_per_sample)
                    .map_err(|_| Qg6HarnessError::InvalidSpec {
                        reason: "QG-6 latency subsample allocation failed".to_owned(),
                    })?;
                for _ in 0..searches_per_sample {
                    let search_started_ns = monotonic_ns(origin);
                    let result = search(
                        self.prepared.arms.get(role),
                        black_box(query),
                        black_box(self.prepared.identity.k),
                        Qg6Phase::Measurement,
                    );
                    let mut search_ended_ns = monotonic_ns(origin);
                    if search_ended_ns <= search_started_ns {
                        search_ended_ns = search_started_ns.saturating_add(1);
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
                    latencies_ns.push(search_ended_ns.saturating_sub(search_started_ns).max(1));
                }
                let mut ended_ns = monotonic_ns(origin);
                if ended_ns <= started_ns {
                    ended_ns = started_ns.saturating_add(1);
                }
                latencies_ns.sort_unstable();
                let observed_latency_ns = median_sorted_u64(&latencies_ns);
                let sample_id = block
                    .block_id
                    .checked_mul(2)
                    .and_then(|base| base.checked_add(u64::from(order == Qg6SampleOrder::Second)))
                    .ok_or_else(|| Qg6HarnessError::InvalidSpec {
                        reason: "timed sample ID overflow".to_owned(),
                    })?;
                samples.push(Qg6TimedSample {
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
                    subsample_count: usize_to_u64(searches_per_sample)?,
                    result_sha256: qg6_result_sequence_sha256(
                        self.expected_results[block.query_index].get(role),
                        usize_to_u64(searches_per_sample)?,
                    )?,
                });
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
        Ok(Qg6Measurement {
            identity: self.prepared.identity,
            schedule_seed,
            warmup_rounds,
            rounds_per_query,
            searches_per_sample,
            schedule,
            samples,
            lifecycle: self.prepared.lifecycle,
            semantic_contract,
        })
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
/// Every query receives `rounds_per_query` null and effect blocks. Each
/// two-block unit contains one null and one effect comparison, while the
/// comparison order and both within-pair arm orders are independently balanced.
///
/// # Errors
///
/// Requires at least one query and two rounds per query, and rejects arithmetic
/// overflow.
pub fn seeded_interleaved_four_arm_schedule(
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
    let comparison_order = balanced_bools(unit_count, seed ^ 0x243f_6a88_85a3_08d3)?;
    let null_left_first = balanced_bools(unit_count, seed ^ 0x1319_8a2e_0370_7344)?;
    let effect_control_first = balanced_bools(unit_count, seed ^ 0xa409_3822_299f_31d0)?;
    let block_capacity = unit_count
        .checked_mul(2)
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
        let null = pair_roles(Qg6Comparison::Null, null_left_first[unit_index]);
        let effect = pair_roles(Qg6Comparison::Effect, effect_control_first[unit_index]);
        let pairs = if comparison_order[unit_index] {
            [null, effect]
        } else {
            [effect, null]
        };
        for (comparison, first, second) in pairs {
            schedule.push(Qg6PairBlock {
                block_id: usize_to_u64(schedule.len())?,
                query_index,
                comparison,
                first,
                second,
            });
        }
    }
    Ok(schedule)
}

fn pair_roles(
    comparison: Qg6Comparison,
    control_first: bool,
) -> (Qg6Comparison, Qg6ArmRole, Qg6ArmRole) {
    let (control, treatment) = match comparison {
        Qg6Comparison::Null => (Qg6ArmRole::NullLeft, Qg6ArmRole::NullRight),
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
    arms: &Qg6FourArms<A>,
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
    arms: &Qg6FourArms<A>,
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
    arms: &Qg6FourArms<A>,
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
    expected_results: &[Qg6FourArmResultReceipts],
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

pub(crate) fn query_manifest_sha256(queries: &[Qg6QuerySpec]) -> String {
    let receipts = queries
        .iter()
        .map(Qg6QueryIdentityReceipt::from_query)
        .collect::<Vec<_>>();
    query_identity_manifest_sha256(receipts.iter())
}

pub(crate) fn query_identity_manifest_sha256<'a>(
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

    #[test]
    fn schedule_is_deterministic_balanced_and_interleaves_comparisons() {
        let first = seeded_interleaved_four_arm_schedule(4, 11, 0x5155_494c).expect("schedule");
        let second = seeded_interleaved_four_arm_schedule(4, 11, 0x5155_494c).expect("schedule");
        assert_eq!(first, second);
        assert_eq!(first.len(), 4 * 11 * 2);

        let mut query_comparison_counts = BTreeMap::new();
        let mut first_counts = BTreeMap::new();
        for pair in first.as_chunks::<2>().0 {
            assert_ne!(pair[0].comparison, pair[1].comparison);
            for block in pair {
                *query_comparison_counts
                    .entry((block.query_index, block.comparison))
                    .or_insert(0_usize) += 1;
                *first_counts.entry(block.first).or_insert(0_usize) += 1;
            }
        }
        for query_index in 0..4 {
            assert_eq!(
                query_comparison_counts[&(query_index, Qg6Comparison::Null)],
                11
            );
            assert_eq!(
                query_comparison_counts[&(query_index, Qg6Comparison::Effect)],
                11
            );
        }
        assert!(
            first_counts
                .values()
                .max()
                .expect("first counts")
                .abs_diff(*first_counts.values().min().expect("first counts"))
                <= 1
        );
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
            Qg6ArmRole::NullLeft,
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
                Qg6ArmRole::NullLeft,
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
                Qg6FourArmResultReceipts {
                    null_left: receipt.clone(),
                    null_right: receipt.clone(),
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
        let mut search = |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize, _phase: Qg6Phase| {
            black_box(arm.role);
            Ok(Qg6SearchResult::from(canonical_result(query)))
        };
        let measurement = validated
            .measure(2, 10, 0x5eed, &mut search)
            .expect("measurement");

        assert_eq!(measurement.samples.len(), 4 * 10 * 4);
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
        let mut search = |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize, _phase: Qg6Phase| {
            black_box(arm.role);
            Ok(Qg6SearchResult::from(canonical_result(query)))
        };
        let measurement = validated
            .measure_query_p50_with_normalizer(1, 2, 3, 0x5eed, &mut search, &mut |result| result)
            .expect("p50 subsample measurement");

        assert_eq!(measurement.searches_per_sample, 3);
        assert_eq!(measurement.samples.len(), 4 * 2 * 4);
        assert!(
            measurement
                .samples
                .iter()
                .all(|sample| sample.subsample_count == 3 && sample.observed_latency_ns > 0)
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
    fn p50_subsamples_reject_zero_searches() {
        let mut preflight = |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize| {
            black_box(arm.role);
            Ok(Qg6SearchResult::from(canonical_result(query)))
        };
        let validated = prepare()
            .validate_exact_parity(&mut preflight)
            .expect("exact parity");
        let mut search = |_arm: &FakeArm, query: &Qg6QuerySpec, _k: usize, _phase: Qg6Phase| {
            Ok(Qg6SearchResult::from(canonical_result(query)))
        };
        let error = validated
            .measure_query_p50_with_normalizer(1, 2, 0, 0x5eed, &mut search, &mut |result| result)
            .expect_err("zero-sized p50 subsample");

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
        let mut timed_search =
            |arm: &FakeArm, query: &Qg6QuerySpec, _k: usize, _phase: Qg6Phase| {
                Ok(Qg6SearchResult::from(native_result(arm.role, query).1))
            };
        let measurement = validated
            .measure(1, 2, 0x5eed, &mut timed_search)
            .expect("per-arm native receipts remain stable");

        assert!(
            measurement
                .samples
                .iter()
                .any(|sample| sample.arm == Qg6ArmRole::EffectTreatment)
        );
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
            if arm.role == Qg6ArmRole::NullRight {
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
                observed_arm: Qg6ArmRole::NullRight,
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
                arm: Qg6ArmRole::NullLeft,
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
        let mut search = |_arm: &FakeArm, query: &Qg6QuerySpec, _k: usize, phase: Qg6Phase| {
            let mut result = canonical_result(query);
            if phase == Qg6Phase::Warmup {
                result[0].push_str("-drift");
            }
            Ok(Qg6SearchResult::from(result))
        };
        let error = validated
            .measure(1, 2, 7, &mut search)
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
        let mut search = |_arm: &FakeArm, query: &Qg6QuerySpec, k: usize, phase: Qg6Phase| {
            if phase == Qg6Phase::Postflight {
                Ok(ranked_result(query, 2.0_f32.to_bits(), 8, k))
            } else {
                Ok(ranked_result(query, 1.0_f32.to_bits(), 7, k))
            }
        };
        let error = validated
            .measure(1, 2, 7, &mut search)
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
                arm: Qg6ArmRole::NullLeft,
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
