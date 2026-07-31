//! Replay-verifiable differential campaign orchestration.
//!
//! The runner is deliberately adapter-first: it proves corpus/query manifest
//! integrity, cross-engine identity, and a shared semantic contract before
//! either engine is allowed to ingest. A live Quill adapter plugs into the
//! same boundary when the scalar G1a facade lands.

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;

use asupersync::Cx;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::Instrument as _;

use frankensearch_core::LexicalSearch;

use crate::GauntletError;
use crate::artifact::{
    ArtifactLexicalContractEvidence, ArtifactObject, ArtifactStore, CampaignArtifactContext,
};
use crate::comparator::{
    ComparatorConfig, ComparisonReport, ComparisonStatus, Divergence, DivergenceClass,
    EngineObservation, LexicalBackendIdentity, LexicalComparisonStatus,
    LexicalContractBuildContext, LexicalEngineRole, LexicalFieldMismatch, LexicalMismatchClass,
    LexicalProbeCoverage, LexicalSideCoverage, RankClass, compare_lexical_contracts,
    compare_observations, observe_live_lexical_contract,
};
#[cfg(feature = "tantivy-oracle")]
use crate::engine::GauntletEngine;
use crate::engine::{
    ComparisonMode, DifferentialCase, DifferentialCaseMetadata, EngineDescriptor,
    EnginePairIdentity, HarnessRun, MAX_SNIPPET_CHARS,
};
#[cfg(feature = "tantivy-oracle")]
use crate::generator::GeneratedSourceFilter;
use crate::generator::{
    CorpusManifest, CorpusSourceManifest, GENERATOR_ID, GeneratedDocument, GeneratedQueryCase,
    GeneratedQueryKind, GeneratedQuerySuite, GlobPatternClass, MAX_DOCUMENT_BYTES, MAX_QUERY_CASES,
    MAX_QUERY_ID_BYTES, QUERY_MANIFEST_SCHEMA_VERSION, QueryManifest, QuerySuiteSource,
    QuerySyntax, RangeClass, StructuredFilterClass, SyntheticCorpus, is_canonical_query_id,
};
use crate::version_contract::oracle_version_contract;

/// Schema version for deterministic campaign reports.
pub const CAMPAIGN_REPORT_SCHEMA_VERSION: u32 = 5;
/// Schema version for the append-only machine-readable Divergence Register.
pub const DIVERGENCE_REGISTER_LEDGER_SCHEMA_VERSION: u32 = 1;
/// Redaction policy required by committed Divergence Register evidence.
pub const DIVERGENCE_REGISTER_REDACTION_POLICY_VERSION: &str = "quill-divergence-redaction-v1";
/// Canonical preimage for the default shipping lexical analyzer protocol.
pub const DEFAULT_ANALYZER_CONTRACT_PREIMAGE: &str =
    "v1;tokenizer=frankensearch_default;split=unicode_alphanumeric;lowercase=unicode_to_lowercase";
/// Default lexical analyzer protocol implemented by the shipping Tantivy adapter.
pub const DEFAULT_ANALYZER_CONTRACT_HASH: &str =
    "7425c0f2d0a909ca4103bd20f439b6282d3ce00ab3c9f6784ec7333398197041";
/// Canonical preimage for the default shipping schema, parser, and rank protocol.
pub const DEFAULT_SCHEMA_CONTRACT_PREIMAGE: &str = "v2;id=text:string+stored;content=text:frankensearch_default+freqs_positions+stored;title=text:frankensearch_default+freqs_positions+stored;metadata_json=text:stored;ord=u64:fast+stored;query_parser=default_fields(content,title);title_boost_bits=1073741824;default_operator=or;max_query_chars=10000;bm25=tantivy-0.26.1-default;pagination=offset_then_limit;counts=exact;snippets=tantivy-html-configured";
/// Scalar G1a subset: identical lexical semantics with snippet evidence disabled.
pub const SCALAR_G1A_SCHEMA_CONTRACT_PREIMAGE: &str = "v1;profile=scalar-g1a;id=text:string+stored;content=text:frankensearch_default+freqs_positions+stored;title=text:frankensearch_default+freqs_positions+stored;metadata_json=text:stored;ord=u64:fast+stored;query_parser=default-fields-term-multiterm-exact-phrase-boolean;title_boost_bits=1073741824;default_operator=or;max_query_chars=10000;bm25=tantivy-0.26.1-default;pagination=offset_then_limit;counts=exact;snippets=disabled";
/// Canonical preimage for the CASS hyphen/prefix analyzer protocol.
pub const CASS_ANALYZER_CONTRACT_PREIMAGE: &str = "v2;tokenizer=cass_hyphen_normalize;token_runs=ascii_alphanumeric_with_interior_hyphens_or_pinned_cjk;hyphen_tokens=compound_and_parts_same_position;cjk_tokens=overlapping_bigrams_or_singleton;normalize=ascii_lowercase;max_token_bytes=inclusive-256;prefix_tokenizer=cass_prefix_normalize_without_hyphen_decomposition;prefix_source_split=unicode_alphanumeric;prefix_edge_ngrams=2..20-unicode-scalars;prefix_cjk=overlapping-bigrams";
/// Canonical CASS field, parser, filtering, ranking, and pagination protocol.
pub const CASS_SCHEMA_CONTRACT_PREIMAGE: &str = "v4;profile=cass;schema=frankensearch-cass-semantic-v1;fields=agent:keyword+stored,workspace:keyword+stored,workspace_original:stored,source_path:stored,msg_idx:u64+indexed+stored,created_at:i64+indexed+fast+stored,title:text+positions+stored,content:text+positions,title_prefix:text+basic,content_prefix:text+basic,preview:stored,source_id:keyword+stored,origin_kind:keyword+stored,origin_host:keyword+stored,conversation_id:i64+stored;derived_title_prefix=edge_ngrams(full_title);derived_content_prefix=edge_ngrams(utf8_boundary_prefix_bytes<=4096);derived_preview=first_400_unicode_scalars_plus_ellipsis_if_truncated;document_identity=source_id#msg_idx;query_parser=cass-or-binds-tighter-than-and;blank_query=match_all;bare_terms=exact_raw_or_bounded_edge-prefix;phrases=title_or_content_positions;cjk_phrases=compound-bigram-and;negation=all-plus-must-not;wildcards=exact-prefix-on-four-search-fields,suffix-substring-complex-regex-on-title-content;filters=agents-or,workspaces-or,created_at-inclusive,local=origin_kind:local,remote=origin_kind:ssh,source_id;bm25=tantivy-0.26.1-default-no-field-boosts;pagination=offset_then_limit;counts=exact;snippets=disabled";
/// Default schema/query/ranking protocol implemented by the shipping Tantivy adapter.
pub const DEFAULT_SCHEMA_CONTRACT_HASH: &str =
    "9fed22a53e5060243e9528fbbf40605a0df8ea120b3d74ac41ecbb097c2df571";
const MISMATCH_SIGNATURE_DOMAIN: &[u8] = b"frankensearch/quill/mismatch-signature/v1\0";
const LEXICAL_MISMATCH_SIGNATURE_DOMAIN: &[u8] =
    b"frankensearch/quill/lexical-mismatch-signature/v1\0";
const LEXICAL_QUERY_CONTRACT_DOMAIN: &[u8] = b"frankensearch/quill/lexical-query-contract/v1\0";
const LEXICAL_INDEX_IDENTITY_DOMAIN: &[u8] = b"frankensearch/quill/lexical-index-identity/v1\0";
const CAMPAIGN_REPORT_HASH_DOMAIN: &[u8] = b"frankensearch/quill/campaign-report/v5\0";
const DIVERGENCE_REGISTRY_HASH_DOMAIN: &[u8] = b"frankensearch/quill/divergence-registry/v1\0";
const DIVERGENCE_REGISTER_LEDGER_HASH_DOMAIN: &[u8] =
    b"frankensearch/quill/divergence-register-ledger/v1\0";
const MAX_DIVERGENCE_REGISTRY_ENTRIES: usize = 1_024;
const MAX_DIVERGENCE_REGISTER_EVENTS: usize = 4_096;
const MAX_DIVERGENCE_REGISTER_PROSE_BYTES: usize = 64 * 1024;
const MAX_DIVERGENCE_REVIEWER_BYTES: usize = 1_024;
const MAX_DIVERGENCE_REGISTRY_TEXT_BYTES: usize = 16 * 1024 * 1024;
const MAX_DIVERGENCE_REGISTER_MARKER_BYTES: usize = 256;
const MAX_CAMPAIGN_REASON_BYTES: usize = 4 * 1024;
const MAX_CAMPAIGN_POINTER_BYTES: usize = 1024 * 1024;
const MAX_MISMATCH_GROUPS: usize = MAX_QUERY_CASES;
const MAX_MISMATCH_TEXT_BYTES: usize = 64 * 1024 * 1024;

/// Shared analyzer and schema profile that both adapters must acknowledge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticContract {
    pub analyzer_contract_hash: String,
    pub schema_contract_hash: String,
}

impl SemanticContract {
    /// Canonical contract declared by the shipping default Tantivy adapter.
    #[must_use]
    pub fn shipping_default() -> Self {
        Self {
            analyzer_contract_hash: sha256_text(DEFAULT_ANALYZER_CONTRACT_PREIMAGE),
            schema_contract_hash: sha256_text(DEFAULT_SCHEMA_CONTRACT_PREIMAGE),
        }
    }

    /// Bounded semantic profile implemented by the scalar G1a subject.
    #[must_use]
    pub fn scalar_g1a() -> Self {
        Self {
            analyzer_contract_hash: sha256_text(DEFAULT_ANALYZER_CONTRACT_PREIMAGE),
            schema_contract_hash: sha256_text(SCALAR_G1A_SCHEMA_CONTRACT_PREIMAGE),
        }
    }

    /// Native Quill versus Tantivy CASS query/schema profile.
    #[must_use]
    pub fn cass() -> Self {
        Self {
            analyzer_contract_hash: sha256_text(CASS_ANALYZER_CONTRACT_PREIMAGE),
            schema_contract_hash: sha256_text(CASS_SCHEMA_CONTRACT_PREIMAGE),
        }
    }

    /// Construct a semantic profile from two lowercase SHA-256 identities.
    ///
    /// # Errors
    ///
    /// Returns an error unless both values are canonical lowercase SHA-256.
    pub fn new(
        analyzer_contract_hash: impl Into<String>,
        schema_contract_hash: impl Into<String>,
    ) -> Result<Self, GauntletError> {
        let contract = Self {
            analyzer_contract_hash: analyzer_contract_hash.into(),
            schema_contract_hash: schema_contract_hash.into(),
        };
        contract.validate()?;
        Ok(contract)
    }

    pub(crate) fn validate(&self) -> Result<(), GauntletError> {
        if !is_lower_sha256(&self.analyzer_contract_hash)
            || !is_lower_sha256(&self.schema_contract_hash)
        {
            return Err(campaign_error(
                "semantic contract hashes must be lowercase SHA-256",
            ));
        }
        Ok(())
    }
}

/// Adapter receipt proving what was indexed and under which semantic profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineIndexReceipt {
    pub corpus_manifest_hash: String,
    pub document_count: u64,
    pub total_content_bytes: u64,
    pub semantic_contract: SemanticContract,
}

impl EngineIndexReceipt {
    /// Construct the exact receipt expected by the campaign runner.
    ///
    /// # Errors
    ///
    /// Returns an error if the manifest cannot be content-addressed.
    pub fn for_manifest(
        manifest: &CorpusManifest,
        semantic_contract: SemanticContract,
    ) -> Result<Self, GauntletError> {
        Ok(Self {
            corpus_manifest_hash: manifest.manifest_hash()?,
            document_count: manifest.document_count,
            total_content_bytes: manifest.total_content_bytes,
            semantic_contract,
        })
    }
}

/// Boxed future returned by object-safe campaign adapters.
pub type CampaignFuture<'a, T> =
    Pin<Box<dyn Future<Output = Result<T, GauntletError>> + Send + 'a>>;

/// Replayable generated-corpus source used by the bounded indexing loop.
///
/// A campaign consumes one replay while validating the manifest and a second
/// replay while sending identical bounded batches to both engines. Synthetic
/// corpora therefore remain streaming at the xlarge scale.
pub trait GeneratedCorpusReplay: Send + Sync {
    fn replay(&self) -> Box<dyn Iterator<Item = GeneratedDocument> + Send + '_>;
}

impl GeneratedCorpusReplay for Vec<GeneratedDocument> {
    fn replay(&self) -> Box<dyn Iterator<Item = GeneratedDocument> + Send + '_> {
        Box::new(self.iter().cloned())
    }
}

impl GeneratedCorpusReplay for SyntheticCorpus {
    fn replay(&self) -> Box<dyn Iterator<Item = GeneratedDocument> + Send + '_> {
        Box::new(self.iter())
    }
}

struct BorrowedCorpus<'a>(&'a [GeneratedDocument]);

impl GeneratedCorpusReplay for BorrowedCorpus<'_> {
    fn replay(&self) -> Box<dyn Iterator<Item = GeneratedDocument> + Send + '_> {
        Box::new(self.0.iter().cloned())
    }
}

/// Full ingest/query boundary required by the E6 differential campaign.
pub trait DifferentialCampaignEngine: Send + Sync {
    fn descriptor(&self) -> EngineDescriptor;
    /// Adapter-owned semantic identity; never copied from the runner request.
    fn semantic_contract(&self) -> SemanticContract;

    /// Ordinary backend-neutral lexical facade for total result-contract proof.
    ///
    /// CASS adapters intentionally retain the default `None`: their richer
    /// retrieval, hydration, post-filter, and CLI projection contract is not
    /// representable by the ordinary [`LexicalSearch`] boundary.
    ///
    /// # Errors
    ///
    /// Implementations may return a typed adapter error when their ordinary
    /// lexical facade cannot be exposed safely for the current lifecycle.
    fn core_lexical_search(&self) -> Result<Option<&dyn LexicalSearch>, GauntletError> {
        Ok(None)
    }

    fn begin_corpus<'a>(
        &'a mut self,
        cx: &'a Cx,
        manifest: &'a CorpusManifest,
        semantic_contract: &'a SemanticContract,
    ) -> CampaignFuture<'a, ()>;

    fn index_batch<'a>(
        &'a mut self,
        cx: &'a Cx,
        documents: &'a [GeneratedDocument],
    ) -> CampaignFuture<'a, ()>;

    fn commit_corpus<'a>(
        &'a mut self,
        cx: &'a Cx,
        manifest: &'a CorpusManifest,
        semantic_contract: &'a SemanticContract,
    ) -> CampaignFuture<'a, EngineIndexReceipt>;

    fn observe_generated<'a>(
        &'a mut self,
        cx: &'a Cx,
        query: &'a GeneratedQueryCase,
        evidence_case: &'a DifferentialCase,
    ) -> CampaignFuture<'a, EngineObservation>;

    /// Abort a partially initialized/indexed campaign.
    ///
    /// The runner invokes this synchronously on error or cancellation before
    /// successful receipt validation. Adapters must release transient state;
    /// callers must discard an adapter whose backend cannot roll back commits.
    fn abort_corpus(&mut self);
}

struct IndexSession<'a> {
    subject: &'a mut dyn DifferentialCampaignEngine,
    oracle: &'a mut dyn DifferentialCampaignEngine,
    armed: bool,
    subject_begin_attempted: bool,
    oracle_begin_attempted: bool,
}

impl IndexSession<'_> {
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for IndexSession<'_> {
    fn drop(&mut self) {
        if self.armed {
            if self.subject_begin_attempted {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.subject.abort_corpus();
                }));
            }
            if self.oracle_begin_attempted {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.oracle.abort_corpus();
                }));
            }
        }
    }
}

/// Query subset executed from a fully verified generated suite.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CampaignSelection {
    /// Execute every query in manifest order.
    #[default]
    All,
    /// Execute the bounded default-parser classes owned by the scalar G1a gate.
    DefaultSyntax,
    /// Execute the complete native CASS Boolean/glob/filter/range profile.
    CassSyntax,
    /// Execute the named cases, retaining their original manifest order.
    CaseIds { ids: Vec<String> },
}

impl CampaignSelection {
    fn select<'a>(
        &self,
        cases: &'a [GeneratedQueryCase],
    ) -> Result<Vec<&'a GeneratedQueryCase>, GauntletError> {
        let selected: Vec<&'a GeneratedQueryCase> = match self {
            Self::All => cases.iter().collect(),
            Self::DefaultSyntax => cases
                .iter()
                .filter(|case| {
                    case.syntax == QuerySyntax::Default
                        && matches!(
                            &case.query_kind,
                            GeneratedQueryKind::Term
                                | GeneratedQueryKind::MultiTerm
                                | GeneratedQueryKind::Phrase
                                | GeneratedQueryKind::Boolean
                                | GeneratedQueryKind::Paginated
                                | GeneratedQueryKind::Counted
                                | GeneratedQueryKind::Harvested { .. }
                        )
                        && case.filters.is_empty()
                })
                .collect(),
            Self::CassSyntax => cases
                .iter()
                .filter(|case| {
                    case.syntax == QuerySyntax::Cass
                        && matches!(
                            &case.query_kind,
                            GeneratedQueryKind::Boolean
                                | GeneratedQueryKind::Glob { .. }
                                | GeneratedQueryKind::Range { .. }
                                | GeneratedQueryKind::StructuredFilter { .. }
                        )
                })
                .collect(),
            Self::CaseIds { ids } => {
                if ids.len() > MAX_QUERY_CASES || ids.iter().any(|id| !is_canonical_query_id(id)) {
                    return Err(campaign_error(
                        "case selection exceeds the bounded query-ID contract",
                    ));
                }
                let requested = ids.iter().map(String::as_str).collect::<BTreeSet<_>>();
                let available = cases
                    .iter()
                    .map(|case| case.id.as_str())
                    .collect::<BTreeSet<_>>();
                if requested.len() != ids.len() || !requested.is_subset(&available) {
                    return Err(campaign_error(
                        "case selection contains a duplicate or unknown query ID",
                    ));
                }
                cases
                    .iter()
                    .filter(|case| requested.contains(case.id.as_str()))
                    .collect()
            }
        };
        if selected.is_empty() {
            return Err(campaign_error(
                "campaign selection must execute at least one query",
            ));
        }
        Ok(selected)
    }
}

/// One reviewed per-fixture divergence allowlist row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DivergenceRegisterEntry {
    pub id: String,
    pub class: DivergenceClass,
    pub fixture_id: String,
    /// Sorted normalized mismatch signatures accepted by this reviewed row.
    pub mismatch_signatures: Vec<String>,
    pub decision: DivergenceRegisterDecision,
    pub root_cause: String,
    pub consumer_impact: String,
    pub reviewer: String,
    /// Review date in canonical `YYYY-MM-DD` form.
    pub reviewed_at: String,
}

/// Register decision copied from the human-reviewed divergence ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DivergenceRegisterDecision {
    Accept,
    Fix,
    Pending,
}

impl DivergenceRegisterEntry {
    pub(crate) fn validate(&self) -> Result<(), GauntletError> {
        let invalid_class = !matches!(
            self.class,
            DivergenceClass::SnippetWindow
                | DivergenceClass::GlobExpansionLimit
                | DivergenceClass::QueryCanonicalization
                | DivergenceClass::OracleBug
                | DivergenceClass::StatsSemantics
                | DivergenceClass::UnicodeEdge
                | DivergenceClass::OversizedQueryToken
        );
        let invalid_signatures = self.mismatch_signatures.is_empty()
            || self.mismatch_signatures.len() > 64
            || self
                .mismatch_signatures
                .iter()
                .any(|signature| !is_lower_sha256(signature))
            || self
                .mismatch_signatures
                .windows(2)
                .any(|pair| pair[0] > pair[1]);
        if !is_register_id(&self.id)
            || !is_bounded_register_text(&self.fixture_id, MAX_QUERY_ID_BYTES)
            || self.decision != DivergenceRegisterDecision::Accept
            || !is_bounded_register_text(&self.root_cause, MAX_DIVERGENCE_REGISTER_PROSE_BYTES)
            || !is_bounded_register_text(&self.consumer_impact, MAX_DIVERGENCE_REGISTER_PROSE_BYTES)
            || !is_bounded_register_text(&self.reviewer, MAX_DIVERGENCE_REVIEWER_BYTES)
            || !is_review_date(&self.reviewed_at)
            || invalid_class
            || invalid_signatures
        {
            return Err(campaign_error(
                "divergence register entries require an accepted classified row with root cause, consumer impact, fixture, reviewer, and review date",
            ));
        }
        Ok(())
    }

    pub(crate) fn matches_comparison(
        &self,
        query: &GeneratedQueryCase,
        comparison: &ComparisonReport,
    ) -> bool {
        let mut observed = comparison
            .divergences
            .iter()
            .filter(|divergence| !is_auto_class(divergence.class))
            .map(|divergence| mismatch_signature(comparison.rank_class, divergence))
            .collect::<Vec<_>>();
        observed.sort();
        self.fixture_id == query.id
            && query.expected_divergence.as_deref() == Some(self.id.as_str())
            && !observed.is_empty()
            && comparison
                .divergences
                .iter()
                .filter(|divergence| !is_auto_class(divergence.class))
                .all(|divergence| divergence.class == self.class)
            && observed == self.mismatch_signatures
    }
}

/// Validated machine-facing subset of the Markdown Divergence Register.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DivergenceRegistry {
    entries: Vec<DivergenceRegisterEntry>,
}

impl DivergenceRegistry {
    /// Validate, sort, and retain reviewed register entries.
    ///
    /// `TieOrder` and `ScoreEpsilon` are bounded accept-by-class policies and
    /// therefore do not belong in this per-fixture registry.
    ///
    /// # Errors
    ///
    /// Returns an error for duplicate/malformed IDs, incomplete review
    /// evidence, or an attempt to bless a raw comparator failure.
    pub fn new(mut entries: Vec<DivergenceRegisterEntry>) -> Result<Self, GauntletError> {
        validate_registry_bounds(&entries)?;
        entries.sort_by(|left, right| left.id.cmp(&right.id));
        let registry = Self { entries };
        registry.validate()?;
        Ok(registry)
    }

    fn validate(&self) -> Result<(), GauntletError> {
        validate_registry_bounds(&self.entries)?;
        for (index, entry) in self.entries.iter().enumerate() {
            let out_of_order_or_duplicate = index > 0 && self.entries[index - 1].id >= entry.id;
            if out_of_order_or_duplicate {
                return Err(campaign_error(
                    "divergence registry entries require unique sorted DIV-NNN IDs",
                ));
            }
            entry.validate()?;
        }
        Ok(())
    }

    fn find(&self, id: &str) -> Option<&DivergenceRegisterEntry> {
        self.entries
            .binary_search_by(|entry| entry.id.as_str().cmp(id))
            .ok()
            .map(|index| &self.entries[index])
    }

    /// Domain-separated identity of the complete reviewed policy input.
    ///
    /// # Errors
    ///
    /// Returns an error if registry validation or serialization fails.
    pub fn registry_hash(&self) -> Result<String, GauntletError> {
        self.validate()?;
        let mut hasher = Sha256::new();
        hasher.update(DIVERGENCE_REGISTRY_HASH_DOMAIN);
        hasher.update(serde_json::to_vec(self)?);
        Ok(lower_hex(&hasher.finalize()))
    }
}

fn is_bounded_register_text(value: &str, max_bytes: usize) -> bool {
    !value.is_empty()
        && value.len() <= max_bytes
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

fn validate_registry_bounds(entries: &[DivergenceRegisterEntry]) -> Result<(), GauntletError> {
    let aggregate_bytes = entries
        .iter()
        .flat_map(|entry| {
            [
                entry.id.len(),
                entry.fixture_id.len(),
                entry.root_cause.len(),
                entry.consumer_impact.len(),
                entry.reviewer.len(),
                entry.reviewed_at.len(),
            ]
            .into_iter()
            .chain(entry.mismatch_signatures.iter().map(String::len))
        })
        .try_fold(0_usize, usize::checked_add);
    if entries.len() > MAX_DIVERGENCE_REGISTRY_ENTRIES
        || aggregate_bytes.is_none_or(|bytes| bytes > MAX_DIVERGENCE_REGISTRY_TEXT_BYTES)
    {
        return Err(campaign_error(
            "divergence registry exceeds its entry or aggregate text budget",
        ));
    }
    Ok(())
}

/// Append-only, machine-readable source of truth for divergence review.
///
/// The event stream retains every observation, disposition, correction, and
/// predicted-class decision. Corrections append a new event whose
/// `supersedes` field names the currently active event for the same logical
/// record; prior evidence is never rewritten or removed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DivergenceRegisterLedger {
    pub schema_version: u32,
    pub register_id: String,
    pub redaction_policy_version: String,
    pub events: Vec<DivergenceRegisterEvent>,
}

/// One immutable event in the Divergence Register ledger.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "event_type",
    content = "event",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum DivergenceRegisterEvent {
    Observation(Box<DivergenceObservationEvent>),
    Disposition(DivergenceDispositionEvent),
    Prediction(DivergencePredictionEvent),
}

/// Common append-only ordering and authorship fields for register events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DivergenceRegisterEventHeader {
    pub sequence: u64,
    pub supersedes: Option<u64>,
    pub recorded_by: String,
    pub recorded_at: String,
}

/// Exact revisions needed to reproduce one observed divergence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DivergenceRevisionSet {
    pub subject_git_revision: String,
    pub oracle_git_revision: String,
    pub corpus_manifest_sha256: String,
    pub query_manifest_sha256: String,
    pub generator_revision: String,
}

/// Minimized, replayable fixture evidence retained with an observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DivergenceFixtureEvidence {
    pub fixture_id: String,
    pub fixture_sha256: String,
    pub regression_test: String,
    pub minimized: bool,
}

/// Source-sensitive diagnostics represented only by a digest and safe marker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RedactedDivergenceDiagnostic {
    pub payload_sha256: String,
    pub marker: String,
}

/// One observed mismatch, including immutable first-seen evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DivergenceObservationEvent {
    pub header: DivergenceRegisterEventHeader,
    pub divergence_id: String,
    pub class: DivergenceClass,
    pub first_seen_artifact_object_hash: String,
    pub first_seen_artifact_sha256: String,
    pub revisions: DivergenceRevisionSet,
    pub fixture: DivergenceFixtureEvidence,
    pub mismatch_signatures: Vec<String>,
    pub observed_behavior: String,
    pub expected_behavior: String,
    pub root_cause: String,
    pub consumer_impact: String,
    pub diagnostic: RedactedDivergenceDiagnostic,
}

/// Current reviewed outcome for an observed divergence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DivergenceDisposition {
    Accepted {
        equivalence_law: String,
        rationale: String,
        reviewer: String,
        reviewed_at: String,
    },
    Fixed {
        fixing_commit: String,
        regression_test: String,
        reviewer: String,
        reviewed_at: String,
    },
    Blocking {
        bead_id: String,
        rationale: String,
        reviewer: String,
        reviewed_at: String,
    },
}

/// Append-only disposition revision for one observed divergence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DivergenceDispositionEvent {
    pub header: DivergenceRegisterEventHeader,
    pub divergence_id: String,
    pub disposition: DivergenceDisposition,
}

/// Lifecycle state for a divergence class predicted before live evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PredictedDivergenceState {
    Declared {
        rationale: String,
        owner: String,
        bead_id: String,
    },
    Observed {
        divergence_id: String,
        reviewer: String,
        reviewed_at: String,
    },
    Retired {
        proof_sha256: String,
        rationale: String,
        reviewer: String,
        reviewed_at: String,
    },
}

/// Append-only lifecycle revision for one predicted divergence class.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DivergencePredictionEvent {
    pub header: DivergenceRegisterEventHeader,
    pub prediction_id: String,
    pub class: DivergenceClass,
    pub state: PredictedDivergenceState,
}

/// Per-class terminal-census counts derived from the authoritative ledger.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DivergenceClassCensus {
    pub class: DivergenceClass,
    pub active_entries: u64,
    pub observed_mismatches: u64,
    pub accepted_entries: u64,
    pub fixed_entries: u64,
    pub blocking_entries: u64,
}

/// Deterministic join of campaign mismatch signatures against the register.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DivergenceCensus {
    pub schema_version: u32,
    pub register_hash: String,
    pub mismatch_count: u64,
    pub registered_mismatch_count: u64,
    pub class_census: Vec<DivergenceClassCensus>,
    pub unclassified_signatures: Vec<String>,
    pub fixed_regression_divergence_ids: Vec<String>,
    pub blocking_divergence_ids: Vec<String>,
    pub unresolved_prediction_ids: Vec<String>,
    pub flip_ready: bool,
}

struct DivergenceRegisterProjection<'a> {
    observations: BTreeMap<String, &'a DivergenceObservationEvent>,
    dispositions: BTreeMap<String, &'a DivergenceDispositionEvent>,
    predictions: BTreeMap<String, &'a DivergencePredictionEvent>,
}

impl DivergenceRegisterEvent {
    const fn header(&self) -> &DivergenceRegisterEventHeader {
        match self {
            Self::Observation(event) => &event.header,
            Self::Disposition(event) => &event.header,
            Self::Prediction(event) => &event.header,
        }
    }
}

impl DivergenceRegisterEventHeader {
    fn validate(&self, expected_sequence: u64) -> Result<(), GauntletError> {
        if self.sequence != expected_sequence
            || self
                .supersedes
                .is_some_and(|sequence| sequence == 0 || sequence >= self.sequence)
            || !is_bounded_register_text(&self.recorded_by, MAX_DIVERGENCE_REVIEWER_BYTES)
            || !is_utc_timestamp(&self.recorded_at)
        {
            return Err(campaign_error(
                "divergence register event header is not canonical or sequential",
            ));
        }
        Ok(())
    }
}

impl DivergenceRevisionSet {
    fn validate(&self) -> Result<(), GauntletError> {
        if !is_git_revision(&self.subject_git_revision)
            || !is_git_revision(&self.oracle_git_revision)
            || !is_lower_sha256(&self.corpus_manifest_sha256)
            || !is_lower_sha256(&self.query_manifest_sha256)
            || !is_bounded_register_text(&self.generator_revision, MAX_QUERY_ID_BYTES)
        {
            return Err(campaign_error(
                "divergence observation revisions are incomplete or noncanonical",
            ));
        }
        Ok(())
    }
}

impl DivergenceFixtureEvidence {
    fn validate(&self) -> Result<(), GauntletError> {
        if !is_bounded_register_text(&self.fixture_id, MAX_QUERY_ID_BYTES)
            || !is_lower_sha256(&self.fixture_sha256)
            || !is_bounded_register_text(&self.regression_test, MAX_DIVERGENCE_REGISTER_PROSE_BYTES)
            || !self.minimized
        {
            return Err(campaign_error(
                "divergence evidence requires a minimized hashed fixture and replay test",
            ));
        }
        Ok(())
    }
}

impl RedactedDivergenceDiagnostic {
    fn validate(&self) -> Result<(), GauntletError> {
        let safe_marker = self
            .marker
            .strip_prefix("<redacted:")
            .and_then(|value| value.strip_suffix('>'))
            .is_some_and(|value| {
                !value.is_empty()
                    && value.bytes().all(|byte| {
                        byte.is_ascii_lowercase()
                            || byte.is_ascii_digit()
                            || matches!(byte, b'-' | b'_')
                    })
            });
        if !is_lower_sha256(&self.payload_sha256)
            || self.marker.len() > MAX_DIVERGENCE_REGISTER_MARKER_BYTES
            || !safe_marker
        {
            return Err(campaign_error(
                "divergence diagnostic must contain only a digest and canonical redaction marker",
            ));
        }
        Ok(())
    }
}

impl DivergenceObservationEvent {
    fn validate(&self) -> Result<(), GauntletError> {
        self.revisions.validate()?;
        self.fixture.validate()?;
        self.diagnostic.validate()?;
        let invalid_signatures = self.mismatch_signatures.is_empty()
            || self.mismatch_signatures.len() > 64
            || self
                .mismatch_signatures
                .iter()
                .any(|signature| !is_lower_sha256(signature))
            || self
                .mismatch_signatures
                .windows(2)
                .any(|pair| pair[0] >= pair[1]);
        if !is_register_id(&self.divergence_id)
            || !is_lower_xxh3(&self.first_seen_artifact_object_hash)
            || !is_lower_sha256(&self.first_seen_artifact_sha256)
            || invalid_signatures
            || !is_bounded_register_text(
                &self.observed_behavior,
                MAX_DIVERGENCE_REGISTER_PROSE_BYTES,
            )
            || !is_bounded_register_text(
                &self.expected_behavior,
                MAX_DIVERGENCE_REGISTER_PROSE_BYTES,
            )
            || !is_bounded_register_text(&self.root_cause, MAX_DIVERGENCE_REGISTER_PROSE_BYTES)
            || !is_bounded_register_text(&self.consumer_impact, MAX_DIVERGENCE_REGISTER_PROSE_BYTES)
        {
            return Err(campaign_error(
                "divergence observation is missing bounded classified evidence",
            ));
        }
        Ok(())
    }
}

impl DivergenceDisposition {
    fn validate(&self, observation: &DivergenceObservationEvent) -> Result<(), GauntletError> {
        match self {
            Self::Accepted {
                equivalence_law,
                rationale,
                reviewer,
                reviewed_at,
            } => {
                let raw_failure_class = matches!(
                    observation.class,
                    DivergenceClass::RankMismatch
                        | DivergenceClass::SnippetMismatch
                        | DivergenceClass::CountMismatch
                        | DivergenceClass::DocumentCountMismatch
                        | DivergenceClass::PostingRecordSemantics
                );
                if raw_failure_class
                    || !is_bounded_register_text(
                        equivalence_law,
                        MAX_DIVERGENCE_REGISTER_PROSE_BYTES,
                    )
                    || !is_bounded_register_text(rationale, MAX_DIVERGENCE_REGISTER_PROSE_BYTES)
                    || !is_bounded_register_text(reviewer, MAX_DIVERGENCE_REVIEWER_BYTES)
                    || reviewer == &observation.header.recorded_by
                    || !is_utc_timestamp(reviewed_at)
                {
                    return Err(campaign_error(
                        "accepted divergence requires a semantic class, law, rationale, and independent review",
                    ));
                }
            }
            Self::Fixed {
                fixing_commit,
                regression_test,
                reviewer,
                reviewed_at,
            } => {
                if !is_git_revision(fixing_commit)
                    || !is_bounded_register_text(
                        regression_test,
                        MAX_DIVERGENCE_REGISTER_PROSE_BYTES,
                    )
                    || !is_bounded_register_text(reviewer, MAX_DIVERGENCE_REVIEWER_BYTES)
                    || !is_utc_timestamp(reviewed_at)
                {
                    return Err(campaign_error(
                        "fixed divergence requires a commit, regression test, and review",
                    ));
                }
            }
            Self::Blocking {
                bead_id,
                rationale,
                reviewer,
                reviewed_at,
            } => {
                if !is_bead_id(bead_id)
                    || !is_bounded_register_text(rationale, MAX_DIVERGENCE_REGISTER_PROSE_BYTES)
                    || !is_bounded_register_text(reviewer, MAX_DIVERGENCE_REVIEWER_BYTES)
                    || !is_utc_timestamp(reviewed_at)
                {
                    return Err(campaign_error(
                        "blocking divergence requires an owned bead, rationale, and review",
                    ));
                }
            }
        }
        Ok(())
    }
}

impl DivergencePredictionEvent {
    fn validate(
        &self,
        observations: &BTreeMap<String, &DivergenceObservationEvent>,
        previous: Option<&Self>,
    ) -> Result<(), GauntletError> {
        if !is_prediction_id(&self.prediction_id) {
            return Err(campaign_error(
                "predicted divergence ID must use canonical PRED-NNN form",
            ));
        }
        if previous.is_some_and(|event| {
            matches!(
                (&event.state, &self.state),
                (_, PredictedDivergenceState::Declared { .. })
            )
        }) {
            return Err(campaign_error(
                "predicted divergence lifecycle cannot return to declared",
            ));
        }
        match &self.state {
            PredictedDivergenceState::Declared {
                rationale,
                owner,
                bead_id,
            } => {
                if previous.is_some()
                    || !is_bounded_register_text(rationale, MAX_DIVERGENCE_REGISTER_PROSE_BYTES)
                    || !is_bounded_register_text(owner, MAX_DIVERGENCE_REVIEWER_BYTES)
                    || !is_bead_id(bead_id)
                {
                    return Err(campaign_error(
                        "declared prediction requires a rationale, owner, and bead",
                    ));
                }
            }
            PredictedDivergenceState::Observed {
                divergence_id,
                reviewer,
                reviewed_at,
            } => {
                let matching_observation = observations
                    .get(divergence_id)
                    .is_some_and(|observation| observation.class == self.class);
                if previous.is_none()
                    || !matching_observation
                    || !is_bounded_register_text(reviewer, MAX_DIVERGENCE_REVIEWER_BYTES)
                    || !is_utc_timestamp(reviewed_at)
                {
                    return Err(campaign_error(
                        "observed prediction must name a same-class registered divergence and review",
                    ));
                }
            }
            PredictedDivergenceState::Retired {
                proof_sha256,
                rationale,
                reviewer,
                reviewed_at,
            } => {
                if previous.is_none()
                    || !is_lower_sha256(proof_sha256)
                    || !is_bounded_register_text(rationale, MAX_DIVERGENCE_REGISTER_PROSE_BYTES)
                    || !is_bounded_register_text(reviewer, MAX_DIVERGENCE_REVIEWER_BYTES)
                    || !is_utc_timestamp(reviewed_at)
                {
                    return Err(campaign_error(
                        "retired prediction requires hashed proof, rationale, and review",
                    ));
                }
            }
        }
        Ok(())
    }
}

impl DivergenceRegisterLedger {
    /// Construct and validate an append-only register ledger.
    ///
    /// # Errors
    ///
    /// Returns an error when any event, revision link, evidence field, or
    /// active projection violates the v1 contract.
    pub fn new(
        register_id: impl Into<String>,
        events: Vec<DivergenceRegisterEvent>,
    ) -> Result<Self, GauntletError> {
        let ledger = Self {
            schema_version: DIVERGENCE_REGISTER_LEDGER_SCHEMA_VERSION,
            register_id: register_id.into(),
            redaction_policy_version: DIVERGENCE_REGISTER_REDACTION_POLICY_VERSION.to_owned(),
            events,
        };
        ledger.validate()?;
        Ok(ledger)
    }

    /// Validate schema, ordering, supersession, evidence, and active state.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed evidence, history rewrites, orphan
    /// dispositions, duplicate active signatures, or incomplete reviews.
    pub fn validate(&self) -> Result<(), GauntletError> {
        if self.schema_version != DIVERGENCE_REGISTER_LEDGER_SCHEMA_VERSION
            || !is_register_name(&self.register_id)
            || self.redaction_policy_version != DIVERGENCE_REGISTER_REDACTION_POLICY_VERSION
            || self.events.len() > MAX_DIVERGENCE_REGISTER_EVENTS
        {
            return Err(campaign_error(
                "divergence register schema, identity, policy, or event budget is invalid",
            ));
        }
        let serialized = serde_json::to_vec(self).map_err(|error| {
            campaign_error(format!("failed to serialize divergence register: {error}"))
        })?;
        if serialized.len() > MAX_DIVERGENCE_REGISTRY_TEXT_BYTES {
            return Err(campaign_error(
                "divergence register exceeds its aggregate byte budget",
            ));
        }

        let mut observations = BTreeMap::<String, &DivergenceObservationEvent>::new();
        let mut dispositions = BTreeMap::<String, &DivergenceDispositionEvent>::new();
        let mut predictions = BTreeMap::<String, &DivergencePredictionEvent>::new();
        for (index, event) in self.events.iter().enumerate() {
            let expected_sequence = u64::try_from(index)
                .ok()
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| campaign_error("divergence register sequence overflow"))?;
            event.header().validate(expected_sequence)?;
            match event {
                DivergenceRegisterEvent::Observation(observation) => {
                    observation.validate()?;
                    if observations
                        .get(&observation.divergence_id)
                        .is_some_and(|previous| previous.class != observation.class)
                    {
                        return Err(campaign_error(
                            "divergence observation correction cannot change its class",
                        ));
                    }
                    validate_revision_link(
                        &observation.header,
                        observations
                            .get(&observation.divergence_id)
                            .map(|event| event.header.sequence),
                        "observation",
                    )?;
                    observations.insert(observation.divergence_id.clone(), observation);
                }
                DivergenceRegisterEvent::Disposition(disposition) => {
                    if !is_register_id(&disposition.divergence_id) {
                        return Err(campaign_error(
                            "divergence disposition has a malformed register ID",
                        ));
                    }
                    let observation =
                        observations
                            .get(&disposition.divergence_id)
                            .ok_or_else(|| {
                                campaign_error(
                                    "divergence disposition precedes its observed evidence",
                                )
                            })?;
                    validate_revision_link(
                        &disposition.header,
                        dispositions
                            .get(&disposition.divergence_id)
                            .map(|event| event.header.sequence),
                        "disposition",
                    )?;
                    disposition.disposition.validate(observation)?;
                    dispositions.insert(disposition.divergence_id.clone(), disposition);
                }
                DivergenceRegisterEvent::Prediction(prediction) => {
                    validate_revision_link(
                        &prediction.header,
                        predictions
                            .get(&prediction.prediction_id)
                            .map(|event| event.header.sequence),
                        "prediction",
                    )?;
                    prediction.validate(
                        &observations,
                        predictions.get(&prediction.prediction_id).copied(),
                    )?;
                    predictions.insert(prediction.prediction_id.clone(), prediction);
                }
            }
        }

        let mut active_signatures = BTreeMap::<&str, &str>::new();
        for (divergence_id, observation) in &observations {
            let disposition = dispositions.get(divergence_id).ok_or_else(|| {
                campaign_error("every active divergence observation requires a disposition")
            })?;
            if disposition.header.sequence <= observation.header.sequence {
                return Err(campaign_error(
                    "corrected divergence evidence requires a later disposition review",
                ));
            }
            disposition.disposition.validate(observation)?;
            for signature in &observation.mismatch_signatures {
                if active_signatures.insert(signature, divergence_id).is_some() {
                    return Err(campaign_error(
                        "one active mismatch signature cannot belong to multiple divergences",
                    ));
                }
            }
        }
        for prediction in predictions.values() {
            if let PredictedDivergenceState::Observed { divergence_id, .. } = &prediction.state
                && !observations
                    .get(divergence_id)
                    .is_some_and(|observation| observation.class == prediction.class)
            {
                return Err(campaign_error(
                    "active observed prediction no longer names a same-class divergence",
                ));
            }
        }
        Ok(())
    }

    /// Domain-separated identity of the complete immutable event stream.
    ///
    /// # Errors
    ///
    /// Returns an error when the ledger is invalid or cannot be serialized.
    pub fn ledger_hash(&self) -> Result<String, GauntletError> {
        self.validate()?;
        let mut hasher = Sha256::new();
        hasher.update(DIVERGENCE_REGISTER_LEDGER_HASH_DOMAIN);
        hasher.update(serde_json::to_vec(self).map_err(|error| {
            campaign_error(format!("failed to serialize divergence register: {error}"))
        })?);
        Ok(lower_hex(&hasher.finalize()))
    }

    /// Prove that this snapshot only appends to a prior valid snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error if schema identity changed or any prior event was
    /// removed, reordered, or edited.
    pub fn validate_append_only_successor(&self, previous: &Self) -> Result<(), GauntletError> {
        previous.validate()?;
        self.validate()?;
        if self.schema_version != previous.schema_version
            || self.register_id != previous.register_id
            || self.redaction_policy_version != previous.redaction_policy_version
            || self.events.len() < previous.events.len()
            || self.events[..previous.events.len()] != previous.events
        {
            return Err(campaign_error(
                "divergence register successor rewrites prior history",
            ));
        }
        Ok(())
    }

    /// Reject caller-supplied sensitive canaries anywhere in committed bytes.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid ledger, malformed canary, or leak.
    pub fn validate_redaction_canaries(
        &self,
        forbidden_canaries: &[&str],
    ) -> Result<(), GauntletError> {
        self.validate()?;
        let serialized = serde_json::to_string(self).map_err(|error| {
            campaign_error(format!("failed to serialize divergence register: {error}"))
        })?;
        for canary in forbidden_canaries {
            if canary.is_empty() || canary.chars().any(char::is_control) {
                return Err(campaign_error(
                    "divergence register redaction canary is empty or malformed",
                ));
            }
            if serialized.contains(canary) {
                return Err(campaign_error(
                    "divergence register contains a forbidden source-sensitive canary",
                ));
            }
        }
        Ok(())
    }

    /// Join a sorted set of emitted mismatch signatures to active entries.
    ///
    /// A returned census is evidence even when `flip_ready` is false. Call
    /// [`Self::require_terminal_census`] when a release gate must fail closed.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid ledger or malformed/duplicate input.
    pub fn census(
        &self,
        mismatch_signatures: &[String],
    ) -> Result<DivergenceCensus, GauntletError> {
        self.validate()?;
        if mismatch_signatures
            .iter()
            .any(|signature| !is_lower_sha256(signature))
            || mismatch_signatures
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(campaign_error(
                "terminal census mismatch signatures must be unique sorted SHA-256 values",
            ));
        }
        let projection = self.projection();
        let mut signature_owners =
            BTreeMap::<&str, (&DivergenceObservationEvent, &DivergenceDispositionEvent)>::new();
        for (divergence_id, observation) in &projection.observations {
            let disposition = projection
                .dispositions
                .get(divergence_id)
                .ok_or_else(|| campaign_error("active divergence is missing a disposition"))?;
            for signature in &observation.mismatch_signatures {
                signature_owners.insert(signature, (observation, disposition));
            }
        }

        let mut unclassified_signatures = Vec::new();
        let mut fixed_regressions = BTreeSet::new();
        for signature in mismatch_signatures {
            match signature_owners.get(signature.as_str()) {
                None => unclassified_signatures.push(signature.clone()),
                Some((observation, disposition)) => {
                    if matches!(
                        &disposition.disposition,
                        DivergenceDisposition::Fixed { .. }
                    ) {
                        fixed_regressions.insert(observation.divergence_id.clone());
                    }
                }
            }
        }
        let blocking_divergence_ids = projection
            .dispositions
            .iter()
            .filter(|(_, event)| {
                matches!(&event.disposition, DivergenceDisposition::Blocking { .. })
            })
            .map(|(divergence_id, _)| divergence_id.clone())
            .collect::<Vec<_>>();
        let unresolved_prediction_ids = projection
            .predictions
            .iter()
            .filter(|(_, event)| matches!(&event.state, PredictedDivergenceState::Declared { .. }))
            .map(|(prediction_id, _)| prediction_id.clone())
            .collect::<Vec<_>>();

        let mut class_census = Vec::with_capacity(DIVERGENCE_CLASSES.len());
        for class in DIVERGENCE_CLASSES {
            let active = projection
                .observations
                .values()
                .filter(|observation| observation.class == class)
                .count();
            let observed = mismatch_signatures
                .iter()
                .filter(|signature| {
                    signature_owners
                        .get(signature.as_str())
                        .is_some_and(|(observation, _)| observation.class == class)
                })
                .count();
            let accepted = projection
                .dispositions
                .iter()
                .filter(|(divergence_id, disposition)| {
                    projection
                        .observations
                        .get(*divergence_id)
                        .is_some_and(|observation| observation.class == class)
                        && matches!(
                            &disposition.disposition,
                            DivergenceDisposition::Accepted { .. }
                        )
                })
                .count();
            let fixed = projection
                .dispositions
                .iter()
                .filter(|(divergence_id, disposition)| {
                    projection
                        .observations
                        .get(*divergence_id)
                        .is_some_and(|observation| observation.class == class)
                        && matches!(
                            &disposition.disposition,
                            DivergenceDisposition::Fixed { .. }
                        )
                })
                .count();
            let blocking = projection
                .dispositions
                .iter()
                .filter(|(divergence_id, disposition)| {
                    projection
                        .observations
                        .get(*divergence_id)
                        .is_some_and(|observation| observation.class == class)
                        && matches!(
                            &disposition.disposition,
                            DivergenceDisposition::Blocking { .. }
                        )
                })
                .count();
            class_census.push(DivergenceClassCensus {
                class,
                active_entries: count_to_u64(active)?,
                observed_mismatches: count_to_u64(observed)?,
                accepted_entries: count_to_u64(accepted)?,
                fixed_entries: count_to_u64(fixed)?,
                blocking_entries: count_to_u64(blocking)?,
            });
        }

        let fixed_regression_divergence_ids = fixed_regressions.into_iter().collect::<Vec<_>>();
        let flip_ready = unclassified_signatures.is_empty()
            && fixed_regression_divergence_ids.is_empty()
            && blocking_divergence_ids.is_empty()
            && unresolved_prediction_ids.is_empty();
        let mismatch_count = count_to_u64(mismatch_signatures.len())?;
        let registered_mismatch_count =
            mismatch_count.saturating_sub(count_to_u64(unclassified_signatures.len())?);
        Ok(DivergenceCensus {
            schema_version: DIVERGENCE_REGISTER_LEDGER_SCHEMA_VERSION,
            register_hash: self.ledger_hash()?,
            mismatch_count,
            registered_mismatch_count,
            class_census,
            unclassified_signatures,
            fixed_regression_divergence_ids,
            blocking_divergence_ids,
            unresolved_prediction_ids,
            flip_ready,
        })
    }

    /// Require a one-to-one, nonblocking, no-regression terminal census.
    ///
    /// # Errors
    ///
    /// Returns an error when unclassified mismatches, fixed regressions,
    /// explicit blockers, or unresolved predictions remain.
    pub fn require_terminal_census(
        &self,
        mismatch_signatures: &[String],
    ) -> Result<DivergenceCensus, GauntletError> {
        let census = self.census(mismatch_signatures)?;
        if !census.flip_ready {
            return Err(campaign_error(format!(
                "divergence terminal census is not flip-ready: {} unclassified, {} fixed regressions, {} blockers, {} unresolved predictions",
                census.unclassified_signatures.len(),
                census.fixed_regression_divergence_ids.len(),
                census.blocking_divergence_ids.len(),
                census.unresolved_prediction_ids.len(),
            )));
        }
        Ok(census)
    }

    /// Render a deterministic, redacted active-entry review table.
    ///
    /// # Errors
    ///
    /// Returns an error when the authoritative ledger is invalid.
    pub fn review_table(&self) -> Result<String, GauntletError> {
        use std::fmt::Write as _;

        self.validate()?;
        let projection = self.projection();
        let mut output = String::from(
            "| ID | Class | Fixture | Disposition | Reviewer |\n|---|---|---|---|---|\n",
        );
        for (divergence_id, observation) in &projection.observations {
            let disposition = projection
                .dispositions
                .get(divergence_id)
                .ok_or_else(|| campaign_error("active divergence is missing a disposition"))?;
            let (label, reviewer) = disposition_label_and_reviewer(&disposition.disposition);
            writeln!(
                output,
                "| {divergence_id} | {} | {} | {label} | {reviewer} |",
                divergence_class_name(observation.class),
                observation.fixture.fixture_id,
            )
            .map_err(|error| {
                campaign_error(format!("failed to render divergence review table: {error}"))
            })?;
        }
        Ok(output)
    }

    fn projection(&self) -> DivergenceRegisterProjection<'_> {
        let mut projection = DivergenceRegisterProjection {
            observations: BTreeMap::new(),
            dispositions: BTreeMap::new(),
            predictions: BTreeMap::new(),
        };
        for event in &self.events {
            match event {
                DivergenceRegisterEvent::Observation(observation) => {
                    projection
                        .observations
                        .insert(observation.divergence_id.clone(), observation);
                }
                DivergenceRegisterEvent::Disposition(disposition) => {
                    projection
                        .dispositions
                        .insert(disposition.divergence_id.clone(), disposition);
                }
                DivergenceRegisterEvent::Prediction(prediction) => {
                    projection
                        .predictions
                        .insert(prediction.prediction_id.clone(), prediction);
                }
            }
        }
        projection
    }
}

const DIVERGENCE_CLASSES: [DivergenceClass; 14] = [
    DivergenceClass::TieOrder,
    DivergenceClass::ScoreEpsilon,
    DivergenceClass::RankMismatch,
    DivergenceClass::SnippetMismatch,
    DivergenceClass::SnippetWindow,
    DivergenceClass::CountMismatch,
    DivergenceClass::DocumentCountMismatch,
    DivergenceClass::GlobExpansionLimit,
    DivergenceClass::QueryCanonicalization,
    DivergenceClass::OracleBug,
    DivergenceClass::StatsSemantics,
    DivergenceClass::PostingRecordSemantics,
    DivergenceClass::UnicodeEdge,
    DivergenceClass::OversizedQueryToken,
];

fn validate_revision_link(
    header: &DivergenceRegisterEventHeader,
    active_sequence: Option<u64>,
    event_kind: &str,
) -> Result<(), GauntletError> {
    if header.supersedes != active_sequence {
        return Err(campaign_error(format!(
            "divergence register {event_kind} must supersede its current active revision"
        )));
    }
    Ok(())
}

fn disposition_label_and_reviewer(disposition: &DivergenceDisposition) -> (&'static str, &str) {
    match disposition {
        DivergenceDisposition::Accepted { reviewer, .. } => ("accepted", reviewer),
        DivergenceDisposition::Fixed { reviewer, .. } => ("fixed", reviewer),
        DivergenceDisposition::Blocking { reviewer, .. } => ("blocking", reviewer),
    }
}

const fn divergence_class_name(class: DivergenceClass) -> &'static str {
    match class {
        DivergenceClass::TieOrder => "tie_order",
        DivergenceClass::ScoreEpsilon => "score_epsilon",
        DivergenceClass::RankMismatch => "rank_mismatch",
        DivergenceClass::SnippetMismatch => "snippet_mismatch",
        DivergenceClass::SnippetWindow => "snippet_window",
        DivergenceClass::CountMismatch => "count_mismatch",
        DivergenceClass::DocumentCountMismatch => "document_count_mismatch",
        DivergenceClass::GlobExpansionLimit => "glob_expansion_limit",
        DivergenceClass::QueryCanonicalization => "query_canonicalization",
        DivergenceClass::OracleBug => "oracle_bug",
        DivergenceClass::StatsSemantics => "stats_semantics",
        DivergenceClass::PostingRecordSemantics => "posting_record_semantics",
        DivergenceClass::UnicodeEdge => "unicode_edge",
        DivergenceClass::OversizedQueryToken => "oversized_query_token",
    }
}

fn count_to_u64(value: usize) -> Result<u64, GauntletError> {
    u64::try_from(value).map_err(|_| campaign_error("divergence census count overflow"))
}

/// Public-contract evidence required from each selected campaign case.
///
/// The value is part of [`CampaignConfig`] and therefore covered by both the
/// run reservation and the final report hash. `RankEnvelopeOnly` is retained
/// for legacy fixtures and the dedicated CASS campaign; it is not admissible
/// evidence for a provenance-bearing default-syntax replacement decision.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CampaignContractMode {
    #[default]
    RankEnvelopeOnly,
    CoreLexicalV3,
}

/// Deterministic runner policy included in the report hash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CampaignConfig {
    pub selection: CampaignSelection,
    pub comparator_config: ComparatorConfig,
    /// Total public lexical contract required for selected query cases.
    #[serde(default)]
    pub contract_mode: CampaignContractMode,
    /// Require complete, environment-matched production provenance.
    ///
    /// Deterministic unit/regression fixtures may leave this disabled. Every
    /// live PR or nightly campaign enables it and therefore fails before
    /// either engine starts ingesting when provenance is missing or stale.
    #[serde(default)]
    pub require_provenance: bool,
    /// Maximum documents sent to each engine per identical indexing batch.
    pub index_batch_size: u64,
    /// Preferred canonical-JSON byte ceiling for an indexing batch.
    ///
    /// One individually valid document may exceed this preference and is sent
    /// alone; the generator's hard per-document cap remains authoritative.
    pub index_batch_max_bytes: u64,
    pub tie_expansion_limit: u64,
    pub snippet_max_chars: Option<u64>,
    /// One-sided posterior confidence, stored as raw f64 bits.
    pub posterior_confidence_bits: u64,
}

impl Default for CampaignConfig {
    fn default() -> Self {
        Self {
            selection: CampaignSelection::All,
            comparator_config: ComparatorConfig::default(),
            contract_mode: CampaignContractMode::RankEnvelopeOnly,
            require_provenance: false,
            index_batch_size: 4_096,
            index_batch_max_bytes: 16 * 1024 * 1024,
            tie_expansion_limit: 256,
            snippet_max_chars: Some(200),
            posterior_confidence_bits: 0.95_f64.to_bits(),
        }
    }
}

impl CampaignConfig {
    fn validate(&self) -> Result<(), GauntletError> {
        self.comparator_config.validate_contract()?;
        let confidence = f64::from_bits(self.posterior_confidence_bits);
        if self.index_batch_size == 0
            || self.index_batch_size > 100_000
            || self.index_batch_max_bytes == 0
            || self.index_batch_max_bytes > u64::from(MAX_DOCUMENT_BYTES) * 512
            || self.tie_expansion_limit > 100_000
            || self
                .snippet_max_chars
                .is_some_and(|value| value > MAX_SNIPPET_CHARS)
            || !confidence.is_finite()
            || !(0.0 < confidence && confidence < 1.0)
        {
            return Err(campaign_error(
                "campaign limits or posterior confidence are outside their bounded contracts",
            ));
        }
        Ok(())
    }
}

/// Gate disposition for one submitted query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CampaignDisposition {
    Exact,
    AutoClassified,
    RegisterClassified,
    Unclassified,
    InfrastructureError,
}

impl CampaignDisposition {
    const fn passes(self) -> bool {
        matches!(
            self,
            Self::Exact | Self::AutoClassified | Self::RegisterClassified
        )
    }
}

/// Total-lexical evidence retained for one submitted query.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case", deny_unknown_fields)]
pub enum CampaignLexicalCaseSummary {
    /// Decode-only marker for pre-v5 campaign reports.
    #[default]
    LegacyMissing,
    /// The case exercised only the older rich result-envelope comparator.
    RankEnvelopeOnly,
    /// Core-v3 was required, but observation or replay failed before an
    /// immutable comparison could be persisted.
    CoreLexicalV3Unavailable,
    /// Replay-derived summary of the immutable total-contract comparison.
    CoreLexicalV3 {
        status: LexicalComparisonStatus,
        first_mismatch: Option<LexicalFieldMismatch>,
        mismatch_count: u64,
        waived_difference_count: u64,
    },
}

/// Stable evidence row for one campaign query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CampaignCaseResult {
    pub case_id: String,
    pub query_class: String,
    pub disposition: CampaignDisposition,
    pub comparison_status: Option<ComparisonStatus>,
    pub rank_class: Option<RankClass>,
    /// Explicit scope and result of the total lexical comparison.
    #[serde(default)]
    pub lexical_contract: CampaignLexicalCaseSummary,
    /// Domain-separated SHA-256 address of the current immutable artifact object.
    ///
    /// Legacy 16-hex XXH3-64 addresses are decode-only and cannot satisfy a
    /// current campaign report.
    pub artifact_hash: Option<String>,
    pub registered_divergence: Option<DivergenceRegisterEntry>,
    pub first_divergence: Option<String>,
    /// Stable machine-facing outcome reason included in canonical reports.
    pub reason: Option<String>,
    /// Noncanonical backend/OS diagnostic retained in memory for triage.
    #[serde(default, skip)]
    pub diagnostic: Option<String>,
}

/// Per-query-class raw counts and an informational Beta(1,1) posterior bound.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryClassSummary {
    pub query_class: String,
    pub total: u64,
    pub exact: u64,
    pub auto_classified: u64,
    pub register_classified: u64,
    pub unclassified: u64,
    pub infrastructure_errors: u64,
    /// Canonical confidence input; the libm-derived bound is computed on demand.
    pub posterior_confidence_bits: u64,
}

impl QueryClassSummary {
    /// Point pass rate reconstructed from its artifact-stable bits.
    #[must_use]
    pub fn pass_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let passed = self
            .exact
            .saturating_add(self.auto_classified)
            .saturating_add(self.register_classified);
        passed as f64 / self.total as f64
    }

    /// One-sided posterior lower bound reconstructed from artifact-stable bits.
    #[must_use]
    pub fn posterior_lower_bound(&self) -> f64 {
        let passed = self
            .exact
            .saturating_add(self.auto_classified)
            .saturating_add(self.register_classified);
        beta_posterior_lower_bound(
            passed,
            self.total,
            f64::from_bits(self.posterior_confidence_bits),
        )
    }
}

/// Deduplicated mismatch descriptor with every affected fixture retained.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MismatchGroup {
    pub signature: String,
    pub divergence: Divergence,
    pub occurrence_count: u64,
    pub case_ids: Vec<String>,
}

/// Deduplicated total-lexical mismatch descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexicalMismatchGroup {
    pub signature: String,
    pub mismatch: LexicalFieldMismatch,
    pub occurrence_count: u64,
    pub case_ids: Vec<String>,
}

/// Aggregate count of each replay-derived probe state.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeCoverageCounts {
    pub success: u64,
    pub restoration: u64,
    pub error: u64,
    pub empty: u64,
    pub not_run: u64,
}

/// Aggregate coverage for every public lexical boundary on one engine side.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalSideCoverageCounts {
    pub full_search: ProbeCoverageCounts,
    pub fusion_candidates: ProbeCoverageCounts,
    pub all_lexical_winners_hydration: ProbeCoverageCounts,
    pub strict_hybrid_winners_hydration: ProbeCoverageCounts,
    pub semantic_only_hydration: ProbeCoverageCounts,
    pub mixed_winners_hydration: ProbeCoverageCounts,
    pub metadata_deferred_cases: u64,
}

/// Campaign-level total-contract coverage, kept separate from equivalence.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case", deny_unknown_fields)]
pub enum CampaignLexicalCoverageSummary {
    /// Decode-only marker for pre-v5 campaign reports.
    #[default]
    LegacyMissing,
    /// This campaign intentionally exercised only the rich rank envelope.
    RankEnvelopeOnly,
    /// Aggregated immutable core-v3 probe coverage.
    CoreLexicalV3 {
        subject: Box<LexicalSideCoverageCounts>,
        oracle: Box<LexicalSideCoverageCounts>,
        admissible: bool,
    },
}

/// Deterministic campaign report; wall-clock data deliberately lives outside it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CampaignReport {
    pub schema_version: u32,
    pub run_id: String,
    pub engines: EnginePairIdentity,
    pub semantic_contract: SemanticContract,
    pub config: CampaignConfig,
    pub divergence_registry: DivergenceRegistry,
    pub corpus_manifest: CorpusManifest,
    pub corpus_manifest_hash: String,
    pub query_suite: GeneratedQuerySuite,
    pub query_manifest_hash: String,
    pub subject_index: EngineIndexReceipt,
    pub oracle_index: EngineIndexReceipt,
    pub submitted_query_count: u64,
    pub selected_query_count: u64,
    pub cases: Vec<CampaignCaseResult>,
    pub query_classes: Vec<QueryClassSummary>,
    pub mismatches: Vec<MismatchGroup>,
    #[serde(default)]
    pub lexical_mismatches: Vec<LexicalMismatchGroup>,
    #[serde(default)]
    pub lexical_coverage: CampaignLexicalCoverageSummary,
    pub passed: bool,
    /// Immutable source/toolchain provenance for production campaigns
    /// (bd-quill-e6-gauntlet-scale-rm3q.9). Deterministic regression fixtures
    /// that are not independently observed live provenance leave it absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<CampaignProvenance>,
}

#[derive(Serialize)]
struct CampaignRunReservation<'a> {
    schema_version: u32,
    run_id: &'a str,
    engines: &'a EnginePairIdentity,
    semantic_contract: &'a SemanticContract,
    config: &'a CampaignConfig,
    corpus_manifest_hash: &'a str,
    query_manifest_hash: &'a str,
    query_source_identity_sha256: &'a str,
    divergence_registry_hash: &'a str,
    selected_case_ids: Vec<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    provenance: Option<&'a CampaignProvenance>,
}

/// Immutable source and semantic provenance stamped into every production
/// campaign (bd-quill-e6-gauntlet-scale-rm3q.9).
///
/// A campaign is admissible evidence only with the full environment pinned:
/// engine commits and dirty state, lockfile, exact toolchain, Unicode tables,
/// and the query profile identity. Missing or mismatched provenance fails
/// closed; the reservation equality check inside
/// [`ArtifactStore::load_verified_campaign`] replays these bytes verbatim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignProvenance {
    /// Subject engine source commit.
    pub subject_git_revision: String,
    /// Whether the subject source tree had uncommitted changes.
    pub subject_source_dirty: bool,
    /// Oracle engine source commit.
    pub oracle_git_revision: String,
    /// Whether the oracle source tree had uncommitted changes.
    pub oracle_source_dirty: bool,
    /// SHA-256 of the workspace `Cargo.lock`.
    pub cargo_lock_sha256: String,
    /// Full `rustc -Vv` output (release, commit, date, host).
    pub rustc_version_verbose: String,
    /// Exact dated channel from `rust-toolchain.toml`, cross-checked against
    /// `rustup`'s active toolchain.
    pub rust_toolchain_channel: String,
    /// `std::char::UNICODE_VERSION` of the executing toolchain.
    pub unicode_version: String,
    /// Locked `unicode-normalization` crate version from `Cargo.lock`.
    pub unicode_normalization_version: String,
    /// Unicode data-table version compiled into `unicode-normalization`.
    pub unicode_normalization_table_version: String,
    /// Query generator implementation identity.
    pub query_generator_id: String,
    /// Query-manifest schema implemented by the generator.
    pub query_generator_schema_version: u32,
    /// Independent query-generator seed.
    pub query_seed: u64,
    /// Content-addressed identity of the exact query-suite source.
    pub query_source_identity_sha256: String,
    /// Canonical hash of generator, source, selection, and semantic profile.
    pub query_profile_sha256: String,
    /// Analyzer protocol selected by the campaign.
    pub analyzer_contract_hash: String,
    /// Schema/query/ranking protocol selected by the campaign.
    pub schema_contract_hash: String,
    /// Canonical corpus-manifest hash.
    pub corpus_manifest_hash: String,
    /// Canonical query-manifest hash.
    pub query_manifest_hash: String,
    /// Synthetic corpus seed when generator-produced; explicit `null` for
    /// non-synthetic corpora. The field itself is always required.
    #[serde(deserialize_with = "deserialize_required_optional_u64")]
    pub corpus_seed: Option<u64>,
}

fn deserialize_required_optional_u64<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<u64>::deserialize(deserializer)
}

impl CampaignProvenance {
    /// Collect the full immutable provenance for one campaign execution.
    ///
    /// Subject Git state is read from the invoking checkout unless explicitly
    /// overridden. Oracle state defaults to the committed oracle version
    /// contract and may likewise be overridden by a complete revision/dirty
    /// pair. The CI path supplies both pairs explicitly. Toolchain facts come
    /// from the executing `rustc` and workspace `Cargo.lock`.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError::InvalidCampaign`] when any required fact
    /// cannot be collected; a campaign must fail closed, never guess.
    pub fn collect(
        corpus_manifest: &CorpusManifest,
        query_manifest: &QueryManifest,
        selection: &CampaignSelection,
        semantic_contract: &SemanticContract,
    ) -> Result<Self, GauntletError> {
        corpus_manifest.validate_contract()?;
        semantic_contract.validate()?;
        let corpus_manifest_hash = corpus_manifest.manifest_hash()?;
        if query_manifest.corpus_manifest_hash != corpus_manifest_hash {
            return Err(campaign_error(
                "provenance query manifest is not bound to the supplied corpus manifest",
            ));
        }
        let query_manifest_hash = query_manifest.manifest_hash()?;
        let corpus_seed = match &corpus_manifest.source {
            CorpusSourceManifest::Synthetic { spec } => Some(spec.seed),
            CorpusSourceManifest::SharedFixtures { .. }
            | CorpusSourceManifest::Repository { .. } => None,
        };
        let (subject_git_revision, subject_source_dirty) =
            collect_git_state("GAUNTLET_SUBJECT_REVISION", "GAUNTLET_SUBJECT_DIRTY")?;
        let (oracle_git_revision, oracle_source_dirty) = collect_oracle_git_state()?;
        let cargo_lock_sha256 = hash_workspace_lockfile()?;
        let rustc_version_verbose = collect_rustc_verbose()?;
        let unicode_normalization_version = locked_crate_version("unicode-normalization")?;
        let query_profile_sha256 =
            query_profile_sha256(query_manifest, selection, semantic_contract)?;
        Ok(Self {
            subject_git_revision,
            subject_source_dirty,
            oracle_git_revision,
            oracle_source_dirty,
            cargo_lock_sha256,
            rustc_version_verbose,
            rust_toolchain_channel: collect_dated_toolchain_channel()?,
            unicode_version: format!(
                "{}.{}.{}",
                char::UNICODE_VERSION.0,
                char::UNICODE_VERSION.1,
                char::UNICODE_VERSION.2
            ),
            unicode_normalization_version,
            unicode_normalization_table_version: unicode_normalization_table_version(),
            query_generator_id: query_manifest.generator_id.clone(),
            query_generator_schema_version: query_manifest.schema_version,
            query_seed: query_manifest.spec.seed,
            query_source_identity_sha256: query_manifest.source_identity_sha256.clone(),
            query_profile_sha256,
            analyzer_contract_hash: semantic_contract.analyzer_contract_hash.clone(),
            schema_contract_hash: semantic_contract.schema_contract_hash.clone(),
            corpus_manifest_hash,
            query_manifest_hash,
            corpus_seed,
        })
    }

    fn validate_for_campaign(
        &self,
        engines: &EnginePairIdentity,
        semantic_contract: &SemanticContract,
        config: &CampaignConfig,
        corpus_manifest: &CorpusManifest,
        query_manifest: &QueryManifest,
    ) -> Result<(), GauntletError> {
        let expected_corpus_seed = match &corpus_manifest.source {
            CorpusSourceManifest::Synthetic { spec } => Some(spec.seed),
            CorpusSourceManifest::SharedFixtures { .. }
            | CorpusSourceManifest::Repository { .. } => None,
        };
        let expected_unicode_version = format!(
            "{}.{}.{}",
            char::UNICODE_VERSION.0,
            char::UNICODE_VERSION.1,
            char::UNICODE_VERSION.2
        );
        let expected_query_profile =
            query_profile_sha256(query_manifest, &config.selection, semantic_contract)?;
        let expected_corpus_manifest_hash = corpus_manifest.manifest_hash()?;
        let expected_query_manifest_hash = query_manifest.manifest_hash()?;
        let rustc_is_complete = self.rustc_version_verbose.len() <= 16 * 1024
            && !self.rustc_version_verbose.contains('\0')
            && ["commit-hash:", "commit-date:", "host:", "release:"]
                .iter()
                .all(|label| {
                    self.rustc_version_verbose
                        .lines()
                        .any(|line| line.starts_with(label))
                });

        if !is_git_revision(&self.subject_git_revision)
            || !is_git_revision(&self.oracle_git_revision)
            || self.subject_git_revision != engines.subject.source_revision
            || self.subject_source_dirty != engines.subject.source_dirty
            || self.oracle_git_revision != engines.oracle.source_revision
            || self.oracle_source_dirty != engines.oracle.source_dirty
            || !is_lower_sha256(&self.cargo_lock_sha256)
            || self.cargo_lock_sha256 != hash_workspace_lockfile()?
            || !rustc_is_complete
            || self.rustc_version_verbose != collect_rustc_verbose()?
            || self.rust_toolchain_channel != collect_dated_toolchain_channel()?
            || self.unicode_version != expected_unicode_version
            || self.unicode_normalization_version != locked_crate_version("unicode-normalization")?
            || self.unicode_normalization_table_version != unicode_normalization_table_version()
            || self.query_generator_id != GENERATOR_ID
            || self.query_generator_id != query_manifest.generator_id
            || self.query_generator_schema_version != QUERY_MANIFEST_SCHEMA_VERSION
            || self.query_generator_schema_version != query_manifest.schema_version
            || self.query_seed != query_manifest.spec.seed
            || !is_lower_sha256(&self.query_source_identity_sha256)
            || self.query_source_identity_sha256 != query_manifest.source_identity_sha256
            || !is_lower_sha256(&self.query_profile_sha256)
            || self.query_profile_sha256 != expected_query_profile
            || self.analyzer_contract_hash != semantic_contract.analyzer_contract_hash
            || self.schema_contract_hash != semantic_contract.schema_contract_hash
            || self.corpus_manifest_hash != expected_corpus_manifest_hash
            || self.query_manifest_hash != expected_query_manifest_hash
            || self.corpus_seed != expected_corpus_seed
        {
            return Err(campaign_error(
                "campaign provenance is missing, malformed, or does not match the exact engines, toolchain, Unicode tables, corpus, and query profile",
            ));
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct QueryProfilePreimage<'a> {
    schema_version: u32,
    generator_id: &'a str,
    generator_schema_version: u32,
    query_seed: u64,
    query_source: QuerySuiteSource,
    query_source_identity_sha256: &'a str,
    selection: &'a CampaignSelection,
    semantic_contract: &'a SemanticContract,
}

fn query_profile_sha256(
    query_manifest: &QueryManifest,
    selection: &CampaignSelection,
    semantic_contract: &SemanticContract,
) -> Result<String, GauntletError> {
    let profile = QueryProfilePreimage {
        schema_version: 1,
        generator_id: &query_manifest.generator_id,
        generator_schema_version: query_manifest.schema_version,
        query_seed: query_manifest.spec.seed,
        query_source: query_manifest.source,
        query_source_identity_sha256: &query_manifest.source_identity_sha256,
        selection,
        semantic_contract,
    };
    let bytes = serde_json::to_vec(&profile).map_err(|error| GauntletError::InvalidCampaign {
        reason: format!("query profile serialization failed: {error}"),
    })?;
    Ok(sha256_hex(&bytes))
}

fn unicode_normalization_table_version() -> String {
    format!(
        "{}.{}.{}",
        unicode_normalization::UNICODE_VERSION.0,
        unicode_normalization::UNICODE_VERSION.1,
        unicode_normalization::UNICODE_VERSION.2
    )
}

fn collect_git_state(revision_env: &str, dirty_env: &str) -> Result<(String, bool), GauntletError> {
    match (std::env::var(revision_env), std::env::var(dirty_env)) {
        (Ok(revision), Ok(dirty)) => {
            return Ok((revision, parse_dirty_state(&dirty, dirty_env)?));
        }
        (Err(std::env::VarError::NotPresent), Err(std::env::VarError::NotPresent)) => {}
        _ => {
            return Err(campaign_error(format!(
                "provenance overrides {revision_env} and {dirty_env} must be supplied together as valid UTF-8"
            )));
        }
    }
    let revision = run_capture(ProvenanceProgram::Git, &["rev-parse", "HEAD"], revision_env)?;
    let porcelain = run_capture(
        ProvenanceProgram::Git,
        &["status", "--porcelain"],
        dirty_env,
    )?;
    Ok((revision.trim().to_owned(), !porcelain.trim().is_empty()))
}

fn collect_oracle_git_state() -> Result<(String, bool), GauntletError> {
    const REVISION_ENV: &str = "GAUNTLET_ORACLE_REVISION";
    const DIRTY_ENV: &str = "GAUNTLET_ORACLE_DIRTY";
    match (std::env::var(REVISION_ENV), std::env::var(DIRTY_ENV)) {
        (Ok(revision), Ok(dirty)) => Ok((revision, parse_dirty_state(&dirty, DIRTY_ENV)?)),
        (Err(std::env::VarError::NotPresent), Err(std::env::VarError::NotPresent)) => {
            let contract = oracle_version_contract()?;
            Ok((contract.lexical_git_revision, false))
        }
        _ => Err(campaign_error(format!(
            "provenance overrides {REVISION_ENV} and {DIRTY_ENV} must be supplied together as valid UTF-8"
        ))),
    }
}

fn parse_dirty_state(value: &str, label: &str) -> Result<bool, GauntletError> {
    match value {
        "1" => Ok(true),
        "0" => Ok(false),
        _ if value.eq_ignore_ascii_case("true") => Ok(true),
        _ if value.eq_ignore_ascii_case("false") => Ok(false),
        _ => Err(campaign_error(format!(
            "provenance dirty-state override {label} must be true, false, 1, or 0"
        ))),
    }
}

#[derive(Clone, Copy)]
enum ProvenanceProgram {
    Git,
    Rustc,
    Rustup,
}

impl ProvenanceProgram {
    fn command(self) -> std::process::Command {
        match self {
            Self::Git => std::process::Command::new("git"),
            Self::Rustc => std::process::Command::new("rustc"),
            Self::Rustup => std::process::Command::new("rustup"),
        }
    }
}

fn run_capture(
    program: ProvenanceProgram,
    args: &[&str],
    label: &str,
) -> Result<String, GauntletError> {
    let output =
        program
            .command()
            .args(args)
            .output()
            .map_err(|error| GauntletError::InvalidCampaign {
                reason: format!("provenance collection failed to spawn {label}: {error}"),
            })?;
    if !output.status.success() {
        return Err(GauntletError::InvalidCampaign {
            reason: format!("provenance collection command {label} failed: {output:?}"),
        });
    }
    String::from_utf8(output.stdout).map_err(|error| GauntletError::InvalidCampaign {
        reason: format!("provenance collection output for {label} is not UTF-8: {error}"),
    })
}

fn collect_rustc_verbose() -> Result<String, GauntletError> {
    run_capture(ProvenanceProgram::Rustc, &["-Vv"], "rustc -Vv")
}

fn workspace_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn collect_dated_toolchain_channel() -> Result<String, GauntletError> {
    let path = workspace_root().join("rust-toolchain.toml");
    let contents =
        std::fs::read_to_string(&path).map_err(|error| GauntletError::InvalidCampaign {
            reason: format!("cannot read {}: {error}", path.display()),
        })?;
    let parsed: toml::Value =
        toml::from_str(&contents).map_err(|error| GauntletError::InvalidCampaign {
            reason: format!("cannot parse {}: {error}", path.display()),
        })?;
    let channel = parsed
        .get("toolchain")
        .and_then(|toolchain| toolchain.get("channel"))
        .and_then(toml::Value::as_str)
        .ok_or_else(|| campaign_error("rust-toolchain.toml does not declare a channel"))?;
    let Some(date) = channel.strip_prefix("nightly-") else {
        return Err(campaign_error(
            "production campaigns require a committed dated nightly toolchain",
        ));
    };
    if !is_review_date(date) {
        return Err(campaign_error(
            "production campaign nightly channel has a malformed date",
        ));
    }
    let active = run_capture(
        ProvenanceProgram::Rustup,
        &["show", "active-toolchain"],
        "rustup active toolchain",
    )?;
    let active_channel = active
        .split_whitespace()
        .next()
        .ok_or_else(|| campaign_error("rustup did not report an active toolchain"))?;
    if active_channel != channel
        && !active_channel
            .strip_prefix(channel)
            .is_some_and(|target| target.starts_with('-') && target.len() > 1)
    {
        return Err(campaign_error(format!(
            "active Rust toolchain {active_channel:?} does not match committed channel {channel:?}"
        )));
    }
    Ok(channel.to_owned())
}

fn hash_workspace_lockfile() -> Result<String, GauntletError> {
    let path = workspace_root().join("Cargo.lock");
    let bytes = std::fs::read(&path).map_err(|error| GauntletError::InvalidCampaign {
        reason: format!("cannot read {}: {error}", path.display()),
    })?;
    Ok(sha256_hex(&bytes))
}

fn locked_crate_version(crate_name: &str) -> Result<String, GauntletError> {
    let path = workspace_root().join("Cargo.lock");
    let contents =
        std::fs::read_to_string(&path).map_err(|error| GauntletError::InvalidCampaign {
            reason: format!("cannot read {}: {error}", path.display()),
        })?;
    let needle = format!("name = \"{crate_name}\"");
    let mut lines = contents.lines();
    while let Some(line) = lines.next() {
        if line.trim() == needle {
            for entry in lines.by_ref().take(4) {
                let trimmed = entry.trim();
                if let Some(version) = trimmed.strip_prefix("version = \"") {
                    return Ok(version.trim_end_matches('"').to_owned());
                }
            }
            break;
        }
    }
    Err(GauntletError::InvalidCampaign {
        reason: format!("crate {crate_name} not found in {}", path.display()),
    })
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

#[derive(Serialize)]
struct LexicalQueryContractPreimage<'a> {
    schema_version: u32,
    analyzer_contract_sha256: &'a str,
    schema_contract_sha256: &'a str,
}

pub fn lexical_query_contract_sha256(
    semantic_contract: &SemanticContract,
) -> Result<String, GauntletError> {
    semantic_contract.validate()?;
    let preimage = LexicalQueryContractPreimage {
        schema_version: 1,
        analyzer_contract_sha256: &semantic_contract.analyzer_contract_hash,
        schema_contract_sha256: &semantic_contract.schema_contract_hash,
    };
    let bytes = serde_json::to_vec(&preimage).map_err(|error| {
        campaign_error(format!(
            "lexical query-contract identity serialization failed: {error}"
        ))
    })?;
    let mut hasher = Sha256::new();
    hasher.update(LEXICAL_QUERY_CONTRACT_DOMAIN);
    hasher.update(bytes);
    Ok(lower_hex(&hasher.finalize()))
}

#[derive(Serialize)]
struct LexicalIndexIdentityPreimage<'a> {
    schema_version: u32,
    descriptor: &'a EngineDescriptor,
    logical_snapshot_sha256: &'a str,
}

pub fn lexical_backend_identity(
    descriptor: &EngineDescriptor,
    logical_snapshot_sha256: &str,
) -> Result<LexicalBackendIdentity, GauntletError> {
    if !is_lower_sha256(logical_snapshot_sha256) {
        return Err(campaign_error(
            "lexical backend logical snapshot must be lowercase SHA-256",
        ));
    }
    let preimage = LexicalIndexIdentityPreimage {
        schema_version: 1,
        descriptor,
        logical_snapshot_sha256,
    };
    let bytes = serde_json::to_vec(&preimage).map_err(|error| {
        campaign_error(format!(
            "lexical backend identity serialization failed: {error}"
        ))
    })?;
    let mut hasher = Sha256::new();
    hasher.update(LEXICAL_INDEX_IDENTITY_DOMAIN);
    hasher.update(bytes);
    Ok(LexicalBackendIdentity {
        engine: descriptor.implementation.clone(),
        revision: descriptor.source_revision.clone(),
        index_identity: lower_hex(&hasher.finalize()),
    })
}

impl CampaignReport {
    /// Validate every self-contained report invariant before structural hashing.
    ///
    /// This does not trust stored summary fields: manifest identities,
    /// selection/order, receipts, dispositions, class summaries, mismatch
    /// structure, and the final pass bit are all recomputed. This deliberately
    /// does not prove that referenced immutable artifacts exist or agree with
    /// the reported classifications. Use
    /// [`crate::ArtifactStore::load_verified_campaign`] when evidence-backed
    /// replay verification is required.
    ///
    /// # Errors
    ///
    /// Returns an error when any report field is malformed or inconsistent.
    pub fn validate_contract(&self) -> Result<(), GauntletError> {
        validate_campaign_run_id(&self.run_id)?;
        self.semantic_contract.validate()?;
        self.config.validate()?;
        self.divergence_registry.validate()?;
        if matches!(self.schema_version, 3 | 4) {
            return Err(campaign_error(
                "legacy campaign report schema lacks the current total lexical contract and is non-admissible; rerun the campaign",
            ));
        }
        if self.schema_version != CAMPAIGN_REPORT_SCHEMA_VERSION {
            return Err(campaign_error("campaign report schema version is invalid"));
        }

        let mut rebuilt_engines = EnginePairIdentity::new(
            self.engines.comparison_mode,
            self.engines.subject.clone(),
            self.engines.oracle.clone(),
        )?;
        rebuilt_engines.bind_semantic_contract(self.semantic_contract.clone())?;
        self.engines.validate_gauntlet_contract()?;
        if rebuilt_engines != self.engines {
            return Err(campaign_error(
                "campaign report engine and semantic identities are inconsistent",
            ));
        }
        self.corpus_manifest.validate_contract()?;
        if self.corpus_manifest.manifest_hash()? != self.corpus_manifest_hash {
            return Err(campaign_error(
                "campaign report corpus manifest hash is inconsistent",
            ));
        }
        self.query_suite.manifest.verify(&self.query_suite.cases)?;
        if self.query_suite.manifest.corpus_manifest_hash != self.corpus_manifest_hash
            || self.query_suite.manifest.manifest_hash()? != self.query_manifest_hash
        {
            return Err(campaign_error(
                "campaign report query suite is not bound to its manifest and corpus",
            ));
        }
        match (&self.provenance, self.config.require_provenance) {
            (Some(provenance), _) => provenance.validate_for_campaign(
                &self.engines,
                &self.semantic_contract,
                &self.config,
                &self.corpus_manifest,
                &self.query_suite.manifest,
            )?,
            (None, true) => {
                return Err(campaign_error(
                    "production campaign report is missing required provenance",
                ));
            }
            (None, false) => {}
        }

        let selected = self.config.selection.select(&self.query_suite.cases)?;
        let submitted_count = u64::try_from(self.query_suite.cases.len()).unwrap_or(u64::MAX);
        let selected_count = u64::try_from(selected.len()).unwrap_or(u64::MAX);
        if self.submitted_query_count != submitted_count
            || self.submitted_query_count != self.query_suite.manifest.query_count
            || self.selected_query_count != selected_count
            || self.cases.len() != selected.len()
        {
            return Err(campaign_error(
                "campaign report submitted or selected query counts are inconsistent",
            ));
        }

        let expected_receipt = EngineIndexReceipt::for_manifest(
            &self.corpus_manifest,
            self.semantic_contract.clone(),
        )?;
        if self.subject_index != expected_receipt || self.oracle_index != expected_receipt {
            return Err(campaign_error(
                "campaign report index receipts do not match its corpus manifest",
            ));
        }

        for (query, result) in selected.iter().zip(&self.cases) {
            validate_campaign_case_result(
                query,
                result,
                &self.divergence_registry,
                self.config.contract_mode,
            )?;
        }
        let confidence = f64::from_bits(self.config.posterior_confidence_bits);
        if summarize_query_classes(&self.cases, confidence) != self.query_classes {
            return Err(campaign_error(
                "campaign report query-class summaries are inconsistent",
            ));
        }
        validate_mismatch_groups(&self.mismatches, &self.cases)?;
        validate_lexical_mismatch_groups(&self.lexical_mismatches, &self.cases)?;
        if self.config.contract_mode == CampaignContractMode::RankEnvelopeOnly
            && !self.lexical_mismatches.is_empty()
        {
            return Err(campaign_error(
                "rank-envelope-only campaign cannot claim total lexical mismatch evidence",
            ));
        }
        validate_lexical_coverage_summary(
            &self.lexical_coverage,
            self.config.contract_mode,
            self.selected_query_count,
        )?;
        if self.passed
            != (self.cases.iter().all(|result| result.disposition.passes())
                && lexical_coverage_is_admissible(&self.lexical_coverage))
        {
            return Err(campaign_error(
                "campaign report pass bit does not match case dispositions and lexical coverage",
            ));
        }
        Ok(())
    }

    pub(crate) fn reservation_bytes_unchecked(&self) -> Result<Vec<u8>, GauntletError> {
        let selected = self.config.selection.select(&self.query_suite.cases)?;
        let divergence_registry_hash = self.divergence_registry.registry_hash()?;
        let reservation = CampaignRunReservation {
            schema_version: self.schema_version,
            run_id: &self.run_id,
            engines: &self.engines,
            semantic_contract: &self.semantic_contract,
            config: &self.config,
            corpus_manifest_hash: &self.corpus_manifest_hash,
            query_manifest_hash: &self.query_manifest_hash,
            query_source_identity_sha256: &self.query_suite.manifest.source_identity_sha256,
            divergence_registry_hash: &divergence_registry_hash,
            selected_case_ids: selected.iter().map(|query| query.id.as_str()).collect(),
            provenance: self.provenance.as_ref(),
        };
        Ok(serde_json::to_vec(&reservation)?)
    }

    pub(crate) fn selected_queries(&self) -> Result<Vec<&GeneratedQueryCase>, GauntletError> {
        self.config.selection.select(&self.query_suite.cases)
    }

    pub(crate) fn begin_evidence_validation(
        &self,
    ) -> Result<CampaignEvidenceValidator<'_>, GauntletError> {
        CampaignEvidenceValidator::new(self)
    }

    /// Canonical compact JSON for the report's self-contained structure.
    ///
    /// # Errors
    ///
    /// Returns an error if validation or serialization fails.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, GauntletError> {
        self.validate_contract()?;
        self.canonical_bytes_unchecked()
    }

    pub(crate) fn canonical_bytes_unchecked(&self) -> Result<Vec<u8>, GauntletError> {
        Ok(serde_json::to_vec(self)?)
    }

    /// Domain-separated lowercase SHA-256 of the self-contained report.
    ///
    /// # Errors
    ///
    /// Returns an error if validation or serialization fails.
    pub fn report_hash(&self) -> Result<String, GauntletError> {
        let mut hasher = Sha256::new();
        hasher.update(CAMPAIGN_REPORT_HASH_DOMAIN);
        hasher.update(self.canonical_bytes()?);
        Ok(lower_hex(&hasher.finalize()))
    }
}

/// Streaming cross-artifact validator for one structurally valid campaign report.
///
/// Artifacts must be observed once in selected-query ordinal order. The validator
/// retains only bounded mismatch aggregates, so campaign completion never holds
/// every decoded object in memory at once.
#[derive(Debug)]
pub struct CampaignEvidenceValidator<'a> {
    report: &'a CampaignReport,
    selected: Vec<&'a GeneratedQueryCase>,
    next_ordinal: usize,
    mismatches: MismatchCollection,
    lexical_mismatches: LexicalMismatchCollection,
    lexical_coverage: CampaignLexicalCoverageAccumulator,
}

impl<'a> CampaignEvidenceValidator<'a> {
    fn new(report: &'a CampaignReport) -> Result<Self, GauntletError> {
        let selected = report.config.selection.select(&report.query_suite.cases)?;
        if selected.len() != report.cases.len() {
            return Err(campaign_error(
                "campaign evidence count does not match the final report",
            ));
        }
        Ok(Self {
            report,
            selected,
            next_ordinal: 0,
            mismatches: MismatchCollection::default(),
            lexical_mismatches: LexicalMismatchCollection::default(),
            lexical_coverage: CampaignLexicalCoverageAccumulator::new(report.config.contract_mode),
        })
    }

    pub(super) fn observe(
        &mut self,
        artifact: Option<(&ArtifactObject, &str)>,
    ) -> Result<(), GauntletError> {
        let query = self.selected.get(self.next_ordinal).ok_or_else(|| {
            campaign_error("campaign supplied more artifacts than selected queries")
        })?;
        let result =
            self.report.cases.get(self.next_ordinal).ok_or_else(|| {
                campaign_error("campaign supplied more artifacts than report cases")
            })?;

        if result.disposition == CampaignDisposition::InfrastructureError {
            if artifact.is_some() {
                return Err(campaign_error(
                    "infrastructure-error case unexpectedly has a completed artifact",
                ));
            }
            self.advance()?;
            return Ok(());
        }

        let (object, object_hash) = artifact.ok_or_else(|| {
            campaign_error("non-infrastructure case is missing its completed artifact")
        })?;
        object.validate()?;
        let expected_case = evidence_case_for(
            &self.report.config,
            query,
            self.report.query_suite.manifest.spec.seed,
            self.report.query_suite.manifest.source,
            &self.report.corpus_manifest_hash,
        );
        let (disposition, reason, registered_divergence) = classify_case_with_lexical(
            query,
            &object.comparison,
            &object.lexical_contract,
            &self.report.divergence_registry,
        );
        let lexical_summary = lexical_case_summary(&object.lexical_contract)?;
        let expected_context = CampaignArtifactContext {
            corpus_manifest_hash: self.report.corpus_manifest_hash.clone(),
            query_manifest_hash: self.report.query_manifest_hash.clone(),
            query_suite_source: self.report.query_suite.manifest.source,
            query_source_identity_sha256: self
                .report
                .query_suite
                .manifest
                .source_identity_sha256
                .clone(),
            semantic_contract: self.report.semantic_contract.clone(),
            contract_mode: self.report.config.contract_mode,
            query_seed: self.report.query_suite.manifest.spec.seed,
            query: (*query).clone(),
            registered_divergence: registered_divergence.clone(),
        };
        if object.engines != self.report.engines
            || object.case != expected_case
            || object.comparator_config != self.report.config.comparator_config
            || object.campaign.as_ref() != Some(&expected_context)
            || result.disposition != disposition
            || result.reason != reason
            || result.registered_divergence != registered_divergence
            || result.comparison_status != Some(object.comparison.status)
            || result.rank_class != Some(object.comparison.rank_class)
            || result.lexical_contract != lexical_summary
            || result.first_divergence != object.comparison.first_divergence
            || result.artifact_hash.as_deref() != Some(object_hash)
        {
            return Err(campaign_error(
                "campaign case result does not match its immutable artifact",
            ));
        }
        self.mismatches.record(&object.comparison, &query.id)?;
        self.lexical_mismatches
            .record(&object.lexical_contract, &query.id)?;
        self.lexical_coverage.record(&object.lexical_contract);
        self.advance()
    }

    pub(super) fn finish(self) -> Result<(), GauntletError> {
        if self.next_ordinal != self.selected.len() || self.next_ordinal != self.report.cases.len()
        {
            return Err(campaign_error(
                "campaign evidence ended before every selected query was validated",
            ));
        }
        if self.mismatches.finish() != self.report.mismatches
            || self.lexical_mismatches.finish() != self.report.lexical_mismatches
            || self.lexical_coverage.finish() != self.report.lexical_coverage
        {
            return Err(campaign_error(
                "campaign mismatch groups or lexical coverage do not match immutable case artifacts",
            ));
        }
        Ok(())
    }

    fn advance(&mut self) -> Result<(), GauntletError> {
        self.next_ordinal = self
            .next_ordinal
            .checked_add(1)
            .ok_or_else(|| campaign_error("campaign evidence ordinal overflow"))?;
        Ok(())
    }
}

fn validate_campaign_case_result(
    query: &GeneratedQueryCase,
    result: &CampaignCaseResult,
    registry: &DivergenceRegistry,
    contract_mode: CampaignContractMode,
) -> Result<(), GauntletError> {
    if result.case_id != query.id
        || result.query_class != query_class(query)
        || !is_canonical_query_id(&result.case_id)
        || result.query_class.is_empty()
        || result.query_class.len() > MAX_QUERY_ID_BYTES * 2
    {
        return Err(campaign_error(
            "campaign case ID, order, or query class is inconsistent",
        ));
    }
    if result.reason.as_ref().is_some_and(|reason| {
        reason.is_empty()
            || reason.len() > MAX_CAMPAIGN_REASON_BYTES
            || reason.trim() != reason
            || reason.chars().any(char::is_control)
    }) || result.first_divergence.as_ref().is_some_and(|pointer| {
        !pointer.starts_with('/')
            || pointer.len() > MAX_CAMPAIGN_POINTER_BYTES
            || pointer.chars().any(char::is_control)
    }) {
        return Err(campaign_error(
            "campaign case reason or divergence pointer is not bounded canonical text",
        ));
    }

    let non_infrastructure_fields = result.comparison_status.is_some()
        && result.rank_class.is_some()
        && result.artifact_hash.as_deref().is_some_and(is_lower_sha256);
    let lexical_shape = match (&result.lexical_contract, contract_mode) {
        (CampaignLexicalCaseSummary::RankEnvelopeOnly, CampaignContractMode::RankEnvelopeOnly) => {
            true
        }
        (
            CampaignLexicalCaseSummary::CoreLexicalV3Unavailable,
            CampaignContractMode::CoreLexicalV3,
        ) => result.disposition == CampaignDisposition::InfrastructureError,
        (
            CampaignLexicalCaseSummary::CoreLexicalV3 {
                status,
                first_mismatch,
                mismatch_count,
                ..
            },
            CampaignContractMode::CoreLexicalV3,
        ) => match status {
            LexicalComparisonStatus::Equivalent => *mismatch_count == 0 && first_mismatch.is_none(),
            LexicalComparisonStatus::Mismatch => *mismatch_count > 0 && first_mismatch.is_some(),
        },
        (
            CampaignLexicalCaseSummary::LegacyMissing
            | CampaignLexicalCaseSummary::RankEnvelopeOnly
            | CampaignLexicalCaseSummary::CoreLexicalV3Unavailable
            | CampaignLexicalCaseSummary::CoreLexicalV3 { .. },
            CampaignContractMode::RankEnvelopeOnly | CampaignContractMode::CoreLexicalV3,
        ) => false,
    };
    let valid_shape = match result.disposition {
        CampaignDisposition::Exact => {
            non_infrastructure_fields
                && result.comparison_status == Some(ComparisonStatus::Exact)
                && result.rank_class == Some(RankClass::RankExact)
                && result.registered_divergence.is_none()
                && result.first_divergence.is_none()
                && result.reason.is_none()
        }
        CampaignDisposition::AutoClassified => {
            non_infrastructure_fields
                && result.comparison_status == Some(ComparisonStatus::Classified)
                && matches!(
                    result.rank_class,
                    Some(RankClass::TieOrder | RankClass::ScoreEpsilon)
                )
                && result.registered_divergence.is_none()
                && result.reason.is_none()
        }
        CampaignDisposition::RegisterClassified => {
            non_infrastructure_fields
                && result.comparison_status == Some(ComparisonStatus::Classified)
                && result.reason.is_none()
                && result.registered_divergence.as_ref().is_some_and(|entry| {
                    entry.validate().is_ok()
                        && registry.find(&entry.id) == Some(entry)
                        && query.expected_divergence.as_deref() == Some(entry.id.as_str())
                })
        }
        CampaignDisposition::Unclassified => {
            non_infrastructure_fields
                && result.registered_divergence.is_none()
                && result.reason.is_some()
        }
        CampaignDisposition::InfrastructureError => {
            result.comparison_status.is_none()
                && result.rank_class.is_none()
                && result.artifact_hash.is_none()
                && result.registered_divergence.is_none()
                && result.first_divergence.is_none()
                && result.reason.is_some()
        }
    };
    let lexical_disposition_matches = match &result.lexical_contract {
        CampaignLexicalCaseSummary::CoreLexicalV3 {
            status: LexicalComparisonStatus::Mismatch,
            ..
        } => {
            result.disposition == CampaignDisposition::Unclassified
                && result.reason.as_deref() == Some("lexical_contract_mismatch")
                && result.registered_divergence.is_none()
        }
        CampaignLexicalCaseSummary::CoreLexicalV3 {
            status: LexicalComparisonStatus::Equivalent,
            ..
        }
        | CampaignLexicalCaseSummary::RankEnvelopeOnly => true,
        CampaignLexicalCaseSummary::CoreLexicalV3Unavailable => {
            result.disposition == CampaignDisposition::InfrastructureError
        }
        CampaignLexicalCaseSummary::LegacyMissing => false,
    };
    if !valid_shape || !lexical_shape || !lexical_disposition_matches {
        return Err(campaign_error(
            "campaign case disposition, lexical scope, and evidence fields are inconsistent",
        ));
    }
    Ok(())
}

fn validate_mismatch_groups(
    mismatches: &[MismatchGroup],
    cases: &[CampaignCaseResult],
) -> Result<(), GauntletError> {
    if mismatches.len() > MAX_MISMATCH_GROUPS {
        return Err(campaign_error(
            "campaign mismatch groups exceed their count budget",
        ));
    }
    let selected_ids = cases
        .iter()
        .map(|case| case.case_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut previous_signature = None::<&str>;
    let mut aggregate_text_bytes = 0_usize;
    for group in mismatches {
        let sorted_unique_case_ids = group.case_ids.windows(2).all(|pair| pair[0] < pair[1]);
        let ids_are_valid = !group.case_ids.is_empty()
            && group.case_ids.iter().all(|case_id| {
                is_canonical_query_id(case_id) && selected_ids.contains(case_id.as_str())
            });
        let divergence_is_bounded = group.divergence.pointer.starts_with('/')
            && group.divergence.pointer.len() <= MAX_CAMPAIGN_POINTER_BYTES
            && group.divergence.oracle.len() <= MAX_CAMPAIGN_POINTER_BYTES
            && group.divergence.subject.len() <= MAX_CAMPAIGN_POINTER_BYTES;
        if !is_lower_sha256(&group.signature)
            || previous_signature.is_some_and(|previous| previous >= group.signature.as_str())
            || group.occurrence_count == 0
            || group.occurrence_count < u64::try_from(group.case_ids.len()).unwrap_or(u64::MAX)
            || !sorted_unique_case_ids
            || !ids_are_valid
            || !divergence_is_bounded
        {
            return Err(campaign_error(
                "campaign mismatch group is malformed, unsorted, or unbounded",
            ));
        }
        aggregate_text_bytes = aggregate_text_bytes
            .checked_add(group.signature.len())
            .and_then(|bytes| bytes.checked_add(group.divergence.pointer.len()))
            .and_then(|bytes| bytes.checked_add(group.divergence.oracle.len()))
            .and_then(|bytes| bytes.checked_add(group.divergence.subject.len()))
            .and_then(|bytes| {
                group
                    .case_ids
                    .iter()
                    .try_fold(bytes, |sum, case_id| sum.checked_add(case_id.len()))
            })
            .ok_or_else(|| campaign_error("campaign mismatch text byte count overflow"))?;
        previous_signature = Some(&group.signature);
    }
    if aggregate_text_bytes > MAX_MISMATCH_TEXT_BYTES {
        return Err(campaign_error(
            "campaign mismatch groups exceed their aggregate text budget",
        ));
    }
    Ok(())
}

fn validate_lexical_mismatch_groups(
    mismatches: &[LexicalMismatchGroup],
    cases: &[CampaignCaseResult],
) -> Result<(), GauntletError> {
    if mismatches.len() > MAX_MISMATCH_GROUPS {
        return Err(campaign_error(
            "campaign lexical mismatch groups exceed their count budget",
        ));
    }
    let selected_ids = cases
        .iter()
        .map(|case| case.case_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut previous_signature = None::<&str>;
    let mut aggregate_text_bytes = 0_usize;
    for group in mismatches {
        let sorted_unique_case_ids = group.case_ids.windows(2).all(|pair| pair[0] < pair[1]);
        let ids_are_valid = !group.case_ids.is_empty()
            && group.case_ids.iter().all(|case_id| {
                is_canonical_query_id(case_id) && selected_ids.contains(case_id.as_str())
            });
        let mismatch_is_bounded = group.mismatch.path.starts_with('/')
            && group.mismatch.path.len() <= MAX_CAMPAIGN_POINTER_BYTES
            && group.mismatch.oracle.len() <= MAX_CAMPAIGN_POINTER_BYTES
            && group.mismatch.subject.len() <= MAX_CAMPAIGN_POINTER_BYTES
            && !group.mismatch.path.chars().any(char::is_control)
            && !group.mismatch.oracle.chars().any(char::is_control)
            && !group.mismatch.subject.chars().any(char::is_control);
        if !is_lower_sha256(&group.signature)
            || previous_signature.is_some_and(|previous| previous >= group.signature.as_str())
            || group.signature != lexical_mismatch_signature(&group.mismatch)
            || group.occurrence_count == 0
            || group.occurrence_count < u64::try_from(group.case_ids.len()).unwrap_or(u64::MAX)
            || !sorted_unique_case_ids
            || !ids_are_valid
            || !mismatch_is_bounded
        {
            return Err(campaign_error(
                "campaign lexical mismatch group is malformed, unsorted, or unbounded",
            ));
        }
        aggregate_text_bytes = aggregate_text_bytes
            .checked_add(group.signature.len())
            .and_then(|bytes| bytes.checked_add(group.mismatch.path.len()))
            .and_then(|bytes| bytes.checked_add(group.mismatch.oracle.len()))
            .and_then(|bytes| bytes.checked_add(group.mismatch.subject.len()))
            .and_then(|bytes| {
                group
                    .case_ids
                    .iter()
                    .try_fold(bytes, |sum, case_id| sum.checked_add(case_id.len()))
            })
            .ok_or_else(|| campaign_error("campaign lexical mismatch text overflow"))?;
        previous_signature = Some(&group.signature);
    }
    if aggregate_text_bytes > MAX_MISMATCH_TEXT_BYTES {
        return Err(campaign_error(
            "campaign lexical mismatch groups exceed their aggregate text budget",
        ));
    }
    Ok(())
}

fn validate_lexical_coverage_summary(
    coverage: &CampaignLexicalCoverageSummary,
    mode: CampaignContractMode,
    selected_query_count: u64,
) -> Result<(), GauntletError> {
    match (mode, coverage) {
        (
            CampaignContractMode::RankEnvelopeOnly,
            CampaignLexicalCoverageSummary::RankEnvelopeOnly,
        ) => Ok(()),
        (
            CampaignContractMode::CoreLexicalV3,
            CampaignLexicalCoverageSummary::CoreLexicalV3 {
                subject,
                oracle,
                admissible,
            },
        ) => {
            let all_probes = [
                &subject.full_search,
                &subject.fusion_candidates,
                &subject.all_lexical_winners_hydration,
                &subject.strict_hybrid_winners_hydration,
                &subject.semantic_only_hydration,
                &subject.mixed_winners_hydration,
                &oracle.full_search,
                &oracle.fusion_candidates,
                &oracle.all_lexical_winners_hydration,
                &oracle.strict_hybrid_winners_hydration,
                &oracle.semantic_only_hydration,
                &oracle.mixed_winners_hydration,
            ];
            for probe in all_probes {
                let total = probe
                    .success
                    .checked_add(probe.restoration)
                    .and_then(|value| value.checked_add(probe.error))
                    .and_then(|value| value.checked_add(probe.empty))
                    .and_then(|value| value.checked_add(probe.not_run))
                    .ok_or_else(|| campaign_error("lexical coverage count overflow"))?;
                if total > selected_query_count {
                    return Err(campaign_error(
                        "lexical coverage exceeds the selected case count",
                    ));
                }
            }
            if subject.metadata_deferred_cases > selected_query_count
                || oracle.metadata_deferred_cases > selected_query_count
            {
                return Err(campaign_error(
                    "lexical deferred-capability coverage exceeds the selected case count",
                ));
            }
            let expected_admissible = lexical_side_coverage_is_admissible(subject)
                && lexical_side_coverage_is_admissible(oracle);
            if *admissible != expected_admissible {
                return Err(campaign_error(
                    "lexical coverage admissibility is inconsistent",
                ));
            }
            Ok(())
        }
        (
            CampaignContractMode::RankEnvelopeOnly | CampaignContractMode::CoreLexicalV3,
            CampaignLexicalCoverageSummary::LegacyMissing
            | CampaignLexicalCoverageSummary::RankEnvelopeOnly
            | CampaignLexicalCoverageSummary::CoreLexicalV3 { .. },
        ) => Err(campaign_error(
            "campaign lexical coverage scope does not match its hashed contract mode",
        )),
    }
}

/// Core E6.2 campaign runner.
#[derive(Debug, Clone)]
pub struct DifferentialCampaignRunner {
    store: ArtifactStore,
    semantic_contract: SemanticContract,
    config: CampaignConfig,
    registry: DivergenceRegistry,
    provenance: Option<CampaignProvenance>,
}

impl DifferentialCampaignRunner {
    /// Construct a runner after validating every fail-closed policy input.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed semantic hashes or runner bounds.
    pub fn new(
        store: ArtifactStore,
        semantic_contract: SemanticContract,
        config: CampaignConfig,
        registry: DivergenceRegistry,
    ) -> Result<Self, GauntletError> {
        semantic_contract.validate()?;
        config.validate()?;
        registry.validate()?;
        Ok(Self {
            store,
            semantic_contract,
            config,
            registry,
            provenance: None,
        })
    }

    /// Attach immutable production provenance (bd-quill-e6-gauntlet-scale-rm3q.9).
    /// Every production campaign run stamps it into the reservation and the
    /// report; regression fixtures deliberately leave it absent.
    #[must_use]
    pub fn with_provenance(mut self, provenance: CampaignProvenance) -> Self {
        self.provenance = Some(provenance);
        self
    }

    /// Verify, index, execute, compare, and persist one differential campaign.
    ///
    /// Manifest, identity, selection, and run-ID validation happen before either
    /// adapter is invoked. Per-query adapter/comparator/storage failures are
    /// recorded and do not suppress later cases; corpus-ingest failures abort
    /// because no subsequent observation can be trusted.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid replay inputs, identity/config drift, or a
    /// corpus-ingest failure.
    pub async fn run(
        &self,
        cx: &Cx,
        run_id: &str,
        subject: &mut dyn DifferentialCampaignEngine,
        oracle: &mut dyn DifferentialCampaignEngine,
        documents: &[GeneratedDocument],
        corpus_manifest: &CorpusManifest,
        query_suite: &GeneratedQuerySuite,
    ) -> Result<CampaignReport, GauntletError> {
        let replay = BorrowedCorpus(documents);
        self.run_replay(
            cx,
            run_id,
            subject,
            oracle,
            &replay,
            corpus_manifest,
            query_suite,
        )
        .await
    }

    /// Streaming variant of [`Self::run`] for deterministic generated corpora.
    ///
    /// The source is replayed once for manifest validation and once for
    /// indexing. The indexing replay is consumed in bounded batches, with each
    /// exact batch submitted to the subject and oracle in the same order.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid replay inputs, identity/config drift, or a
    /// corpus lifecycle failure.
    #[allow(clippy::too_many_arguments)]
    pub async fn run_replay(
        &self,
        cx: &Cx,
        run_id: &str,
        subject: &mut dyn DifferentialCampaignEngine,
        oracle: &mut dyn DifferentialCampaignEngine,
        documents: &dyn GeneratedCorpusReplay,
        corpus_manifest: &CorpusManifest,
        query_suite: &GeneratedQuerySuite,
    ) -> Result<CampaignReport, GauntletError> {
        self.config.validate()?;
        self.registry.validate()?;
        validate_campaign_run_id(run_id)?;
        corpus_manifest.verify_documents(documents.replay())?;
        let corpus_manifest_hash = corpus_manifest.manifest_hash()?;
        query_suite.manifest.verify(&query_suite.cases)?;
        if query_suite.manifest.corpus_manifest_hash != corpus_manifest_hash {
            return Err(campaign_error(
                "query manifest is not bound to the supplied corpus manifest",
            ));
        }
        let query_manifest_hash = query_suite.manifest.manifest_hash()?;
        let selected = self.config.selection.select(&query_suite.cases)?;
        let mut prepared_cases = Vec::with_capacity(selected.len());
        for query in selected {
            let evidence_case = self.evidence_case(
                query,
                query_suite.manifest.spec.seed,
                query_suite.manifest.source,
                &corpus_manifest_hash,
            );
            evidence_case.validate_shape()?;
            prepared_cases.push((query, query_class(query), evidence_case));
        }
        let selected_default_query = prepared_cases
            .iter()
            .any(|(query, _, _)| query.syntax == QuerySyntax::Default);
        let selected_non_default_query = prepared_cases
            .iter()
            .any(|(query, _, _)| query.syntax != QuerySyntax::Default);
        match self.config.contract_mode {
            CampaignContractMode::CoreLexicalV3 if selected_non_default_query => {
                return Err(campaign_error(
                    "core lexical v3 cannot be used for CASS or other non-default query syntax",
                ));
            }
            CampaignContractMode::RankEnvelopeOnly
                if self.config.require_provenance && selected_default_query =>
            {
                return Err(campaign_error(
                    "provenance-bearing default campaigns require core lexical v3 evidence",
                ));
            }
            CampaignContractMode::RankEnvelopeOnly | CampaignContractMode::CoreLexicalV3 => {}
        }
        let mut engines = EnginePairIdentity::new(
            ComparisonMode::CrossEngine,
            subject.descriptor(),
            oracle.descriptor(),
        )?;
        let subject_semantics = subject.semantic_contract();
        let oracle_semantics = oracle.semantic_contract();
        subject_semantics.validate()?;
        oracle_semantics.validate()?;
        if subject_semantics != self.semantic_contract || oracle_semantics != self.semantic_contract
        {
            return Err(campaign_error(
                "engine-declared semantic contracts do not match the campaign contract",
            ));
        }
        engines.bind_semantic_contract(self.semantic_contract.clone())?;
        engines.validate_gauntlet_contract()?;
        match (&self.provenance, self.config.require_provenance) {
            (Some(provenance), _) => provenance.validate_for_campaign(
                &engines,
                &self.semantic_contract,
                &self.config,
                corpus_manifest,
                &query_suite.manifest,
            )?,
            (None, true) => {
                return Err(campaign_error(
                    "production campaign is missing required provenance",
                ));
            }
            (None, false) => {}
        }

        let divergence_registry_hash = self.registry.registry_hash()?;
        let reservation = CampaignRunReservation {
            schema_version: CAMPAIGN_REPORT_SCHEMA_VERSION,
            run_id,
            engines: &engines,
            semantic_contract: &self.semantic_contract,
            config: &self.config,
            corpus_manifest_hash: &corpus_manifest_hash,
            query_manifest_hash: &query_manifest_hash,
            query_source_identity_sha256: &query_suite.manifest.source_identity_sha256,
            divergence_registry_hash: &divergence_registry_hash,
            selected_case_ids: prepared_cases
                .iter()
                .map(|(query, _, _)| query.id.as_str())
                .collect(),
            provenance: self.provenance.as_ref(),
        };
        let reservation_bytes = serde_json::to_vec(&reservation)?;
        self.store
            .reserve_campaign_run(run_id, &reservation_bytes)?;

        let expected_receipt =
            EngineIndexReceipt::for_manifest(corpus_manifest, self.semantic_contract.clone())?;
        let mut index_session = IndexSession {
            subject,
            oracle,
            armed: true,
            subject_begin_attempted: false,
            oracle_begin_attempted: false,
        };
        index_session.subject_begin_attempted = true;
        index_session
            .subject
            .begin_corpus(cx, corpus_manifest, &self.semantic_contract)
            .await?;
        index_session.oracle_begin_attempted = true;
        index_session
            .oracle
            .begin_corpus(cx, corpus_manifest, &self.semantic_contract)
            .await?;
        let batch_size = usize::try_from(self.config.index_batch_size)
            .map_err(|_| campaign_error("index batch size does not fit usize"))?;
        let mut batch = Vec::with_capacity(batch_size);
        let mut batch_bytes = 0_u64;
        let mut ingest_verifier = corpus_manifest.replay_verifier();
        for document in documents.replay() {
            let document_bytes = ingest_verifier.observe(&document)?;
            let would_exceed_bytes = batch_bytes
                .checked_add(document_bytes)
                .is_none_or(|bytes| bytes > self.config.index_batch_max_bytes);
            if !batch.is_empty() && (batch.len() == batch_size || would_exceed_bytes) {
                index_session.subject.index_batch(cx, &batch).await?;
                index_session.oracle.index_batch(cx, &batch).await?;
                batch.clear();
                batch_bytes = 0;
            }
            batch.push(document);
            batch_bytes = batch_bytes
                .checked_add(document_bytes)
                .ok_or_else(|| campaign_error("index batch canonical byte count overflow"))?;
            if batch.len() == batch_size || batch_bytes >= self.config.index_batch_max_bytes {
                index_session.subject.index_batch(cx, &batch).await?;
                index_session.oracle.index_batch(cx, &batch).await?;
                batch.clear();
                batch_bytes = 0;
            }
        }
        if !batch.is_empty() {
            index_session.subject.index_batch(cx, &batch).await?;
            index_session.oracle.index_batch(cx, &batch).await?;
        }
        ingest_verifier.finish(corpus_manifest)?;
        let subject_index = index_session
            .subject
            .commit_corpus(cx, corpus_manifest, &self.semantic_contract)
            .await?;
        let oracle_index = index_session
            .oracle
            .commit_corpus(cx, corpus_manifest, &self.semantic_contract)
            .await?;
        if subject_index != expected_receipt || oracle_index != expected_receipt {
            return Err(campaign_error(
                "an engine indexed a different corpus or semantic contract",
            ));
        }
        validate_engine_state(
            &*index_session.subject,
            &*index_session.oracle,
            &engines,
            &self.semantic_contract,
        )?;
        index_session.disarm();

        let mut cases = Vec::with_capacity(prepared_cases.len());
        let mut mismatches = MismatchCollection::default();
        let mut lexical_mismatches = LexicalMismatchCollection::default();
        let mut lexical_coverage =
            CampaignLexicalCoverageAccumulator::new(self.config.contract_mode);
        for (ordinal, (query, query_class, evidence_case)) in prepared_cases.into_iter().enumerate()
        {
            validate_engine_state(
                &*index_session.subject,
                &*index_session.oracle,
                &engines,
                &self.semantic_contract,
            )?;
            let subject_result = index_session
                .subject
                .observe_generated(cx, query, &evidence_case)
                .await;
            let oracle_result = index_session
                .oracle
                .observe_generated(cx, query, &evidence_case)
                .await;
            let lexical_result = match self.config.contract_mode {
                CampaignContractMode::RankEnvelopeOnly => {
                    Ok(ArtifactLexicalContractEvidence::RankEnvelopeOnly)
                }
                CampaignContractMode::CoreLexicalV3 => {
                    let subject_bundle = observe_core_lexical_bundle(
                        cx,
                        &*index_session.subject,
                        LexicalEngineRole::Subject,
                        &engines.subject,
                        &corpus_manifest_hash,
                        &self.semantic_contract,
                        query_suite.manifest.spec.seed,
                        query,
                    )
                    .await;
                    let oracle_bundle = observe_core_lexical_bundle(
                        cx,
                        &*index_session.oracle,
                        LexicalEngineRole::Oracle,
                        &engines.oracle,
                        &corpus_manifest_hash,
                        &self.semantic_contract,
                        query_suite.manifest.spec.seed,
                        query,
                    )
                    .await;
                    match (subject_bundle, oracle_bundle) {
                        (Ok(subject), Ok(oracle)) => compare_lexical_contracts(subject, oracle)
                            .map(
                                |comparison| ArtifactLexicalContractEvidence::CoreLexicalV3 {
                                    comparison: Box::new(comparison),
                                },
                            ),
                        (Err(subject), Err(oracle)) => Err(campaign_error(format!(
                            "both core lexical observations failed; subject: {subject}; oracle: {oracle}"
                        ))),
                        (Err(subject), Ok(_)) => Err(campaign_error(format!(
                            "subject core lexical observation failed: {subject}"
                        ))),
                        (Ok(_), Err(oracle)) => Err(campaign_error(format!(
                            "oracle core lexical observation failed: {oracle}"
                        ))),
                    }
                }
            };
            validate_engine_state(
                &*index_session.subject,
                &*index_session.oracle,
                &engines,
                &self.semantic_contract,
            )?;
            let result = match (subject_result, oracle_result, lexical_result) {
                (Ok(subject_observation), Ok(oracle_observation), Ok(lexical_contract)) => self
                    .finish_case(
                        run_id,
                        ordinal,
                        query,
                        query_class,
                        &query_manifest_hash,
                        query_suite.manifest.source,
                        &query_suite.manifest.source_identity_sha256,
                        query_suite.manifest.spec.seed,
                        &engines,
                        corpus_manifest.document_count,
                        evidence_case,
                        subject_observation,
                        oracle_observation,
                        &mut mismatches,
                        &mut lexical_mismatches,
                        &mut lexical_coverage,
                        lexical_contract,
                    ),
                (subject_result, oracle_result, lexical_result) => {
                    let (reason, diagnostic) =
                        observation_error_details(&subject_result, &oracle_result, &lexical_result);
                    CampaignCaseResult {
                        case_id: query.id.clone(),
                        query_class,
                        disposition: CampaignDisposition::InfrastructureError,
                        comparison_status: None,
                        rank_class: None,
                        lexical_contract: unavailable_lexical_summary(self.config.contract_mode),
                        artifact_hash: None,
                        registered_divergence: None,
                        first_divergence: None,
                        reason: Some(reason),
                        diagnostic: Some(diagnostic),
                    }
                }
            };
            cases.push(result);
        }

        let confidence = f64::from_bits(self.config.posterior_confidence_bits);
        let query_classes = summarize_query_classes(&cases, confidence);
        let mismatches = mismatches.finish();
        let lexical_mismatches = lexical_mismatches.finish();
        let lexical_coverage = lexical_coverage.finish();
        let passed = cases.iter().all(|result| result.disposition.passes())
            && lexical_coverage_is_admissible(&lexical_coverage);
        let report = CampaignReport {
            schema_version: CAMPAIGN_REPORT_SCHEMA_VERSION,
            run_id: run_id.to_owned(),
            engines,
            semantic_contract: self.semantic_contract.clone(),
            config: self.config.clone(),
            divergence_registry: self.registry.clone(),
            corpus_manifest: corpus_manifest.clone(),
            corpus_manifest_hash,
            query_suite: query_suite.clone(),
            query_manifest_hash,
            subject_index,
            oracle_index,
            submitted_query_count: query_suite.manifest.query_count,
            selected_query_count: u64::try_from(cases.len()).unwrap_or(u64::MAX),
            cases,
            query_classes,
            mismatches,
            lexical_mismatches,
            lexical_coverage,
            passed,
            provenance: self.provenance.clone(),
        };
        self.store.complete_campaign(&report)?;
        Ok(report)
    }

    fn evidence_case(
        &self,
        query: &GeneratedQueryCase,
        query_seed: u64,
        query_suite_source: QuerySuiteSource,
        corpus_manifest_hash: &str,
    ) -> DifferentialCase {
        evidence_case_for(
            &self.config,
            query,
            query_seed,
            query_suite_source,
            corpus_manifest_hash,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_case(
        &self,
        campaign_run_id: &str,
        ordinal: usize,
        query: &GeneratedQueryCase,
        query_class: String,
        query_manifest_hash: &str,
        query_suite_source: QuerySuiteSource,
        query_source_identity_sha256: &str,
        query_seed: u64,
        engines: &EnginePairIdentity,
        expected_doc_count: u64,
        evidence_case: DifferentialCase,
        subject: EngineObservation,
        oracle: EngineObservation,
        mismatches: &mut MismatchCollection,
        lexical_mismatches: &mut LexicalMismatchCollection,
        lexical_coverage: &mut CampaignLexicalCoverageAccumulator,
        lexical_contract: ArtifactLexicalContractEvidence,
    ) -> CampaignCaseResult {
        if subject.doc_count != expected_doc_count || oracle.doc_count != expected_doc_count {
            return infrastructure_case(
                query,
                query_class,
                self.config.contract_mode,
                "observation_document_count_drift",
                format!(
                    "expected {expected_doc_count}; subject {}; oracle {}",
                    subject.doc_count, oracle.doc_count
                ),
            );
        }
        let comparison = match evidence_case
            .validate_observations(engines, &subject, &oracle)
            .and_then(|()| compare_observations(subject, oracle, self.config.comparator_config))
        {
            Ok(comparison) => comparison,
            Err(error) => {
                return infrastructure_case(
                    query,
                    query_class,
                    self.config.contract_mode,
                    "comparison_failed",
                    error.to_string(),
                );
            }
        };
        let mismatch_text_bytes = match mismatches.preflight(&comparison, &query.id) {
            Ok(text_bytes) => text_bytes,
            Err(error) => {
                return infrastructure_case(
                    query,
                    query_class,
                    self.config.contract_mode,
                    "mismatch_budget_exceeded",
                    error.to_string(),
                );
            }
        };
        let lexical_mismatch_text_bytes =
            match lexical_mismatches.preflight(&lexical_contract, &query.id) {
                Ok(text_bytes) => text_bytes,
                Err(error) => {
                    return infrastructure_case(
                        query,
                        query_class,
                        self.config.contract_mode,
                        "lexical_mismatch_budget_exceeded",
                        error.to_string(),
                    );
                }
            };
        let (disposition, reason, registered_divergence) =
            classify_case_with_lexical(query, &comparison, &lexical_contract, &self.registry);
        let first_divergence = comparison.first_divergence.clone();
        let comparison_status = Some(comparison.status);
        let rank_class = Some(comparison.rank_class);
        let lexical_summary = match lexical_case_summary(&lexical_contract) {
            Ok(summary) => summary,
            Err(error) => {
                return infrastructure_case(
                    query,
                    query_class,
                    self.config.contract_mode,
                    "lexical_summary_failed",
                    error.to_string(),
                );
            }
        };
        let run = HarnessRun {
            engines: engines.clone(),
            case: evidence_case,
            comparator_config: self.config.comparator_config,
            comparison,
        };
        let context = CampaignArtifactContext {
            corpus_manifest_hash: run.case.metadata.corpus_hash.clone().unwrap_or_default(),
            query_manifest_hash: query_manifest_hash.to_owned(),
            query_suite_source,
            query_source_identity_sha256: query_source_identity_sha256.to_owned(),
            semantic_contract: self.semantic_contract.clone(),
            contract_mode: self.config.contract_mode,
            query_seed,
            query: query.clone(),
            registered_divergence: registered_divergence.clone(),
        };
        let object = match ArtifactObject::from_campaign_run(run, context, lexical_contract) {
            Ok(object) => object,
            Err(error) => {
                return infrastructure_case(
                    query,
                    query_class,
                    self.config.contract_mode,
                    "artifact_validation_failed",
                    error.to_string(),
                );
            }
        };
        let provenance = BTreeMap::from([
            ("campaign_run_id".to_owned(), campaign_run_id.to_owned()),
            ("query_class".to_owned(), query_class.clone()),
            ("query_source".to_owned(), query.source.clone()),
        ]);
        let prepared =
            match self
                .store
                .prepare_campaign_case(campaign_run_id, ordinal, &object, provenance)
            {
                Ok(prepared) => prepared,
                Err(error) => {
                    return infrastructure_case(
                        query,
                        query_class,
                        self.config.contract_mode,
                        "artifact_prepare_failed",
                        error.to_string(),
                    );
                }
            };
        if let Err(error) = self.store.persist(&prepared) {
            return infrastructure_case(
                query,
                query_class,
                self.config.contract_mode,
                "artifact_persist_failed",
                error.to_string(),
            );
        }
        mismatches.apply(&object.comparison, &query.id, mismatch_text_bytes);
        lexical_mismatches.apply(
            &object.lexical_contract,
            &query.id,
            lexical_mismatch_text_bytes,
        );
        lexical_coverage.record(&object.lexical_contract);
        CampaignCaseResult {
            case_id: query.id.clone(),
            query_class,
            disposition,
            comparison_status,
            rank_class,
            lexical_contract: lexical_summary,
            artifact_hash: Some(prepared.object_hash().to_owned()),
            registered_divergence,
            first_divergence,
            reason,
            diagnostic: None,
        }
    }
}

fn evidence_case_for(
    config: &CampaignConfig,
    query: &GeneratedQueryCase,
    query_seed: u64,
    query_suite_source: QuerySuiteSource,
    corpus_manifest_hash: &str,
) -> DifferentialCase {
    let mut case = DifferentialCase::new(&query.id, &query.query, query.limit);
    case.offset = query.offset;
    case.tie_expansion_limit = config.tie_expansion_limit;
    case.count_requested = query.count_requested;
    case.snippet_max_chars = config.snippet_max_chars;
    case.metadata = DifferentialCaseMetadata {
        generator_id: (query_suite_source == QuerySuiteSource::Generated)
            .then(|| GENERATOR_ID.to_owned()),
        generator_seed: (query_suite_source == QuerySuiteSource::Generated).then_some(query_seed),
        corpus_hash: Some(corpus_manifest_hash.to_owned()),
    };
    case
}

fn infrastructure_case(
    query: &GeneratedQueryCase,
    query_class: String,
    contract_mode: CampaignContractMode,
    reason: &'static str,
    diagnostic: String,
) -> CampaignCaseResult {
    CampaignCaseResult {
        case_id: query.id.clone(),
        query_class,
        disposition: CampaignDisposition::InfrastructureError,
        comparison_status: None,
        rank_class: None,
        lexical_contract: unavailable_lexical_summary(contract_mode),
        artifact_hash: None,
        registered_divergence: None,
        first_divergence: None,
        reason: Some(reason.to_owned()),
        diagnostic: Some(diagnostic),
    }
}

fn unavailable_lexical_summary(contract_mode: CampaignContractMode) -> CampaignLexicalCaseSummary {
    match contract_mode {
        CampaignContractMode::RankEnvelopeOnly => CampaignLexicalCaseSummary::RankEnvelopeOnly,
        CampaignContractMode::CoreLexicalV3 => CampaignLexicalCaseSummary::CoreLexicalV3Unavailable,
    }
}

fn lexical_case_summary(
    evidence: &ArtifactLexicalContractEvidence,
) -> Result<CampaignLexicalCaseSummary, GauntletError> {
    match evidence {
        ArtifactLexicalContractEvidence::LegacyPreV3Missing => Err(campaign_error(
            "legacy lexical evidence cannot produce a current case summary",
        )),
        ArtifactLexicalContractEvidence::RankEnvelopeOnly => {
            Ok(CampaignLexicalCaseSummary::RankEnvelopeOnly)
        }
        ArtifactLexicalContractEvidence::CoreLexicalV3 { comparison } => {
            comparison.validate_replay()?;
            Ok(CampaignLexicalCaseSummary::CoreLexicalV3 {
                status: comparison.status,
                first_mismatch: comparison.first_mismatch.clone(),
                mismatch_count: u64::try_from(comparison.mismatches.len())
                    .map_err(|_| campaign_error("lexical mismatch count does not fit u64"))?,
                waived_difference_count: u64::try_from(comparison.waived_differences.len())
                    .map_err(|_| {
                        campaign_error("lexical waived-difference count does not fit u64")
                    })?,
            })
        }
    }
}

fn classify_case(
    query: &GeneratedQueryCase,
    comparison: &ComparisonReport,
    registry: &DivergenceRegistry,
) -> (
    CampaignDisposition,
    Option<String>,
    Option<DivergenceRegisterEntry>,
) {
    if comparison.status == ComparisonStatus::Failed {
        return (
            CampaignDisposition::Unclassified,
            Some("comparator reported an unclassified result-level failure".to_owned()),
            None,
        );
    }
    let has_register_divergence = comparison
        .divergences
        .iter()
        .any(|divergence| !is_auto_class(divergence.class));
    match query.expected_divergence.as_deref() {
        None if !has_register_divergence => {
            if comparison.status == ComparisonStatus::Exact {
                (CampaignDisposition::Exact, None, None)
            } else {
                (CampaignDisposition::AutoClassified, None, None)
            }
        }
        None => (
            CampaignDisposition::Unclassified,
            Some("classified divergence has no reviewed register entry".to_owned()),
            None,
        ),
        Some(expected_id) => {
            let Some(entry) = registry.find(expected_id) else {
                return (
                    CampaignDisposition::Unclassified,
                    Some(format!(
                        "expected divergence {expected_id} is not registered"
                    )),
                    None,
                );
            };
            let matches = entry.matches_comparison(query, comparison);
            if matches {
                (
                    CampaignDisposition::RegisterClassified,
                    None,
                    Some(entry.clone()),
                )
            } else {
                (
                    CampaignDisposition::Unclassified,
                    Some(format!(
                        "expected divergence {expected_id} did not match this fixture and comparator class"
                    )),
                    None,
                )
            }
        }
    }
}

fn classify_case_with_lexical(
    query: &GeneratedQueryCase,
    comparison: &ComparisonReport,
    lexical_contract: &ArtifactLexicalContractEvidence,
    registry: &DivergenceRegistry,
) -> (
    CampaignDisposition,
    Option<String>,
    Option<DivergenceRegisterEntry>,
) {
    if let ArtifactLexicalContractEvidence::CoreLexicalV3 {
        comparison: lexical,
    } = lexical_contract
        && lexical.status == LexicalComparisonStatus::Mismatch
    {
        return (
            CampaignDisposition::Unclassified,
            Some("lexical_contract_mismatch".to_owned()),
            None,
        );
    }
    classify_case(query, comparison, registry)
}

fn is_auto_class(class: DivergenceClass) -> bool {
    match class {
        DivergenceClass::TieOrder | DivergenceClass::ScoreEpsilon => true,
        DivergenceClass::RankMismatch
        | DivergenceClass::SnippetMismatch
        | DivergenceClass::SnippetWindow
        | DivergenceClass::CountMismatch
        | DivergenceClass::DocumentCountMismatch
        | DivergenceClass::GlobExpansionLimit
        | DivergenceClass::QueryCanonicalization
        | DivergenceClass::OracleBug
        | DivergenceClass::StatsSemantics
        | DivergenceClass::PostingRecordSemantics
        | DivergenceClass::UnicodeEdge
        | DivergenceClass::OversizedQueryToken => false,
    }
}

fn query_class(query: &GeneratedQueryCase) -> String {
    let syntax = match query.syntax {
        QuerySyntax::Default => "default",
        QuerySyntax::Cass => "cass",
    };
    let kind = match &query.query_kind {
        GeneratedQueryKind::Term => "term".to_owned(),
        GeneratedQueryKind::MultiTerm => "multi_term".to_owned(),
        GeneratedQueryKind::Phrase => "phrase".to_owned(),
        GeneratedQueryKind::Boolean => "boolean".to_owned(),
        GeneratedQueryKind::Glob { pattern_class } => format!(
            "glob_{}",
            match pattern_class {
                GlobPatternClass::Exact => "exact",
                GlobPatternClass::Prefix => "prefix",
                GlobPatternClass::Suffix => "suffix",
                GlobPatternClass::Substring => "substring",
                GlobPatternClass::Complex => "complex",
            }
        ),
        GeneratedQueryKind::Range { range_class } => format!(
            "range_{}",
            match range_class {
                RangeClass::Inclusive => "inclusive",
                RangeClass::From => "from",
                RangeClass::To => "to",
            }
        ),
        GeneratedQueryKind::StructuredFilter { filter_class } => format!(
            "filter_{}",
            match filter_class {
                StructuredFilterClass::Agent => "agent",
                StructuredFilterClass::Workspace => "workspace",
                StructuredFilterClass::Local => "local",
                StructuredFilterClass::Remote => "remote",
                StructuredFilterClass::SourceId => "source_id",
                StructuredFilterClass::Combined => "combined",
            }
        ),
        GeneratedQueryKind::Paginated => "paginated".to_owned(),
        GeneratedQueryKind::Counted => "counted".to_owned(),
        GeneratedQueryKind::Harvested { semantic_class } => {
            format!("harvested_{semantic_class}")
        }
    };
    format!("{syntax}.{kind}")
}

#[derive(Default)]
struct SummaryAccumulator {
    total: u64,
    exact: u64,
    auto_classified: u64,
    register_classified: u64,
    unclassified: u64,
    infrastructure_errors: u64,
}

fn summarize_query_classes(
    cases: &[CampaignCaseResult],
    confidence: f64,
) -> Vec<QueryClassSummary> {
    let mut summaries = BTreeMap::<String, SummaryAccumulator>::new();
    for case in cases {
        let summary = summaries.entry(case.query_class.clone()).or_default();
        summary.total = summary.total.saturating_add(1);
        match case.disposition {
            CampaignDisposition::Exact => summary.exact = summary.exact.saturating_add(1),
            CampaignDisposition::AutoClassified => {
                summary.auto_classified = summary.auto_classified.saturating_add(1);
            }
            CampaignDisposition::RegisterClassified => {
                summary.register_classified = summary.register_classified.saturating_add(1);
            }
            CampaignDisposition::Unclassified => {
                summary.unclassified = summary.unclassified.saturating_add(1);
            }
            CampaignDisposition::InfrastructureError => {
                summary.infrastructure_errors = summary.infrastructure_errors.saturating_add(1);
            }
        }
    }
    summaries
        .into_iter()
        .map(|(query_class, summary)| QueryClassSummary {
            query_class,
            total: summary.total,
            exact: summary.exact,
            auto_classified: summary.auto_classified,
            register_classified: summary.register_classified,
            unclassified: summary.unclassified,
            infrastructure_errors: summary.infrastructure_errors,
            posterior_confidence_bits: confidence.to_bits(),
        })
        .collect()
}

#[derive(Debug)]
struct MismatchAccumulator {
    signature: String,
    divergence: Divergence,
    occurrence_count: u64,
    case_ids: BTreeSet<String>,
}

impl MismatchAccumulator {
    fn finish(self) -> MismatchGroup {
        MismatchGroup {
            signature: self.signature,
            divergence: self.divergence,
            occurrence_count: self.occurrence_count,
            case_ids: self.case_ids.into_iter().collect(),
        }
    }
}

#[derive(Debug, Default)]
struct MismatchCollection {
    entries: BTreeMap<String, MismatchAccumulator>,
    text_bytes: usize,
}

impl MismatchCollection {
    fn preflight(
        &self,
        comparison: &ComparisonReport,
        case_id: &str,
    ) -> Result<usize, GauntletError> {
        let mut new_groups = BTreeMap::<String, Divergence>::new();
        let mut case_id_additions = BTreeSet::<String>::new();
        for divergence in &comparison.divergences {
            if !divergence.pointer.starts_with('/')
                || divergence.pointer.len() > MAX_CAMPAIGN_POINTER_BYTES
                || divergence.oracle.len() > MAX_CAMPAIGN_POINTER_BYTES
                || divergence.subject.len() > MAX_CAMPAIGN_POINTER_BYTES
            {
                return Err(campaign_error(
                    "comparison divergence exceeds the campaign mismatch budget",
                ));
            }
            let signature = mismatch_signature(comparison.rank_class, divergence);
            if !self.entries.contains_key(&signature) {
                new_groups
                    .entry(signature.clone())
                    .or_insert_with(|| divergence.clone());
            }
            let already_has_case = self
                .entries
                .get(&signature)
                .is_some_and(|entry| entry.case_ids.contains(case_id));
            if !already_has_case {
                case_id_additions.insert(signature);
            }
        }

        let new_text_bytes = new_groups
            .iter()
            .try_fold(0_usize, |bytes, (signature, divergence)| {
                bytes
                    .checked_add(signature.len())
                    .and_then(|sum| sum.checked_add(divergence.pointer.len()))
                    .and_then(|sum| sum.checked_add(divergence.oracle.len()))
                    .and_then(|sum| sum.checked_add(divergence.subject.len()))
            })
            .and_then(|bytes| {
                case_id
                    .len()
                    .checked_mul(case_id_additions.len())
                    .and_then(|case_bytes| bytes.checked_add(case_bytes))
            })
            .ok_or_else(|| campaign_error("campaign mismatch text byte count overflow"))?;
        let final_group_count = self
            .entries
            .len()
            .checked_add(new_groups.len())
            .ok_or_else(|| campaign_error("campaign mismatch group count overflow"))?;
        let final_text_bytes = self
            .text_bytes
            .checked_add(new_text_bytes)
            .ok_or_else(|| campaign_error("campaign mismatch text byte count overflow"))?;
        if final_group_count > MAX_MISMATCH_GROUPS || final_text_bytes > MAX_MISMATCH_TEXT_BYTES {
            return Err(campaign_error(
                "campaign mismatch groups exceed their count or text budget",
            ));
        }

        Ok(final_text_bytes)
    }

    fn apply(&mut self, comparison: &ComparisonReport, case_id: &str, final_text_bytes: usize) {
        for divergence in &comparison.divergences {
            let signature = mismatch_signature(comparison.rank_class, divergence);
            let entry =
                self.entries
                    .entry(signature.clone())
                    .or_insert_with(|| MismatchAccumulator {
                        signature,
                        divergence: divergence.clone(),
                        occurrence_count: 0,
                        case_ids: BTreeSet::new(),
                    });
            entry.occurrence_count = entry.occurrence_count.saturating_add(1);
            entry.case_ids.insert(case_id.to_owned());
        }
        self.text_bytes = final_text_bytes;
    }

    fn record(
        &mut self,
        comparison: &ComparisonReport,
        case_id: &str,
    ) -> Result<(), GauntletError> {
        let final_text_bytes = self.preflight(comparison, case_id)?;
        self.apply(comparison, case_id, final_text_bytes);
        Ok(())
    }

    fn finish(self) -> Vec<MismatchGroup> {
        self.entries
            .into_values()
            .map(MismatchAccumulator::finish)
            .collect()
    }
}

#[derive(Debug)]
struct LexicalMismatchAccumulator {
    signature: String,
    mismatch: LexicalFieldMismatch,
    occurrence_count: u64,
    case_ids: BTreeSet<String>,
}

impl LexicalMismatchAccumulator {
    fn finish(self) -> LexicalMismatchGroup {
        LexicalMismatchGroup {
            signature: self.signature,
            mismatch: self.mismatch,
            occurrence_count: self.occurrence_count,
            case_ids: self.case_ids.into_iter().collect(),
        }
    }
}

#[derive(Debug, Default)]
struct LexicalMismatchCollection {
    entries: BTreeMap<String, LexicalMismatchAccumulator>,
    text_bytes: usize,
}

impl LexicalMismatchCollection {
    fn preflight(
        &self,
        evidence: &ArtifactLexicalContractEvidence,
        case_id: &str,
    ) -> Result<usize, GauntletError> {
        let mismatches = match evidence {
            ArtifactLexicalContractEvidence::LegacyPreV3Missing
            | ArtifactLexicalContractEvidence::RankEnvelopeOnly => return Ok(self.text_bytes),
            ArtifactLexicalContractEvidence::CoreLexicalV3 { comparison } => {
                comparison.validate_replay()?;
                &comparison.mismatches
            }
        };
        let mut new_groups = BTreeMap::<String, LexicalFieldMismatch>::new();
        let mut case_id_additions = BTreeSet::<String>::new();
        for mismatch in mismatches {
            if !mismatch.path.starts_with('/')
                || mismatch.path.len() > MAX_CAMPAIGN_POINTER_BYTES
                || mismatch.oracle.len() > MAX_CAMPAIGN_POINTER_BYTES
                || mismatch.subject.len() > MAX_CAMPAIGN_POINTER_BYTES
            {
                return Err(campaign_error(
                    "lexical mismatch exceeds the campaign mismatch budget",
                ));
            }
            let signature = lexical_mismatch_signature(mismatch);
            if !self.entries.contains_key(&signature) {
                new_groups
                    .entry(signature.clone())
                    .or_insert_with(|| mismatch.clone());
            }
            if !self
                .entries
                .get(&signature)
                .is_some_and(|entry| entry.case_ids.contains(case_id))
            {
                case_id_additions.insert(signature);
            }
        }
        let new_text_bytes = new_groups
            .iter()
            .try_fold(0_usize, |bytes, (signature, mismatch)| {
                bytes
                    .checked_add(signature.len())
                    .and_then(|sum| sum.checked_add(mismatch.path.len()))
                    .and_then(|sum| sum.checked_add(mismatch.oracle.len()))
                    .and_then(|sum| sum.checked_add(mismatch.subject.len()))
            })
            .and_then(|bytes| {
                case_id
                    .len()
                    .checked_mul(case_id_additions.len())
                    .and_then(|case_bytes| bytes.checked_add(case_bytes))
            })
            .ok_or_else(|| campaign_error("lexical mismatch text byte count overflow"))?;
        let final_group_count = self
            .entries
            .len()
            .checked_add(new_groups.len())
            .ok_or_else(|| campaign_error("lexical mismatch group count overflow"))?;
        let final_text_bytes = self
            .text_bytes
            .checked_add(new_text_bytes)
            .ok_or_else(|| campaign_error("lexical mismatch text byte count overflow"))?;
        if final_group_count > MAX_MISMATCH_GROUPS || final_text_bytes > MAX_MISMATCH_TEXT_BYTES {
            return Err(campaign_error(
                "lexical mismatch groups exceed their count or text budget",
            ));
        }
        Ok(final_text_bytes)
    }

    fn apply(
        &mut self,
        evidence: &ArtifactLexicalContractEvidence,
        case_id: &str,
        final_text_bytes: usize,
    ) {
        if let ArtifactLexicalContractEvidence::CoreLexicalV3 { comparison } = evidence {
            for mismatch in &comparison.mismatches {
                let signature = lexical_mismatch_signature(mismatch);
                let entry = self.entries.entry(signature.clone()).or_insert_with(|| {
                    LexicalMismatchAccumulator {
                        signature,
                        mismatch: mismatch.clone(),
                        occurrence_count: 0,
                        case_ids: BTreeSet::new(),
                    }
                });
                entry.occurrence_count = entry.occurrence_count.saturating_add(1);
                entry.case_ids.insert(case_id.to_owned());
            }
        }
        self.text_bytes = final_text_bytes;
    }

    fn record(
        &mut self,
        evidence: &ArtifactLexicalContractEvidence,
        case_id: &str,
    ) -> Result<(), GauntletError> {
        let final_text_bytes = self.preflight(evidence, case_id)?;
        self.apply(evidence, case_id, final_text_bytes);
        Ok(())
    }

    fn finish(self) -> Vec<LexicalMismatchGroup> {
        self.entries
            .into_values()
            .map(LexicalMismatchAccumulator::finish)
            .collect()
    }
}

#[derive(Debug)]
struct CampaignLexicalCoverageAccumulator {
    mode: CampaignContractMode,
    subject: LexicalSideCoverageCounts,
    oracle: LexicalSideCoverageCounts,
}

impl CampaignLexicalCoverageAccumulator {
    fn new(mode: CampaignContractMode) -> Self {
        Self {
            mode,
            subject: LexicalSideCoverageCounts::default(),
            oracle: LexicalSideCoverageCounts::default(),
        }
    }

    fn record(&mut self, evidence: &ArtifactLexicalContractEvidence) {
        if let ArtifactLexicalContractEvidence::CoreLexicalV3 { comparison } = evidence {
            record_side_coverage(&comparison.coverage.subject, &mut self.subject);
            record_side_coverage(&comparison.coverage.oracle, &mut self.oracle);
            if comparison.subject.fusion_metadata_is_deferred() {
                self.subject.metadata_deferred_cases =
                    self.subject.metadata_deferred_cases.saturating_add(1);
            }
            if comparison.oracle.fusion_metadata_is_deferred() {
                self.oracle.metadata_deferred_cases =
                    self.oracle.metadata_deferred_cases.saturating_add(1);
            }
        }
    }

    fn finish(self) -> CampaignLexicalCoverageSummary {
        match self.mode {
            CampaignContractMode::RankEnvelopeOnly => {
                CampaignLexicalCoverageSummary::RankEnvelopeOnly
            }
            CampaignContractMode::CoreLexicalV3 => {
                let admissible = lexical_side_coverage_is_admissible(&self.subject)
                    && lexical_side_coverage_is_admissible(&self.oracle);
                CampaignLexicalCoverageSummary::CoreLexicalV3 {
                    subject: Box::new(self.subject),
                    oracle: Box::new(self.oracle),
                    admissible,
                }
            }
        }
    }
}

fn record_side_coverage(observed: &LexicalSideCoverage, aggregate: &mut LexicalSideCoverageCounts) {
    record_probe_coverage(&observed.full_search, &mut aggregate.full_search);
    record_probe_coverage(
        &observed.fusion_candidates,
        &mut aggregate.fusion_candidates,
    );
    record_probe_coverage(
        &observed.all_lexical_winners_hydration,
        &mut aggregate.all_lexical_winners_hydration,
    );
    record_probe_coverage(
        &observed.strict_hybrid_winners_hydration,
        &mut aggregate.strict_hybrid_winners_hydration,
    );
    record_probe_coverage(
        &observed.semantic_only_hydration,
        &mut aggregate.semantic_only_hydration,
    );
    record_probe_coverage(
        &observed.mixed_winners_hydration,
        &mut aggregate.mixed_winners_hydration,
    );
}

fn record_probe_coverage(observed: &LexicalProbeCoverage, aggregate: &mut ProbeCoverageCounts) {
    match observed {
        LexicalProbeCoverage::ExercisedSuccess => {
            aggregate.success = aggregate.success.saturating_add(1);
        }
        LexicalProbeCoverage::ExercisedRestoration => {
            aggregate.restoration = aggregate.restoration.saturating_add(1);
        }
        LexicalProbeCoverage::ExercisedError => {
            aggregate.error = aggregate.error.saturating_add(1);
        }
        LexicalProbeCoverage::ExercisedEmpty => {
            aggregate.empty = aggregate.empty.saturating_add(1);
        }
        LexicalProbeCoverage::NotRun { .. } => {
            aggregate.not_run = aggregate.not_run.saturating_add(1);
        }
    }
}

fn lexical_side_coverage_is_admissible(side: &LexicalSideCoverageCounts) -> bool {
    let hydration_success = [
        &side.all_lexical_winners_hydration,
        &side.strict_hybrid_winners_hydration,
        &side.semantic_only_hydration,
        &side.mixed_winners_hydration,
    ]
    .into_iter()
    .all(|probe| probe.success.saturating_add(probe.restoration) > 0);
    let deferred_metadata_shapes_are_exercised = side.metadata_deferred_cases == 0
        || [
            &side.all_lexical_winners_hydration,
            &side.strict_hybrid_winners_hydration,
            &side.mixed_winners_hydration,
        ]
        .into_iter()
        .all(|probe| probe.restoration > 0);
    side.full_search.success > 0
        && side.full_search.empty > 0
        && side.fusion_candidates.success > 0
        && side.fusion_candidates.empty > 0
        && hydration_success
        && deferred_metadata_shapes_are_exercised
}

fn lexical_coverage_is_admissible(coverage: &CampaignLexicalCoverageSummary) -> bool {
    match coverage {
        CampaignLexicalCoverageSummary::LegacyMissing => false,
        CampaignLexicalCoverageSummary::RankEnvelopeOnly => true,
        CampaignLexicalCoverageSummary::CoreLexicalV3 { admissible, .. } => *admissible,
    }
}

fn lexical_mismatch_signature(mismatch: &LexicalFieldMismatch) -> String {
    let pointer = normalized_pointer(&mismatch.path);
    let cause = format!(
        "{}:{}",
        normalized_diagnostic_shape(&mismatch.oracle),
        normalized_diagnostic_shape(&mismatch.subject)
    );
    let mut hasher = Sha256::new();
    hasher.update(LEXICAL_MISMATCH_SIGNATURE_DOMAIN);
    hasher.update([lexical_mismatch_class_tag(mismatch.class)]);
    hasher.update(
        u64::try_from(pointer.len())
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
    hasher.update(pointer.as_bytes());
    hasher.update(u64::try_from(cause.len()).unwrap_or(u64::MAX).to_le_bytes());
    hasher.update(cause.as_bytes());
    lower_hex(&hasher.finalize())
}

const fn lexical_mismatch_class_tag(class: LexicalMismatchClass) -> u8 {
    match class {
        LexicalMismatchClass::Context => 0,
        LexicalMismatchClass::Outcome => 1,
        LexicalMismatchClass::Ordering => 2,
        LexicalMismatchClass::Score => 3,
        LexicalMismatchClass::SourceIdentity => 4,
        LexicalMismatchClass::Snippet => 5,
        LexicalMismatchClass::Highlight => 6,
        LexicalMismatchClass::Metadata => 7,
        LexicalMismatchClass::Explanation => 8,
        LexicalMismatchClass::Count => 9,
        LexicalMismatchClass::Error => 10,
    }
}

fn mismatch_signature(rank_class: RankClass, divergence: &Divergence) -> String {
    let pointer = normalized_pointer(&divergence.pointer);
    let cause = mismatch_cause_shape(divergence);
    let mut hasher = Sha256::new();
    hasher.update(MISMATCH_SIGNATURE_DOMAIN);
    hasher.update([
        rank_class_tag(rank_class),
        divergence_class_tag(divergence.class),
    ]);
    hasher.update(
        u64::try_from(pointer.len())
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
    hasher.update(pointer.as_bytes());
    hasher.update(u64::try_from(cause.len()).unwrap_or(u64::MAX).to_le_bytes());
    hasher.update(cause.as_bytes());
    lower_hex(&hasher.finalize())
}

fn mismatch_cause_shape(divergence: &Divergence) -> String {
    fn rank_value(value: &str) -> &'static str {
        if value.rsplit_once('@').is_some() {
            "hit"
        } else {
            "length"
        }
    }

    fn presence(value: &str) -> &'static str {
        if value == "<missing>" {
            "missing"
        } else {
            "present"
        }
    }

    fn count(value: &str) -> &'static str {
        if value == "not_requested" {
            "not_requested"
        } else {
            "value"
        }
    }

    match divergence.class {
        DivergenceClass::TieOrder
        | DivergenceClass::ScoreEpsilon
        | DivergenceClass::RankMismatch
        | DivergenceClass::PostingRecordSemantics => format!(
            "rank:{}:{}",
            rank_value(&divergence.oracle),
            rank_value(&divergence.subject)
        ),
        DivergenceClass::SnippetMismatch | DivergenceClass::SnippetWindow => format!(
            "snippet:{}:{}",
            presence(&divergence.oracle),
            presence(&divergence.subject)
        ),
        DivergenceClass::CountMismatch => format!(
            "count:{}:{}",
            count(&divergence.oracle),
            count(&divergence.subject)
        ),
        DivergenceClass::DocumentCountMismatch => "document_count:value:value".to_owned(),
        DivergenceClass::GlobExpansionLimit
        | DivergenceClass::QueryCanonicalization
        | DivergenceClass::OracleBug
        | DivergenceClass::StatsSemantics
        | DivergenceClass::UnicodeEdge
        | DivergenceClass::OversizedQueryToken => format!(
            "ast:{}:{}",
            normalized_diagnostic_shape(&divergence.oracle),
            normalized_diagnostic_shape(&divergence.subject)
        ),
    }
}

fn normalized_diagnostic_shape(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_digit() {
                '#'
            } else {
                character
            }
        })
        .collect()
}

const fn rank_class_tag(class: RankClass) -> u8 {
    match class {
        RankClass::RankExact => 0,
        RankClass::TieOrder => 1,
        RankClass::ScoreEpsilon => 2,
        RankClass::RankMismatch => 3,
    }
}

const fn divergence_class_tag(class: DivergenceClass) -> u8 {
    match class {
        DivergenceClass::TieOrder => 0,
        DivergenceClass::ScoreEpsilon => 1,
        DivergenceClass::RankMismatch => 2,
        DivergenceClass::SnippetMismatch => 3,
        DivergenceClass::CountMismatch => 4,
        DivergenceClass::DocumentCountMismatch => 5,
        DivergenceClass::OversizedQueryToken => 6,
        DivergenceClass::SnippetWindow => 7,
        DivergenceClass::GlobExpansionLimit => 8,
        DivergenceClass::QueryCanonicalization => 9,
        DivergenceClass::OracleBug => 10,
        DivergenceClass::StatsSemantics => 11,
        DivergenceClass::UnicodeEdge => 12,
        DivergenceClass::PostingRecordSemantics => 13,
    }
}

fn normalized_pointer(pointer: &str) -> String {
    pointer
        .split('/')
        .map(|component| {
            if !component.is_empty() && component.bytes().all(|byte| byte.is_ascii_digit()) {
                "*"
            } else {
                component
            }
        })
        .collect::<Vec<_>>()
        .join("/")
}

#[allow(clippy::too_many_arguments)]
async fn observe_core_lexical_bundle(
    cx: &Cx,
    engine: &dyn DifferentialCampaignEngine,
    role: LexicalEngineRole,
    descriptor: &EngineDescriptor,
    corpus_manifest_hash: &str,
    semantic_contract: &SemanticContract,
    query_seed: u64,
    query: &GeneratedQueryCase,
) -> Result<crate::comparator::LexicalContractBundle, GauntletError> {
    if query.syntax != QuerySyntax::Default {
        return Err(campaign_error(
            "core lexical contract observation requires default query syntax",
        ));
    }
    if engine.descriptor() != *descriptor {
        return Err(campaign_error(
            "engine descriptor changed before total lexical observation",
        ));
    }
    let limit = usize::try_from(query.limit)
        .map_err(|_| campaign_error("core lexical query limit does not fit usize"))?;
    let backend = lexical_backend_identity(descriptor, corpus_manifest_hash)?;
    let query_contract_sha256 = lexical_query_contract_sha256(semantic_contract)?;
    let build = LexicalContractBuildContext::new(
        role,
        backend,
        corpus_manifest_hash.to_owned(),
        corpus_manifest_hash.to_owned(),
        query_contract_sha256,
        &query.query,
        query_seed,
        limit,
    )?;
    let lexical = engine.core_lexical_search()?.ok_or_else(|| {
        campaign_error(format!(
            "engine {} does not expose the required ordinary LexicalSearch contract",
            descriptor.implementation
        ))
    })?;
    observe_live_lexical_contract(cx, lexical, build).await
}

fn validate_engine_state(
    subject: &dyn DifferentialCampaignEngine,
    oracle: &dyn DifferentialCampaignEngine,
    expected_engines: &EnginePairIdentity,
    expected_semantics: &SemanticContract,
) -> Result<(), GauntletError> {
    let mut observed = EnginePairIdentity::new(
        ComparisonMode::CrossEngine,
        subject.descriptor(),
        oracle.descriptor(),
    )?;
    observed.bind_semantic_contract(expected_semantics.clone())?;
    if &observed != expected_engines
        || subject.semantic_contract() != *expected_semantics
        || oracle.semantic_contract() != *expected_semantics
    {
        return Err(campaign_error(
            "engine identity or semantic contract changed during campaign execution",
        ));
    }
    Ok(())
}

fn observation_error_details(
    subject: &Result<EngineObservation, GauntletError>,
    oracle: &Result<EngineObservation, GauntletError>,
    lexical: &Result<ArtifactLexicalContractEvidence, GauntletError>,
) -> (String, String) {
    match (subject, oracle, lexical) {
        (Ok(_), Ok(_), Err(lexical)) => (
            "lexical_contract_observation_failed".to_owned(),
            lexical.to_string(),
        ),
        (Err(subject), Err(oracle), Ok(_)) => (
            "both_engine_executions_failed".to_owned(),
            format!("subject: {subject}; oracle: {oracle}"),
        ),
        (Err(subject), Ok(_), Ok(_)) => {
            ("subject_execution_failed".to_owned(), subject.to_string())
        }
        (Ok(_), Err(oracle), Ok(_)) => ("oracle_execution_failed".to_owned(), oracle.to_string()),
        (Err(subject), oracle, Err(lexical)) => (
            "multiple_observation_lanes_failed".to_owned(),
            format!(
                "subject legacy: {subject}; oracle legacy: {}; lexical: {lexical}",
                oracle
                    .as_ref()
                    .err()
                    .map_or_else(|| "ok".to_owned(), ToString::to_string)
            ),
        ),
        (Ok(_), Err(oracle), Err(lexical)) => (
            "multiple_observation_lanes_failed".to_owned(),
            format!("oracle legacy: {oracle}; lexical: {lexical}"),
        ),
        (Ok(_), Ok(_), Ok(_)) => (
            "invalid_engine_error_state".to_owned(),
            "all observation results unexpectedly succeeded".to_owned(),
        ),
    }
}

fn validate_campaign_run_id(run_id: &str) -> Result<(), GauntletError> {
    let safe = !run_id.is_empty()
        && run_id.len() <= 112
        && run_id != "."
        && run_id != ".."
        && run_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'));
    if safe {
        Ok(())
    } else {
        Err(GauntletError::InvalidRunId {
            run_id: run_id.to_owned(),
        })
    }
}

fn is_register_id(value: &str) -> bool {
    value.len() == 7
        && value.starts_with("DIV-")
        && value[4..].bytes().all(|byte| byte.is_ascii_digit())
}

fn is_prediction_id(value: &str) -> bool {
    value.len() == 8
        && value.starts_with("PRED-")
        && value[5..].bytes().all(|byte| byte.is_ascii_digit())
}

fn is_register_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 112
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_')
        })
}

fn is_bead_id(value: &str) -> bool {
    value.starts_with("bd-")
        && value.len() <= 160
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

fn is_utc_timestamp(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 20
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
        || !bytes.iter().enumerate().all(|(index, byte)| {
            matches!(index, 4 | 7 | 10 | 13 | 16 | 19) || byte.is_ascii_digit()
        })
        || !is_review_date(&value[..10])
    {
        return false;
    }
    let hour = u32::from(bytes[11] - b'0') * 10 + u32::from(bytes[12] - b'0');
    let minute = u32::from(bytes[14] - b'0') * 10 + u32::from(bytes[15] - b'0');
    let second = u32::from(bytes[17] - b'0') * 10 + u32::from(bytes[18] - b'0');
    hour < 24 && minute < 60 && second < 60
}

fn is_review_date(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 10
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || !bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| matches!(index, 4 | 7) || byte.is_ascii_digit())
    {
        return false;
    }
    let year = u32::from(bytes[0] - b'0') * 1_000
        + u32::from(bytes[1] - b'0') * 100
        + u32::from(bytes[2] - b'0') * 10
        + u32::from(bytes[3] - b'0');
    let month = u32::from(bytes[5] - b'0') * 10 + u32::from(bytes[6] - b'0');
    let day = u32::from(bytes[8] - b'0') * 10 + u32::from(bytes[9] - b'0');
    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    (1..=days).contains(&day)
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_git_revision(value: &str) -> bool {
    matches!(value.len(), 40 | 64)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_lower_xxh3(value: &str) -> bool {
    value.len() == 16
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn campaign_error(reason: impl Into<String>) -> GauntletError {
    GauntletError::InvalidCampaign {
        reason: reason.into(),
    }
}

fn lower_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut output = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn sha256_text(value: &str) -> String {
    lower_hex(&Sha256::digest(value.as_bytes()))
}

/// One-sided lower quantile of a Beta(successes+1, failures+1) posterior.
fn beta_posterior_lower_bound(successes: u64, total: u64, confidence: f64) -> f64 {
    let alpha = successes as f64 + 1.0;
    let beta = total.saturating_sub(successes) as f64 + 1.0;
    let target = 1.0 - confidence;
    let mut low = 0.0;
    let mut high = 1.0;
    for _ in 0..80 {
        let middle = f64::midpoint(low, high);
        if regularized_beta(middle, alpha, beta) < target {
            low = middle;
        } else {
            high = middle;
        }
    }
    f64::midpoint(low, high)
}

fn regularized_beta(x: f64, alpha: f64, beta: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front = (ln_gamma(alpha + beta) - ln_gamma(alpha) - ln_gamma(beta)
        + alpha * x.ln()
        + beta * (-x).ln_1p())
    .exp();
    if x < (alpha + 1.0) / (alpha + beta + 2.0) {
        front * beta_continued_fraction(alpha, beta, x) / alpha
    } else {
        1.0 - front * beta_continued_fraction(beta, alpha, 1.0 - x) / beta
    }
}

fn beta_continued_fraction(alpha: f64, beta: f64, x: f64) -> f64 {
    const MAX_ITERATIONS: u32 = 256;
    const EPSILON: f64 = 3.0e-14;
    const MIN_DENOMINATOR: f64 = 1.0e-300;

    let sum = alpha + beta;
    let alpha_plus_one = alpha + 1.0;
    let alpha_minus_one = alpha - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - sum * x / alpha_plus_one;
    if d.abs() < MIN_DENOMINATOR {
        d = MIN_DENOMINATOR;
    }
    d = 1.0 / d;
    let mut fraction = d;
    for iteration in 1..=MAX_ITERATIONS {
        let m = f64::from(iteration);
        let twice_m = 2.0 * m;
        let mut coefficient =
            m * (beta - m) * x / ((alpha_minus_one + twice_m) * (alpha + twice_m));
        d = 1.0 + coefficient * d;
        if d.abs() < MIN_DENOMINATOR {
            d = MIN_DENOMINATOR;
        }
        c = 1.0 + coefficient / c;
        if c.abs() < MIN_DENOMINATOR {
            c = MIN_DENOMINATOR;
        }
        d = 1.0 / d;
        fraction *= d * c;

        coefficient =
            -(alpha + m) * (sum + m) * x / ((alpha + twice_m) * (alpha_plus_one + twice_m));
        d = 1.0 + coefficient * d;
        if d.abs() < MIN_DENOMINATOR {
            d = MIN_DENOMINATOR;
        }
        c = 1.0 + coefficient / c;
        if c.abs() < MIN_DENOMINATOR {
            c = MIN_DENOMINATOR;
        }
        d = 1.0 / d;
        let delta = d * c;
        fraction *= delta;
        if (delta - 1.0).abs() <= EPSILON {
            break;
        }
    }
    fraction
}

#[allow(clippy::excessive_precision)]
fn ln_gamma(value: f64) -> f64 {
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let shifted = value - 1.0;
    let series = COEFFICIENTS
        .iter()
        .enumerate()
        .skip(1)
        .fold(COEFFICIENTS[0], |sum, (index, coefficient)| {
            sum + coefficient / (shifted + index as f64)
        });
    let scale = shifted + 7.5;
    0.918_938_533_204_672_7 + (shifted + 0.5) * scale.ln() - scale + series.ln()
}

impl DifferentialCampaignEngine for crate::engine::QuillSubject {
    fn descriptor(&self) -> EngineDescriptor {
        crate::engine::GauntletEngine::descriptor(self)
    }

    fn semantic_contract(&self) -> SemanticContract {
        SemanticContract::scalar_g1a()
    }

    fn core_lexical_search(&self) -> Result<Option<&dyn LexicalSearch>, GauntletError> {
        self.require_committed()?;
        Ok(Some(self.index()?))
    }

    fn begin_corpus<'a>(
        &'a mut self,
        _cx: &'a Cx,
        _manifest: &'a CorpusManifest,
        semantic_contract: &'a SemanticContract,
    ) -> CampaignFuture<'a, ()> {
        Box::pin(async move {
            if semantic_contract != &SemanticContract::scalar_g1a() {
                return Err(campaign_error(
                    "Quill scalar subject requires the scalar G1a semantic contract",
                ));
            }
            self.claim_fresh_campaign()?;
            if self.index()?.doc_count() != 0 || self.index()?.has_uncommitted_changes() {
                return Err(campaign_error(
                    "Quill campaign adapter must own a fresh empty index",
                ));
            }
            Ok(())
        })
    }

    fn index_batch<'a>(
        &'a mut self,
        cx: &'a Cx,
        documents: &'a [GeneratedDocument],
    ) -> CampaignFuture<'a, ()> {
        Box::pin(async move {
            self.require_ingesting()?;
            let indexable = documents
                .iter()
                .cloned()
                .map(frankensearch_core::IndexableDocument::from)
                .collect::<Vec<_>>();
            self.index_mut()?.index_documents(cx, &indexable).await?;
            Ok(())
        })
    }

    fn commit_corpus<'a>(
        &'a mut self,
        cx: &'a Cx,
        manifest: &'a CorpusManifest,
        semantic_contract: &'a SemanticContract,
    ) -> CampaignFuture<'a, EngineIndexReceipt> {
        Box::pin(async move {
            self.require_ingesting()?;
            self.index_mut()?.commit(cx).await?;
            let actual_count = self.index()?.doc_count();
            if actual_count != manifest.document_count {
                return Err(campaign_error(
                    "Quill committed document count differs from the corpus manifest",
                ));
            }
            let receipt = EngineIndexReceipt {
                corpus_manifest_hash: manifest.manifest_hash()?,
                document_count: actual_count,
                total_content_bytes: manifest.total_content_bytes,
                semantic_contract: semantic_contract.clone(),
            };
            self.mark_committed()?;
            Ok(receipt)
        })
    }

    fn observe_generated<'a>(
        &'a mut self,
        cx: &'a Cx,
        query: &'a GeneratedQueryCase,
        evidence_case: &'a DifferentialCase,
    ) -> CampaignFuture<'a, EngineObservation> {
        let query_span = tracing::info_span!(
            target: "frankensearch.quill",
            "frankensearch::quill::gauntlet::query",
            query_id = %query.id,
            query_seed = evidence_case.metadata.generator_seed.unwrap_or_default(),
            corpus_hash = %evidence_case.metadata.corpus_hash.as_deref().unwrap_or("missing"),
        );
        Box::pin(
            async move {
                self.require_committed()?;
                if query.syntax != QuerySyntax::Default
                    || query.filters.created_from_ms.is_some()
                    || query.filters.created_to_ms.is_some()
                {
                    return Err(GauntletError::InvalidCase {
                        reason: "the scalar Quill adapter cannot lower CASS syntax or structured filters"
                            .to_owned(),
                    });
                }
                crate::engine::GauntletEngine::observe(self, cx, evidence_case).await
            }
            .instrument(query_span),
        )
    }

    fn abort_corpus(&mut self) {
        self.abort();
    }
}

#[cfg(feature = "tantivy-oracle")]
impl DifferentialCampaignEngine for crate::engine::TantivyOracle {
    fn descriptor(&self) -> EngineDescriptor {
        GauntletEngine::descriptor(self)
    }

    fn semantic_contract(&self) -> SemanticContract {
        self.campaign_semantic_contract().clone()
    }

    fn core_lexical_search(&self) -> Result<Option<&dyn LexicalSearch>, GauntletError> {
        self.require_committed()?;
        Ok(Some(self.index()))
    }

    fn begin_corpus<'a>(
        &'a mut self,
        _cx: &'a Cx,
        _manifest: &'a CorpusManifest,
        _semantic_contract: &'a SemanticContract,
    ) -> CampaignFuture<'a, ()> {
        Box::pin(async move {
            use frankensearch_core::LexicalSearch;

            if self.index().doc_count() != 0 {
                return Err(campaign_error(
                    "Tantivy campaign adapter must own a fresh empty index",
                ));
            }
            self.claim_fresh_campaign()?;
            Ok(())
        })
    }

    fn index_batch<'a>(
        &'a mut self,
        cx: &'a Cx,
        documents: &'a [GeneratedDocument],
    ) -> CampaignFuture<'a, ()> {
        Box::pin(async move {
            use frankensearch_core::LexicalSearch;

            self.require_ingesting()?;
            let indexable = documents
                .iter()
                .cloned()
                .map(frankensearch_core::IndexableDocument::from)
                .collect::<Vec<_>>();
            self.index().index_documents(cx, &indexable).await?;
            Ok(())
        })
    }

    fn commit_corpus<'a>(
        &'a mut self,
        cx: &'a Cx,
        manifest: &'a CorpusManifest,
        semantic_contract: &'a SemanticContract,
    ) -> CampaignFuture<'a, EngineIndexReceipt> {
        Box::pin(async move {
            use frankensearch_core::LexicalSearch;

            self.require_ingesting()?;
            self.index().commit(cx).await?;
            let actual_count = u64::try_from(self.index().doc_count()).unwrap_or(u64::MAX);
            if actual_count != manifest.document_count {
                return Err(campaign_error(
                    "Tantivy committed document count differs from the corpus manifest",
                ));
            }
            let receipt = EngineIndexReceipt {
                corpus_manifest_hash: manifest.manifest_hash()?,
                document_count: actual_count,
                total_content_bytes: manifest.total_content_bytes,
                semantic_contract: semantic_contract.clone(),
            };
            self.mark_committed()?;
            Ok(receipt)
        })
    }

    fn observe_generated<'a>(
        &'a mut self,
        cx: &'a Cx,
        query: &'a GeneratedQueryCase,
        evidence_case: &'a DifferentialCase,
    ) -> CampaignFuture<'a, EngineObservation> {
        Box::pin(async move {
            self.require_committed()?;
            if query.syntax != QuerySyntax::Default
                || query.filters.created_from_ms.is_some()
                || query.filters.created_to_ms.is_some()
            {
                return Err(GauntletError::InvalidCase {
                    reason: "the shipping-schema Tantivy adapter cannot lower CASS syntax or structured filters"
                        .to_owned(),
                });
            }
            GauntletEngine::observe(self, cx, evidence_case).await
        })
    }

    fn abort_corpus(&mut self) {
        // Tantivy does not expose rollback through the shipping lexical
        // facade. Poison the one-shot adapter so no post-abort differential
        // operation can mistake retained backend bytes for an admissible
        // campaign snapshot.
        self.abort_campaign();
    }
}

#[cfg(feature = "tantivy-oracle")]
fn lower_quill_cass_filters(
    filters: &crate::generator::GeneratedQueryFilters,
) -> frankensearch_quill::CassQueryFilters {
    let source_filter = match &filters.source_filter {
        GeneratedSourceFilter::All => frankensearch_quill::CassSourceFilter::All,
        GeneratedSourceFilter::Local => frankensearch_quill::CassSourceFilter::Local,
        GeneratedSourceFilter::Remote => frankensearch_quill::CassSourceFilter::Remote,
        GeneratedSourceFilter::SourceId { source_id } => {
            frankensearch_quill::CassSourceFilter::SourceId(source_id.clone())
        }
    };
    frankensearch_quill::CassQueryFilters {
        agents: filters.agents.clone(),
        workspaces: filters.workspaces.clone(),
        created_from: filters.created_from_ms,
        created_to: filters.created_to_ms,
        source_filter,
    }
}

#[cfg(feature = "tantivy-oracle")]
fn lower_tantivy_cass_filters(
    filters: &crate::generator::GeneratedQueryFilters,
) -> frankensearch_lexical::CassQueryFilters {
    let source_filter = match &filters.source_filter {
        GeneratedSourceFilter::All => frankensearch_lexical::CassSourceFilter::All,
        GeneratedSourceFilter::Local => frankensearch_lexical::CassSourceFilter::Local,
        GeneratedSourceFilter::Remote => frankensearch_lexical::CassSourceFilter::Remote,
        GeneratedSourceFilter::SourceId { source_id } => {
            frankensearch_lexical::CassSourceFilter::SourceId(source_id.clone())
        }
    };
    frankensearch_lexical::CassQueryFilters {
        agents: filters.agents.clone(),
        workspaces: filters.workspaces.clone(),
        created_from: filters.created_from_ms,
        created_to: filters.created_to_ms,
        source_filter,
    }
}

#[cfg(feature = "tantivy-oracle")]
impl DifferentialCampaignEngine for crate::engine::CassQuillSubject {
    fn descriptor(&self) -> EngineDescriptor {
        Self::descriptor(self)
    }

    fn semantic_contract(&self) -> SemanticContract {
        SemanticContract::cass()
    }

    fn begin_corpus<'a>(
        &'a mut self,
        _cx: &'a Cx,
        _manifest: &'a CorpusManifest,
        semantic_contract: &'a SemanticContract,
    ) -> CampaignFuture<'a, ()> {
        Box::pin(async move {
            if semantic_contract != &SemanticContract::cass() {
                return Err(campaign_error(
                    "Quill CASS subject requires the CASS semantic contract",
                ));
            }
            self.claim_fresh_campaign()
        })
    }

    fn index_batch<'a>(
        &'a mut self,
        cx: &'a Cx,
        documents: &'a [GeneratedDocument],
    ) -> CampaignFuture<'a, ()> {
        Box::pin(async move { self.index_generated_batch(cx, documents) })
    }

    fn commit_corpus<'a>(
        &'a mut self,
        cx: &'a Cx,
        manifest: &'a CorpusManifest,
        semantic_contract: &'a SemanticContract,
    ) -> CampaignFuture<'a, EngineIndexReceipt> {
        Box::pin(async move {
            if semantic_contract != &SemanticContract::cass() {
                return Err(campaign_error(
                    "Quill CASS commit received a different semantic contract",
                ));
            }
            let actual_count = u64::try_from(Self::commit_corpus(self, cx)?).unwrap_or(u64::MAX);
            if actual_count != manifest.document_count {
                return Err(campaign_error(
                    "Quill CASS committed document count differs from the corpus manifest",
                ));
            }
            Ok(EngineIndexReceipt {
                corpus_manifest_hash: manifest.manifest_hash()?,
                document_count: actual_count,
                total_content_bytes: manifest.total_content_bytes,
                semantic_contract: semantic_contract.clone(),
            })
        })
    }

    fn observe_generated<'a>(
        &'a mut self,
        cx: &'a Cx,
        query: &'a GeneratedQueryCase,
        evidence_case: &'a DifferentialCase,
    ) -> CampaignFuture<'a, EngineObservation> {
        let query_span = tracing::info_span!(
            target: "frankensearch.quill",
            "frankensearch::quill::gauntlet::cass_query",
            query_id = %query.id,
            query_seed = evidence_case.metadata.generator_seed.unwrap_or_default(),
            corpus_hash = %evidence_case.metadata.corpus_hash.as_deref().unwrap_or("missing"),
        );
        Box::pin(
            async move {
                if query.syntax != QuerySyntax::Cass {
                    return Err(GauntletError::InvalidCase {
                        reason: "the CASS Quill adapter rejects default query syntax".to_owned(),
                    });
                }
                let filters = lower_quill_cass_filters(&query.filters);
                self.observe_cass(cx, evidence_case, &filters)
            }
            .instrument(query_span),
        )
    }

    fn abort_corpus(&mut self) {
        Self::abort(self);
    }
}

#[cfg(feature = "tantivy-oracle")]
impl DifferentialCampaignEngine for crate::engine::CassTantivyOracle {
    fn descriptor(&self) -> EngineDescriptor {
        Self::descriptor(self)
    }

    fn semantic_contract(&self) -> SemanticContract {
        SemanticContract::cass()
    }

    fn begin_corpus<'a>(
        &'a mut self,
        _cx: &'a Cx,
        _manifest: &'a CorpusManifest,
        semantic_contract: &'a SemanticContract,
    ) -> CampaignFuture<'a, ()> {
        Box::pin(async move {
            if semantic_contract != &SemanticContract::cass() {
                return Err(campaign_error(
                    "Tantivy CASS oracle requires the CASS semantic contract",
                ));
            }
            self.claim_fresh_campaign()
        })
    }

    fn index_batch<'a>(
        &'a mut self,
        cx: &'a Cx,
        documents: &'a [GeneratedDocument],
    ) -> CampaignFuture<'a, ()> {
        Box::pin(async move { self.index_generated_batch(cx, documents) })
    }

    fn commit_corpus<'a>(
        &'a mut self,
        cx: &'a Cx,
        manifest: &'a CorpusManifest,
        semantic_contract: &'a SemanticContract,
    ) -> CampaignFuture<'a, EngineIndexReceipt> {
        Box::pin(async move {
            if semantic_contract != &SemanticContract::cass() {
                return Err(campaign_error(
                    "Tantivy CASS commit received a different semantic contract",
                ));
            }
            let actual_count = u64::try_from(Self::commit_corpus(self, cx)?).unwrap_or(u64::MAX);
            if actual_count != manifest.document_count {
                return Err(campaign_error(
                    "Tantivy CASS committed document count differs from the corpus manifest",
                ));
            }
            Ok(EngineIndexReceipt {
                corpus_manifest_hash: manifest.manifest_hash()?,
                document_count: actual_count,
                total_content_bytes: manifest.total_content_bytes,
                semantic_contract: semantic_contract.clone(),
            })
        })
    }

    fn observe_generated<'a>(
        &'a mut self,
        cx: &'a Cx,
        query: &'a GeneratedQueryCase,
        evidence_case: &'a DifferentialCase,
    ) -> CampaignFuture<'a, EngineObservation> {
        Box::pin(async move {
            if query.syntax != QuerySyntax::Cass {
                return Err(GauntletError::InvalidCase {
                    reason: "the CASS Tantivy adapter rejects default query syntax".to_owned(),
                });
            }
            let filters = lower_tantivy_cass_filters(&query.filters);
            self.observe_cass(cx, evidence_case, &filters)
        })
    }

    fn abort_corpus(&mut self) {
        Self::abort(self);
    }
}

// ============================================================================
// Divergence shrinker + explanation-driven auto-triage
// (bd-quill-duel-shrinker-2j21)
//
// Given a divergent (corpus, query) pair, ddmin the corpus to a minimal
// reproducer and greedily minimize the query, exploiting engine determinism:
// a candidate pair either still exhibits the divergence or it does not. The
// shrunk case persists as a permanent regression fixture alongside the
// ORIGINAL query text and corpus manifest hash (over-minimization loses
// parser-edge context, so the original is always retained).
// ============================================================================

/// Default candidate-evaluation budget for one shrink run.
pub const DEFAULT_SHRINK_FUEL: usize = 256;
/// Corpus size at which ddmin refinement stops.
const SHRINK_TARGET_DOCS: usize = 3;

/// Suspected engine layer for one triaged score divergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuspectedLayer {
    /// BM25 arithmetic (fieldnorm quantization, avgdl, idf) diverges.
    FieldNormArithmetic,
    /// Query parsing or AST lowering produced a different plan.
    ParserLowering,
    /// Native tie-break ordering diverges on equal scores.
    TieOrder,
    /// Rank-safe pruning dropped a different candidate set.
    Pruning,
    /// Documents indexed differently (content or identity loss).
    Indexing,
    /// Evidence does not isolate a layer.
    Unknown,
}

/// Confidence of one auto-triage verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TriageConfidence {
    /// Direct structural evidence (AST diff, tie-group proof).
    High,
    /// Strong statistical shape (score deltas with identical sets).
    Medium,
    /// Weak shape; needs human review.
    Low,
}

/// Auto-triage verdict for one shrunk score divergence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TriageVerdict {
    /// Comparator class that persisted through the shrink.
    pub class: DivergenceClass,
    /// Suspected engine layer.
    pub suspected_layer: SuspectedLayer,
    /// Verdict confidence.
    pub confidence: TriageConfidence,
    /// Human-readable evidence rows.
    pub evidence: Vec<String>,
}

/// One permanent shrunk regression fixture: minimal reproduction plus the
/// original context (Gemini's anti-over-minimization amendment).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShrunkReproduction {
    /// Fixture schema version.
    pub schema_version: u32,
    /// Divergence class that persisted to the minimal reproducer.
    pub divergence_class: DivergenceClass,
    /// Content hash of the FULL original corpus manifest.
    pub original_corpus_manifest_hash: String,
    /// Original corpus size before shrinking.
    pub original_document_count: usize,
    /// Original query text, untouched.
    pub original_query_text: String,
    /// Original structured query identity.
    pub original_query_id: String,
    /// Minimal corpus that still diverges.
    pub minimized_documents: Vec<GeneratedDocument>,
    /// Minimal query text that still diverges.
    pub minimized_query_text: String,
    /// Auto-triage verdict over the minimal reproducer.
    pub triage: TriageVerdict,
    /// Accepted reduction steps (document or query removals).
    pub reduction_steps: usize,
    /// Total candidates evaluated (fuel consumed).
    pub candidates_evaluated: usize,
}

/// Shadow-mode divergence record (`.quill-shadow/divergences.jsonl`).
///
/// The stamped generation is the snapshot witness for exact-snapshot replay:
/// a shadow reader can rebuild the same committed generation and re-run the
/// shrunk reproduction against it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowDivergenceRecord {
    /// Record schema version (currently 1).
    pub schema_version: u32,
    /// MANIFEST generation stamped when the divergence fired.
    pub stamped_generation: u64,
    /// Full-corpus manifest hash for replay identity.
    pub corpus_manifest_hash: String,
    /// Full corpus as indexed by the shadow pair.
    pub documents: Vec<GeneratedDocument>,
    /// Structured divergent query.
    pub query: GeneratedQueryCase,
    /// Engine-neutral evidence envelope.
    pub evidence_case: DifferentialCase,
    /// Class the shadow comparator reported.
    pub divergence_class: DivergenceClass,
}

/// Errors from shrink orchestration.
#[derive(Debug, thiserror::Error)]
pub enum ShrinkError {
    /// The candidate budget ran out before reaching a fixpoint.
    #[error("shrink fuel exhausted after {evaluated} candidate evaluations")]
    FuelExhausted {
        /// Candidates evaluated before exhaustion.
        evaluated: usize,
    },
    /// A campaign adapter or comparator failed mid-evaluation.
    #[error("shrink campaign failed: {0}")]
    Campaign(#[from] GauntletError),
    /// A shadow divergence record line is malformed.
    #[error("shadow divergence record invalid: {reason}")]
    InvalidShadowRecord {
        /// Parse/validation detail.
        reason: String,
    },
    /// The permanent fixture could not be written durably.
    #[error("shrunk reproduction persist failed at {path}: {reason}")]
    Persist {
        /// Target fixture path.
        path: std::path::PathBuf,
        /// I/O detail.
        reason: String,
    },
}

/// Input to one shrink run.
pub struct ShrinkRequest {
    /// Full divergent corpus.
    pub documents: Vec<GeneratedDocument>,
    /// Content hash of the full corpus manifest.
    pub corpus_manifest_hash: String,
    /// Structured divergent query.
    pub query: GeneratedQueryCase,
    /// Engine-neutral evidence envelope shared by both engines.
    pub evidence_case: DifferentialCase,
    /// Comparator failure class to preserve through the shrink.
    pub divergence_class: DivergenceClass,
}

/// Factory for one fresh, empty campaign engine.
pub type ShrinkEngineFactory =
    Box<dyn FnMut() -> Result<Box<dyn DifferentialCampaignEngine>, GauntletError>>;

/// ddmin + greedy-query shrinker over the campaign engine boundary.
pub struct ShrinkDriver {
    comparator_config: ComparatorConfig,
    semantic_contract: SemanticContract,
    fuel: usize,
}

impl ShrinkDriver {
    /// Construct a driver with explicit comparator configuration and fuel.
    #[must_use]
    pub const fn new(
        comparator_config: ComparatorConfig,
        semantic_contract: SemanticContract,
        fuel: usize,
    ) -> Self {
        Self {
            comparator_config,
            semantic_contract,
            fuel,
        }
    }

    /// Shrink one divergent (corpus, query) pair to a minimal reproduction.
    ///
    /// Document ddmin follows Zeller's delta-debugging: split the corpus into
    /// `n` chunks and drop chunks while the divergence persists, increasing
    /// `n` when no single chunk drops. Query minimization greedily removes
    /// whitespace-delimited tokens while the divergence persists.
    ///
    /// # Errors
    ///
    /// Returns [`ShrinkError::FuelExhausted`] when the candidate budget runs
    /// out, or [`ShrinkError::Campaign`] when an adapter fails.
    #[allow(clippy::future_not_send)]
    pub async fn shrink(
        &self,
        cx: &Cx,
        request: &ShrinkRequest,
        make_subject: &mut ShrinkEngineFactory,
        make_oracle: &mut ShrinkEngineFactory,
    ) -> Result<ShrunkReproduction, ShrinkError> {
        let mut budget = ShrinkBudget {
            remaining: self.fuel,
            evaluated: 0,
        };
        let mut documents = request.documents.clone();
        let mut steps = 0_usize;

        // ddmin on documents.
        let mut n = 2_usize;
        while documents.len() > SHRINK_TARGET_DOCS && n <= documents.len() {
            let chunk = documents.len().div_ceil(n);
            let mut reduced = false;
            for start in (0..documents.len()).step_by(chunk) {
                let end = (start + chunk).min(documents.len());
                let mut candidate = documents[..start].to_vec();
                candidate.extend_from_slice(&documents[end..]);
                if candidate.is_empty() {
                    continue;
                }
                if self
                    .persists(
                        cx,
                        &candidate,
                        &request.query,
                        &request.evidence_case,
                        request.divergence_class,
                        make_subject,
                        make_oracle,
                        &mut budget,
                    )
                    .await?
                {
                    documents = candidate;
                    n = (n - 1).max(2);
                    steps += 1;
                    reduced = true;
                    break;
                }
            }
            if !reduced {
                if n >= documents.len() {
                    break;
                }
                n = (n * 2).min(documents.len());
            }
        }

        // Greedy token-level query minimization.
        let mut tokens: Vec<String> = request
            .query
            .query
            .split_whitespace()
            .map(str::to_owned)
            .collect();
        if tokens.len() > 1 {
            let mut index = 0;
            while index < tokens.len() {
                let candidate_tokens: Vec<String> = tokens
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != index)
                    .map(|(_, token)| token.clone())
                    .collect();
                let candidate_text = candidate_tokens.join(" ");
                if candidate_text.trim().is_empty() {
                    index += 1;
                    continue;
                }
                let mut candidate_query = request.query.clone();
                candidate_query.query = candidate_text.clone();
                let mut candidate_case = request.evidence_case.clone();
                candidate_case.query = candidate_text.clone();
                if self
                    .persists(
                        cx,
                        &documents,
                        &candidate_query,
                        &candidate_case,
                        request.divergence_class,
                        make_subject,
                        make_oracle,
                        &mut budget,
                    )
                    .await?
                {
                    tokens = candidate_tokens;
                    steps += 1;
                } else {
                    index += 1;
                }
            }
        }
        let query_text = tokens.join(" ");

        // Final evidence + auto-triage on the minimal reproducer.
        let mut minimized_query = request.query.clone();
        minimized_query.query = query_text.clone();
        let mut minimized_case = request.evidence_case.clone();
        minimized_case.query = query_text.clone();
        let final_report = self
            .evaluate(
                cx,
                &documents,
                &minimized_query,
                &minimized_case,
                make_subject,
                make_oracle,
                &mut budget,
            )
            .await?;
        let triage = auto_triage(request.divergence_class, &final_report);

        Ok(ShrunkReproduction {
            schema_version: 1,
            divergence_class: request.divergence_class,
            original_corpus_manifest_hash: request.corpus_manifest_hash.clone(),
            original_document_count: request.documents.len(),
            original_query_text: request.query.query.clone(),
            original_query_id: request.query.id.clone(),
            minimized_documents: documents,
            minimized_query_text: query_text,
            triage,
            reduction_steps: steps,
            candidates_evaluated: budget.evaluated,
        })
    }

    /// Parse one `.quill-shadow/divergences.jsonl` line and shrink it.
    ///
    /// The stamped generation is preserved in the reproduction's manifest
    /// hash fields for exact-snapshot replay by the shadow reader.
    ///
    /// # Errors
    ///
    /// Returns [`ShrinkError::InvalidShadowRecord`] for malformed lines, or
    /// the shrinker's own errors otherwise.
    #[allow(clippy::future_not_send)]
    pub async fn shrink_shadow_line(
        &self,
        cx: &Cx,
        line: &str,
        make_subject: &mut ShrinkEngineFactory,
        make_oracle: &mut ShrinkEngineFactory,
    ) -> Result<ShrunkReproduction, ShrinkError> {
        let envelope: serde_json::Value =
            serde_json::from_str(line).map_err(|error| ShrinkError::InvalidShadowRecord {
                reason: error.to_string(),
            })?;
        let schema_version = envelope
            .get("schema_version")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| ShrinkError::InvalidShadowRecord {
                reason: "shadow record has no integer schema_version".to_owned(),
            })?;
        let request = match schema_version {
            1 => {
                let record: ShadowDivergenceRecord =
                    serde_json::from_value(envelope).map_err(|error| {
                        ShrinkError::InvalidShadowRecord {
                            reason: error.to_string(),
                        }
                    })?;
                legacy_shadow_request(record)?
            }
            version
                if version == u64::from(frankensearch_core::SHADOW_DIVERGENCE_SCHEMA_VERSION) =>
            {
                let record: frankensearch_core::ShadowDivergenceRecord =
                    serde_json::from_value(envelope).map_err(|error| {
                        ShrinkError::InvalidShadowRecord {
                            reason: error.to_string(),
                        }
                    })?;
                production_shadow_request(record)?
            }
            unsupported => {
                return Err(ShrinkError::InvalidShadowRecord {
                    reason: format!("unsupported shadow record schema {unsupported}"),
                });
            }
        };
        self.shrink(cx, &request, make_subject, make_oracle).await
    }

    /// Whether the candidate pair still exhibits the target divergence class.
    #[allow(clippy::future_not_send)]
    async fn persists(
        &self,
        cx: &Cx,
        documents: &[GeneratedDocument],
        query: &GeneratedQueryCase,
        evidence_case: &DifferentialCase,
        target_class: DivergenceClass,
        make_subject: &mut ShrinkEngineFactory,
        make_oracle: &mut ShrinkEngineFactory,
        budget: &mut ShrinkBudget,
    ) -> Result<bool, ShrinkError> {
        let report = self
            .evaluate(
                cx,
                documents,
                query,
                evidence_case,
                make_subject,
                make_oracle,
                budget,
            )
            .await?;
        Ok(report
            .divergences
            .iter()
            .any(|divergence| divergence.class == target_class))
    }

    /// Index a candidate into fresh engines and compare observations.
    #[allow(clippy::future_not_send)]
    async fn evaluate(
        &self,
        cx: &Cx,
        documents: &[GeneratedDocument],
        query: &GeneratedQueryCase,
        evidence_case: &DifferentialCase,
        make_subject: &mut ShrinkEngineFactory,
        make_oracle: &mut ShrinkEngineFactory,
        budget: &mut ShrinkBudget,
    ) -> Result<ComparisonReport, ShrinkError> {
        budget.spend()?;
        let manifest = subset_manifest(documents)?;
        let mut subject = make_subject()?;
        let mut oracle = make_oracle()?;
        subject
            .begin_corpus(cx, &manifest, &self.semantic_contract)
            .await?;
        oracle
            .begin_corpus(cx, &manifest, &self.semantic_contract)
            .await?;
        subject.index_batch(cx, documents).await?;
        oracle.index_batch(cx, documents).await?;
        subject
            .commit_corpus(cx, &manifest, &self.semantic_contract)
            .await?;
        oracle
            .commit_corpus(cx, &manifest, &self.semantic_contract)
            .await?;
        let subject_observation = subject.observe_generated(cx, query, evidence_case).await?;
        let oracle_observation = oracle.observe_generated(cx, query, evidence_case).await?;
        Ok(compare_observations(
            subject_observation,
            oracle_observation,
            self.comparator_config,
        )?)
    }
}

fn legacy_shadow_request(record: ShadowDivergenceRecord) -> Result<ShrinkRequest, ShrinkError> {
    if record.documents.is_empty() {
        return Err(ShrinkError::InvalidShadowRecord {
            reason: "shadow record carries no documents".to_owned(),
        });
    }
    Ok(ShrinkRequest {
        documents: record.documents,
        corpus_manifest_hash: format!(
            "{}#gen-{}",
            record.corpus_manifest_hash, record.stamped_generation
        ),
        query: record.query,
        evidence_case: record.evidence_case,
        divergence_class: record.divergence_class,
    })
}

fn production_shadow_request(
    record: frankensearch_core::ShadowDivergenceRecord,
) -> Result<ShrinkRequest, ShrinkError> {
    record
        .validate()
        .map_err(|reason| ShrinkError::InvalidShadowRecord { reason })?;
    let manifest_generation = record.manifest_generation;
    let corpus_hash = record.corpus_hash;
    let query_text = record.query.text;
    let limit =
        u64::try_from(record.query.limit).map_err(|_| ShrinkError::InvalidShadowRecord {
            reason: "shadow query limit does not fit u64".to_owned(),
        })?;
    let query_kind = if query_text.split_whitespace().nth(1).is_some() {
        GeneratedQueryKind::MultiTerm
    } else {
        GeneratedQueryKind::Term
    };
    let query_id = format!("shadow-generation-{manifest_generation}");
    let query = GeneratedQueryCase {
        id: query_id.clone(),
        syntax: QuerySyntax::Default,
        query_kind,
        query: query_text.clone(),
        limit,
        offset: 0,
        count_requested: true,
        filters: crate::generator::GeneratedQueryFilters::default(),
        expected_divergence: None,
        source: "shadow-oracle".to_owned(),
    };
    let mut evidence_case = DifferentialCase::new(query_id, query_text, limit);
    evidence_case.metadata = DifferentialCaseMetadata {
        generator_id: Some("shadow-oracle-v2".to_owned()),
        generator_seed: Some(manifest_generation),
        corpus_hash: Some(corpus_hash.clone()),
    };
    let divergence_class = match record.classification {
        frankensearch_core::ShadowDivergenceClass::ScoreEpsilon => DivergenceClass::ScoreEpsilon,
        frankensearch_core::ShadowDivergenceClass::TieOrder => DivergenceClass::TieOrder,
        frankensearch_core::ShadowDivergenceClass::RankMismatch
        | frankensearch_core::ShadowDivergenceClass::ScoreMismatch => DivergenceClass::RankMismatch,
        frankensearch_core::ShadowDivergenceClass::Exact => {
            return Err(ShrinkError::InvalidShadowRecord {
                reason: "divergence stream cannot contain exact comparisons".to_owned(),
            });
        }
    };
    let documents = record
        .documents
        .into_iter()
        .map(|document| {
            let created_at_ms = document
                .metadata
                .get("created_at_ms")
                .and_then(|value| value.parse::<i64>().ok())
                .unwrap_or_default();
            GeneratedDocument {
                id: document.id,
                title: document.title,
                content: document.content,
                created_at_ms,
                cass: None,
                metadata: document.metadata,
                pathology: None,
                unicode_lane: crate::generator::UnicodeLane::Mixed,
            }
        })
        .collect();
    Ok(ShrinkRequest {
        documents,
        corpus_manifest_hash: format!("{corpus_hash}#gen-{manifest_generation}"),
        query,
        evidence_case,
        divergence_class,
    })
}

struct ShrinkBudget {
    remaining: usize,
    evaluated: usize,
}

impl ShrinkBudget {
    fn spend(&mut self) -> Result<(), ShrinkError> {
        if self.remaining == 0 {
            return Err(ShrinkError::FuelExhausted {
                evaluated: self.evaluated,
            });
        }
        self.remaining -= 1;
        self.evaluated += 1;
        Ok(())
    }
}

/// Build a valid corpus manifest for a shrunk document subset.
fn subset_manifest(documents: &[GeneratedDocument]) -> Result<CorpusManifest, GauntletError> {
    let mut hasher = Sha256::new();
    let mut total_content_bytes = 0_u64;
    for document in documents {
        let bytes =
            serde_json::to_vec(document).map_err(|error| GauntletError::InvalidGenerator {
                reason: format!("subset manifest canonicalization failed: {error}"),
            })?;
        hasher.update((bytes.len() as u64).to_be_bytes());
        hasher.update(&bytes);
        total_content_bytes = total_content_bytes
            .checked_add(u64::try_from(document.content.len()).unwrap_or(u64::MAX))
            .ok_or_else(|| GauntletError::InvalidGenerator {
                reason: "subset content byte overflow".to_owned(),
            })?;
    }
    let digest = hasher.finalize();
    let mut content_sha256 = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(content_sha256, "{byte:02x}");
    }
    Ok(CorpusManifest {
        schema_version: 1,
        generator_id: GENERATOR_ID.to_owned(),
        source: crate::generator::CorpusSourceManifest::Synthetic {
            spec: crate::generator::SyntheticCorpusSpec {
                seed: 0,
                document_count: u64::try_from(documents.len()).map_err(|_| {
                    GauntletError::InvalidGenerator {
                        reason: "subset document count does not fit u64".to_owned(),
                    }
                })?,
                vocabulary_size: 1,
                zipf_exponent: crate::generator::ZipfExponent::S08,
                max_document_bytes: crate::generator::MAX_DOCUMENT_BYTES,
            },
        },
        document_count: u64::try_from(documents.len()).map_err(|_| {
            GauntletError::InvalidGenerator {
                reason: "subset document count does not fit u64".to_owned(),
            }
        })?,
        total_content_bytes,
        content_sha256,
        skipped_repository_entries: Vec::new(),
    })
}

/// Explanation-driven auto-triage over the minimal reproducer.
///
/// The v1 verdict maps comparator evidence to a suspected layer with explicit
/// confidence: AST differences name parser lowering directly; tie classes
/// name ordering; score-only deltas with identical document sets point at
/// BM25 arithmetic; missing/extra documents point at indexing. Factor-level
/// (idf/tf/norm) decomposition lands when observations carry factor
/// breakdowns.
fn auto_triage(target: DivergenceClass, report: &ComparisonReport) -> TriageVerdict {
    let mut evidence = Vec::new();
    let has_ast_difference = report
        .subject
        .ast_differences
        .iter()
        .chain(report.oracle.ast_differences.iter())
        .any(|difference| {
            matches!(
                difference.kind,
                crate::comparator::AstLoweringKind::OversizedQueryToken
            )
        });
    let subject_ids: BTreeSet<&str> = report
        .subject
        .hits
        .iter()
        .map(|hit| hit.doc_id.as_str())
        .collect();
    let oracle_ids: BTreeSet<&str> = report
        .oracle
        .hits
        .iter()
        .map(|hit| hit.doc_id.as_str())
        .collect();
    let same_set = subject_ids == oracle_ids;
    evidence.push(format!(
        "document sets {} (subject {} hits, oracle {} hits)",
        if same_set { "identical" } else { "differ" },
        subject_ids.len(),
        oracle_ids.len()
    ));
    let max_score_delta = report
        .subject
        .hits
        .iter()
        .filter_map(|hit| {
            report
                .oracle
                .hits
                .iter()
                .find(|oracle_hit| oracle_hit.doc_id == hit.doc_id)
                .map(|oracle_hit| {
                    let delta = i64::from(hit.score_bits) - i64::from(oracle_hit.score_bits);
                    delta.unsigned_abs()
                })
        })
        .fold(0_u64, u64::max);
    evidence.push(format!(
        "max score-bit delta over shared hits: {max_score_delta:.0}"
    ));

    let (suspected_layer, confidence) = if has_ast_difference
        || target == DivergenceClass::OversizedQueryToken
    {
        evidence.push("oversized-token AST lowering difference present".to_owned());
        (SuspectedLayer::ParserLowering, TriageConfidence::High)
    } else {
        match target {
            DivergenceClass::TieOrder => {
                evidence.push("rank flips confined to equal-score tie groups".to_owned());
                (SuspectedLayer::TieOrder, TriageConfidence::High)
            }
            DivergenceClass::ScoreEpsilon => {
                evidence.push("identical result sets with sub-epsilon score deltas".to_owned());
                (
                    SuspectedLayer::FieldNormArithmetic,
                    TriageConfidence::Medium,
                )
            }
            DivergenceClass::RankMismatch if same_set => {
                evidence.push("rank flips beyond tie groups with identical sets".to_owned());
                (
                    SuspectedLayer::FieldNormArithmetic,
                    TriageConfidence::Medium,
                )
            }
            DivergenceClass::RankMismatch => {
                evidence
                    .push("result sets differ; indexing or parse-time document loss".to_owned());
                (SuspectedLayer::Indexing, TriageConfidence::Low)
            }
            DivergenceClass::SnippetMismatch => {
                evidence.push("snippet windows disagree on identical hits".to_owned());
                (SuspectedLayer::ParserLowering, TriageConfidence::Low)
            }
            DivergenceClass::SnippetWindow => {
                evidence.push("reviewed snippet-window selection differs".to_owned());
                (SuspectedLayer::ParserLowering, TriageConfidence::High)
            }
            DivergenceClass::CountMismatch | DivergenceClass::DocumentCountMismatch => {
                evidence.push("count evidence disagrees with the oracle".to_owned());
                (SuspectedLayer::Indexing, TriageConfidence::Medium)
            }
            DivergenceClass::GlobExpansionLimit
            | DivergenceClass::QueryCanonicalization
            | DivergenceClass::OracleBug
            | DivergenceClass::UnicodeEdge => {
                evidence.push("reviewed query-lowering divergence present".to_owned());
                (SuspectedLayer::ParserLowering, TriageConfidence::High)
            }
            DivergenceClass::StatsSemantics => {
                evidence.push("reviewed snapshot-statistics divergence present".to_owned());
                (SuspectedLayer::FieldNormArithmetic, TriageConfidence::High)
            }
            DivergenceClass::PostingRecordSemantics => {
                evidence.push(
                    "posting record option disagrees with scorer or pruning-bound frequency"
                        .to_owned(),
                );
                (SuspectedLayer::FieldNormArithmetic, TriageConfidence::High)
            }
            DivergenceClass::OversizedQueryToken => unreachable!("covered above"),
        }
    };
    TriageVerdict {
        class: target,
        suspected_layer,
        confidence,
        evidence,
    }
}

/// Persist a shrunk reproduction as a permanent regression fixture.
///
/// The fixture is content-addressed (`<root>/shrunk/<sha256>.json`) and
/// written with the house temp+rename+dir-fsync discipline, so concurrent
/// shrink runs never observe a torn fixture.
///
/// # Errors
///
/// Returns [`ShrinkError::Persist`] for canonicalization or I/O failures.
pub fn persist_shrunk_reproduction(
    root: &std::path::Path,
    reproduction: &ShrunkReproduction,
) -> Result<std::path::PathBuf, ShrinkError> {
    let bytes = serde_json::to_vec_pretty(reproduction).map_err(|error| ShrinkError::Persist {
        path: root.to_path_buf(),
        reason: format!("fixture canonicalization failed: {error}"),
    })?;
    let digest = Sha256::digest(&bytes);
    let mut hash = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(hash, "{byte:02x}");
    }
    let directory = root.join("shrunk");
    std::fs::create_dir_all(&directory).map_err(|error| ShrinkError::Persist {
        path: directory.clone(),
        reason: error.to_string(),
    })?;
    let target = directory.join(format!("{hash}.json"));
    let temporary = directory.join(format!(".tmp-shrunk-{}-{hash}", std::process::id()));
    std::fs::write(&temporary, &bytes).map_err(|error| ShrinkError::Persist {
        path: temporary.clone(),
        reason: error.to_string(),
    })?;
    {
        let file = std::fs::File::open(&temporary).map_err(|error| ShrinkError::Persist {
            path: temporary.clone(),
            reason: error.to_string(),
        })?;
        file.sync_all().map_err(|error| ShrinkError::Persist {
            path: temporary.clone(),
            reason: error.to_string(),
        })?;
    }
    std::fs::rename(&temporary, &target).map_err(|error| ShrinkError::Persist {
        path: target.clone(),
        reason: error.to_string(),
    })?;
    if let Ok(directory_file) = std::fs::File::open(&directory) {
        let _ = directory_file.sync_all();
    }
    Ok(target)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "tantivy-oracle")]
    use std::io::{self, Read, Write};
    #[cfg(feature = "tantivy-oracle")]
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::comparator::{AstDifference, AstLoweringKind, CountState, NativeTieKey, RankedHit};
    #[cfg(feature = "tantivy-oracle")]
    use crate::comparator::{
        LexicalContractBundle, LexicalContractComparison, LexicalCountExposure, LexicalCountState,
        LexicalEmptyShape, LexicalErrorClass, LexicalErrorObservation, LexicalFieldExposure,
        LexicalHitObservation, LexicalHydrationExecution, LexicalHydrationResult,
        LexicalHydrationSelection, LexicalNonLexicalControlKind, LexicalNormalizedQuery,
        LexicalObservation, LexicalObservationContext, LexicalObservationOutcome, LexicalObserved,
        LexicalQueryClass, LexicalScoreSource, LexicalWinnerOrigin, LexicalWinnerProjection,
        SensitiveValueObservation,
    };
    use crate::engine::{EngineFamily, TANTIVY_ORACLE_CONFIG_HASH};
    use crate::generator::{
        QueryGeneratorSpec, RepositoryEntry, RepositorySnapshot, SharedFixtureSuite,
        SyntheticCorpus, SyntheticCorpusSpec, ZipfExponent,
    };
    use crate::version_contract::oracle_version_contract;

    use super::*;

    #[cfg(feature = "tantivy-oracle")]
    #[derive(Clone, Debug)]
    struct TraceLogWriter {
        buffer: Arc<Mutex<Vec<u8>>>,
    }

    #[cfg(feature = "tantivy-oracle")]
    impl Write for TraceLogWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.buffer
                .lock()
                .expect("trace buffer lock is not poisoned")
                .extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[cfg(feature = "tantivy-oracle")]
    const E410_RANK_GOLDEN_JSON: &str = include_str!("../fixtures/argus-e410-ranks-v1.json");
    const DIVERGENCE_REGISTER_FIXTURE_JSON: &str =
        include_str!("../fixtures/divergence-register-v1.json");
    const DIVERGENCE_REGISTER_SCHEMA_JSON: &str =
        include_str!("../../../schemas/quill-divergence-register-v1.schema.json");

    fn divergence_ledger_fixture() -> DivergenceRegisterLedger {
        serde_json::from_str(DIVERGENCE_REGISTER_FIXTURE_JSON)
            .expect("decode Divergence Register contract fixture")
    }

    #[cfg(feature = "tantivy-oracle")]
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct E410RankGolden {
        schema_version: u32,
        corpus_manifest_hash: String,
        query_manifest_hash: String,
        query_seed: u64,
        cases: Vec<E410RankCase>,
    }

    #[cfg(feature = "tantivy-oracle")]
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct E410RankCase {
        query_id: String,
        ranked_document_ids: Vec<String>,
    }

    #[cfg(feature = "tantivy-oracle")]
    fn trace_field_u64(line: &str, field: &str) -> Option<u64> {
        trace_field_text(line, field)?.parse().ok()
    }

    #[cfg(feature = "tantivy-oracle")]
    fn trace_field_text<'a>(line: &'a str, field: &str) -> Option<&'a str> {
        let prefix = format!("{field}=");
        let start = line.find(&prefix)?.saturating_add(prefix.len());
        let value = line.get(start..)?;
        if let Some(quoted) = value.strip_prefix('"') {
            return quoted.split_once('"').map(|(field_value, _)| field_value);
        }
        let end = value
            .find(|ch: char| ch.is_ascii_whitespace() || matches!(ch, ',' | '}'))
            .unwrap_or(value.len());
        value.get(..end)
    }

    #[cfg(feature = "tantivy-oracle")]
    fn trace_has_text_field(line: &str, field: &str, expected: &str) -> bool {
        trace_field_text(line, field) == Some(expected)
    }

    #[cfg(feature = "tantivy-oracle")]
    fn is_stage_close(line: &str, stage: &str) -> bool {
        if !line.contains(": close") {
            return false;
        }
        let Some(stage_position) = line.rfind(stage) else {
            return false;
        };
        frankensearch_quill::tracing_conventions::ALL_SPAN_NAMES
            .iter()
            .filter_map(|candidate| line.rfind(candidate))
            .all(|candidate_position| candidate_position <= stage_position)
    }

    #[cfg(feature = "tantivy-oracle")]
    fn assert_score_trace_contract(score: &str, context: &str) {
        let plan = trace_field_text(score, "plan")
            .unwrap_or_else(|| panic!("{context}: score span omitted plan: {score}"));
        let segments_touched = trace_field_u64(score, "segments_touched");
        let pruning_windows = trace_field_u64(score, "pruning_windows")
            .unwrap_or_else(|| panic!("{context}: score span omitted pruning windows: {score}"));
        let blocks_skipped = trace_field_u64(score, "blocks_skipped")
            .unwrap_or_else(|| panic!("{context}: score span omitted skipped blocks: {score}"));
        assert!(
            trace_field_u64(score, "candidate_docs").is_some(),
            "{context}: score span omitted candidate count: {score}",
        );
        assert_eq!(
            segments_touched,
            Some(1),
            "{context}: each score span must describe exactly one touched leaf: {score}",
        );
        match plan {
            "exhaustive" => {
                assert_eq!(
                    (pruning_windows, blocks_skipped),
                    (0, 0),
                    "{context}: exhaustive traversal reported pruning work: {score}",
                );
            }
            "max_score" => {
                assert!(
                    pruning_windows > 0 && blocks_skipped == 0,
                    "{context}: MaxScore counters are inconsistent: {score}",
                );
            }
            "block_max_wand" | "mixed_pruning" => {
                assert!(
                    pruning_windows > 0,
                    "{context}: block-pruned traversal reported no pruning window: {score}",
                );
            }
            other => panic!("{context}: unknown score plan {other:?}: {score}"),
        }
    }

    #[cfg(feature = "tantivy-oracle")]
    fn assert_harvested_query_trace_contract(logs: &str, golden: &E410RankGolden) {
        use frankensearch_quill::tracing_conventions::{ARGUS_COLLECT, ARGUS_PARSE, ARGUS_SCORE};

        for case in &golden.cases {
            let query_lines = logs
                .lines()
                .filter(|line| trace_has_text_field(line, "query_id", &case.query_id))
                .collect::<Vec<_>>();
            let context = format!(
                "corpus_hash={} query_seed={} query_id={}",
                golden.corpus_manifest_hash, golden.query_seed, case.query_id,
            );
            assert!(
                !query_lines.is_empty(),
                "{context}: no correlated Quill trace records",
            );
            assert!(
                query_lines.iter().any(|line| {
                    trace_field_u64(line, "query_seed") == Some(golden.query_seed)
                        && trace_has_text_field(line, "corpus_hash", &golden.corpus_manifest_hash)
                }),
                "{context}: trace omitted replay provenance",
            );
            let parse = query_lines
                .iter()
                .copied()
                .find(|line| is_stage_close(line, ARGUS_PARSE))
                .unwrap_or_else(|| panic!("{context}: missing parse close record"));
            assert!(
                parse.contains("query_root=")
                    && trace_field_u64(parse, "query_shape_hash").is_some()
                    && trace_field_u64(parse, "query_nodes").is_some_and(|count| count > 0)
                    && trace_field_u64(parse, "query_depth").is_some_and(|depth| depth > 0)
                    && parse.contains("duration_us="),
                "{context}: parse trace lacks tree shape or timing: {parse}",
            );
            let score_lines = query_lines
                .iter()
                .copied()
                .filter(|line| is_stage_close(line, ARGUS_SCORE))
                .collect::<Vec<_>>();
            assert!(
                !score_lines.is_empty(),
                "{context}: missing score close record",
            );
            for score in score_lines {
                assert_score_trace_contract(score, &context);
                assert!(
                    score.contains("duration_us="),
                    "{context}: score trace omitted timing: {score}",
                );
            }
            let collect = query_lines
                .iter()
                .copied()
                .find(|line| is_stage_close(line, ARGUS_COLLECT))
                .unwrap_or_else(|| panic!("{context}: missing collect close record"));
            assert!(
                trace_field_u64(collect, "segments_touched").is_some_and(|count| count > 0)
                    && collect.contains("duration_us="),
                "{context}: collect trace lacks touched leaves or timing: {collect}",
            );
        }
    }

    #[cfg(feature = "tantivy-oracle")]
    fn assert_scalar_g1a_trace_contract(logs: &str) {
        use frankensearch_quill::tracing_conventions::{
            ARGUS_COLLECT, ARGUS_PARSE, ARGUS_QUERY, ARGUS_SCORE, KEEPER_OPEN, KEEPER_SEAL,
            SCRIBE_ACCUMULATE, SCRIBE_FLUSH, SCRIBE_TOKENIZE,
        };

        let required = [
            SCRIBE_TOKENIZE,
            SCRIBE_ACCUMULATE,
            SCRIBE_FLUSH,
            KEEPER_SEAL,
            KEEPER_OPEN,
            ARGUS_PARSE,
            ARGUS_SCORE,
            ARGUS_COLLECT,
        ];
        for stage in required {
            let close = logs
                .lines()
                .find(|line| is_stage_close(line, stage))
                .unwrap_or_else(|| panic!("missing close record for {stage}: {logs}"));
            assert!(
                close.contains("duration_us="),
                "stage {stage} omitted explicit duration_us: {close}",
            );
            assert!(
                close.contains("time.busy=") && close.contains("time.idle="),
                "stage {stage} omitted subscriber timing: {close}",
            );
        }

        let accumulate = logs
            .lines()
            .find(|line| is_stage_close(line, SCRIBE_ACCUMULATE))
            .expect("accumulate close record");
        let used = trace_field_u64(accumulate, "arena_bytes_used_high_water")
            .expect("accumulate used high-water field");
        let reserved = trace_field_u64(accumulate, "arena_bytes_reserved_high_water")
            .expect("accumulate reserved high-water field");
        assert!(
            used > 0 && reserved >= used,
            "invalid arena high-water evidence: {accumulate}"
        );
        assert!(
            trace_field_u64(accumulate, "result_count").is_some_and(|count| count > 0),
            "accumulate span lacks a non-vacuous result count: {accumulate}",
        );

        let seal = logs
            .lines()
            .find(|line| is_stage_close(line, KEEPER_SEAL))
            .expect("seal close record");
        assert!(
            trace_field_u64(seal, "doc_count").is_some_and(|count| count > 0),
            "seal span lacks a non-vacuous document count: {seal}",
        );

        let parse = logs
            .lines()
            .find(|line| is_stage_close(line, ARGUS_PARSE))
            .expect("parse close record");
        assert!(
            parse.contains("query_root=")
                && trace_field_u64(parse, "query_shape_hash").is_some()
                && trace_field_u64(parse, "query_nodes").is_some_and(|count| count > 0)
                && trace_field_u64(parse, "query_depth").is_some_and(|depth| depth > 0),
            "parse span lacks a privacy-safe tree shape: {parse}",
        );

        let score = logs
            .lines()
            .find(|line| is_stage_close(line, ARGUS_SCORE))
            .expect("score close record");
        assert_score_trace_contract(score, "scalar G1a aggregate trace");

        let collect = logs
            .lines()
            .find(|line| is_stage_close(line, ARGUS_COLLECT))
            .expect("collect close record");
        assert!(
            trace_field_u64(collect, "segments_touched").is_some_and(|count| count > 0),
            "collect span lacks touched-segment evidence: {collect}",
        );

        let close_position = |stage: &str| {
            logs.lines()
                .position(|line| is_stage_close(line, stage))
                .unwrap_or_else(|| panic!("missing close position for {stage}"))
        };
        assert!(close_position(SCRIBE_TOKENIZE) < close_position(SCRIBE_ACCUMULATE));
        assert!(close_position(SCRIBE_FLUSH) < close_position(KEEPER_SEAL));
        let committed_open = logs
            .lines()
            .position(|line| {
                is_stage_close(line, KEEPER_OPEN) && line.contains("phase=\"open.committed\"")
            })
            .unwrap_or_else(|| panic!("missing post-seal committed-open close record: {logs}"));
        assert!(close_position(KEEPER_SEAL) < committed_open);
        assert!(close_position(ARGUS_PARSE) < close_position(ARGUS_SCORE));
        assert!(close_position(ARGUS_SCORE) < close_position(ARGUS_COLLECT));
        let paginated_queries = logs
            .lines()
            .filter(|line| {
                is_stage_close(line, ARGUS_QUERY)
                    && trace_has_text_field(line, "query_id", "paginated")
            })
            .collect::<Vec<_>>();
        assert!(
            paginated_queries.iter().any(|line| {
                trace_field_u64(line, "limit") == Some(7)
                    && trace_field_u64(line, "offset") == Some(17)
                    && line.contains("exact_count=false")
                    && trace_field_u64(line, "result_count").is_some_and(|count| count > 0)
            }),
            "live G1a trace did not execute Quill's count-free requested page: {logs}",
        );
        assert!(
            paginated_queries.iter().any(|line| {
                trace_field_u64(line, "limit") == Some(280)
                    && trace_field_u64(line, "offset") == Some(0)
                    && line.contains("exact_count=false")
                    && trace_field_u64(line, "result_count").is_some_and(|count| count > 0)
            }),
            "live G1a trace did not execute Quill's expanded ranked evidence: {logs}",
        );
        assert!(
            paginated_queries.iter().any(|line| {
                trace_field_u64(line, "limit") == Some(0)
                    && trace_field_u64(line, "offset") == Some(0)
                    && line.contains("exact_count=true")
                    && trace_field_u64(line, "result_count") == Some(0)
                    && trace_field_u64(line, "total_count").is_some_and(|count| count > 0)
            }),
            "live G1a trace did not execute Quill's independent count-only evidence: {logs}",
        );
        let golden: E410RankGolden = serde_json::from_str(E410_RANK_GOLDEN_JSON)
            .expect("parse committed E4.10 rank-list golden for trace contract");
        assert_harvested_query_trace_contract(logs, &golden);
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ScriptedBehavior {
        Exact,
        TieOrder,
        RankMismatch,
        OversizedQueryToken,
        DuplicateOversizedQueryToken,
        Error,
    }

    // Independent flags make failure-injection setup explicit in these tests.
    #[allow(clippy::struct_excessive_bools)]
    #[derive(Debug)]
    struct ScriptedEngine {
        descriptor: EngineDescriptor,
        semantic_contract: SemanticContract,
        behaviors: BTreeMap<String, ScriptedBehavior>,
        index_calls: AtomicUsize,
        abort_calls: AtomicUsize,
        observe_calls: AtomicUsize,
        indexed_document_count: u64,
        indexed_content_bytes: u64,
        indexed_payloads: Mutex<Vec<Vec<u8>>>,
        observed_queries: Mutex<Vec<GeneratedQueryCase>>,
        tamper_receipt: bool,
        fail_index_batch: bool,
        fail_begin: bool,
        panic_abort: bool,
        reported_doc_count_override: Option<u64>,
        drift_semantic_on_commit: bool,
    }

    impl ScriptedEngine {
        fn new(
            descriptor: EngineDescriptor,
            behaviors: BTreeMap<String, ScriptedBehavior>,
        ) -> Self {
            Self {
                descriptor,
                semantic_contract: semantic_contract(),
                behaviors,
                index_calls: AtomicUsize::new(0),
                abort_calls: AtomicUsize::new(0),
                observe_calls: AtomicUsize::new(0),
                indexed_document_count: 0,
                indexed_content_bytes: 0,
                indexed_payloads: Mutex::new(Vec::new()),
                observed_queries: Mutex::new(Vec::new()),
                tamper_receipt: false,
                fail_index_batch: false,
                fail_begin: false,
                panic_abort: false,
                reported_doc_count_override: None,
                drift_semantic_on_commit: false,
            }
        }

        fn with_tampered_receipt(mut self) -> Self {
            self.tamper_receipt = true;
            self
        }

        fn with_semantic_contract(mut self, semantic_contract: SemanticContract) -> Self {
            self.semantic_contract = semantic_contract;
            self
        }

        fn with_failing_index_batch(mut self) -> Self {
            self.fail_index_batch = true;
            self
        }

        fn with_failing_begin(mut self) -> Self {
            self.fail_begin = true;
            self
        }

        fn with_panicking_abort(mut self) -> Self {
            self.panic_abort = true;
            self
        }

        fn with_reported_doc_count(mut self, doc_count: u64) -> Self {
            self.reported_doc_count_override = Some(doc_count);
            self
        }

        fn with_semantic_drift_on_commit(mut self) -> Self {
            self.drift_semantic_on_commit = true;
            self
        }

        fn family(&self) -> EngineFamily {
            self.descriptor.family
        }

        fn behavior(&self, case_id: &str) -> ScriptedBehavior {
            self.behaviors
                .get(case_id)
                .copied()
                .unwrap_or(ScriptedBehavior::Exact)
        }

        fn observation(&self, query: &GeneratedQueryCase) -> EngineObservation {
            let behavior = self.behavior(&query.id);
            let (hits, ast_differences) = match behavior {
                ScriptedBehavior::Exact | ScriptedBehavior::Error => (Vec::new(), Vec::new()),
                ScriptedBehavior::TieOrder => {
                    let external_ids = match self.family() {
                        EngineFamily::Quill => ["alpha", "beta"],
                        EngineFamily::Tantivy => ["beta", "alpha"],
                    };
                    (
                        external_ids
                            .into_iter()
                            .enumerate()
                            .map(|(index, doc_id)| scripted_hit(self.family(), doc_id, index, 4.0))
                            .collect(),
                        Vec::new(),
                    )
                }
                ScriptedBehavior::RankMismatch => {
                    let doc_id = match self.family() {
                        EngineFamily::Quill => "subject-only",
                        EngineFamily::Tantivy => "oracle-only",
                    };
                    (
                        vec![scripted_hit(self.family(), doc_id, 0, 3.0)],
                        Vec::new(),
                    )
                }
                ScriptedBehavior::OversizedQueryToken
                | ScriptedBehavior::DuplicateOversizedQueryToken => {
                    let mut differences = if self.family() == EngineFamily::Quill {
                        vec![AstDifference {
                            kind: AstLoweringKind::OversizedQueryToken,
                            oracle: "BooleanQuery(TermQuery(content:oversized))".to_owned(),
                            subject: "MatchNone(oversized-query-token)".to_owned(),
                        }]
                    } else {
                        Vec::new()
                    };
                    if behavior == ScriptedBehavior::DuplicateOversizedQueryToken
                        && let Some(difference) = differences.first().cloned()
                    {
                        differences.push(difference);
                    }
                    (Vec::new(), differences)
                }
            };
            let match_count = if query.count_requested {
                CountState::Value(u64::try_from(hits.len()).unwrap_or(u64::MAX))
            } else {
                CountState::NotRequested
            };
            EngineObservation {
                hits,
                cutoff_tie_group: Vec::new(),
                cutoff_tie_complete: true,
                offset_tie_group: Vec::new(),
                offset_tie_complete: false,
                snippets: BTreeMap::new(),
                match_count,
                doc_count: self
                    .reported_doc_count_override
                    .unwrap_or(self.indexed_document_count),
                ast_differences,
            }
        }
    }

    impl DifferentialCampaignEngine for ScriptedEngine {
        fn descriptor(&self) -> EngineDescriptor {
            self.descriptor.clone()
        }

        fn semantic_contract(&self) -> SemanticContract {
            self.semantic_contract.clone()
        }

        fn begin_corpus<'a>(
            &'a mut self,
            _cx: &'a Cx,
            _manifest: &'a CorpusManifest,
            _semantic_contract: &'a SemanticContract,
        ) -> CampaignFuture<'a, ()> {
            Box::pin(async move {
                self.index_calls.fetch_add(1, Ordering::Relaxed);
                if self.fail_begin {
                    return Err(campaign_error("scripted begin failure"));
                }
                self.indexed_document_count = 0;
                self.indexed_content_bytes = 0;
                Ok(())
            })
        }

        fn index_batch<'a>(
            &'a mut self,
            _cx: &'a Cx,
            documents: &'a [GeneratedDocument],
        ) -> CampaignFuture<'a, ()> {
            Box::pin(async move {
                if self.fail_index_batch {
                    return Err(campaign_error("scripted index batch failure"));
                }
                self.indexed_document_count = self
                    .indexed_document_count
                    .checked_add(u64::try_from(documents.len()).unwrap_or(u64::MAX))
                    .ok_or_else(|| campaign_error("scripted document count overflow"))?;
                for document in documents {
                    self.indexed_content_bytes = self
                        .indexed_content_bytes
                        .checked_add(u64::try_from(document.content.len()).unwrap_or(u64::MAX))
                        .ok_or_else(|| campaign_error("scripted content byte count overflow"))?;
                }
                self.indexed_payloads
                    .lock()
                    .expect("indexed payload lock")
                    .push(serde_json::to_vec(documents)?);
                Ok(())
            })
        }

        fn commit_corpus<'a>(
            &'a mut self,
            _cx: &'a Cx,
            manifest: &'a CorpusManifest,
            semantic_contract: &'a SemanticContract,
        ) -> CampaignFuture<'a, EngineIndexReceipt> {
            Box::pin(async move {
                let mut receipt = EngineIndexReceipt {
                    corpus_manifest_hash: manifest.manifest_hash()?,
                    document_count: self.indexed_document_count,
                    total_content_bytes: self.indexed_content_bytes,
                    semantic_contract: semantic_contract.clone(),
                };
                if self.tamper_receipt {
                    receipt.document_count = receipt.document_count.saturating_add(1);
                }
                if self.drift_semantic_on_commit {
                    self.semantic_contract = SemanticContract::new("c".repeat(64), "d".repeat(64))?;
                }
                Ok(receipt)
            })
        }

        fn observe_generated<'a>(
            &'a mut self,
            _cx: &'a Cx,
            query: &'a GeneratedQueryCase,
            _evidence_case: &'a DifferentialCase,
        ) -> CampaignFuture<'a, EngineObservation> {
            Box::pin(async move {
                self.observe_calls.fetch_add(1, Ordering::Relaxed);
                self.observed_queries
                    .lock()
                    .expect("observed query lock")
                    .push(query.clone());
                if self.behavior(&query.id) == ScriptedBehavior::Error {
                    return Err(campaign_error("scripted query execution failure"));
                }
                Ok(self.observation(query))
            })
        }

        fn abort_corpus(&mut self) {
            self.abort_calls.fetch_add(1, Ordering::Relaxed);
            assert!(!self.panic_abort, "scripted abort panic");
        }
    }

    fn scripted_hit(family: EngineFamily, doc_id: &str, index: usize, score: f32) -> RankedHit {
        let ordinal = u32::try_from(index).unwrap_or(u32::MAX).saturating_add(1);
        let native_tie_key = match family {
            EngineFamily::Quill => NativeTieKey::QuillDocId { doc_id: ordinal },
            EngineFamily::Tantivy => NativeTieKey::TantivyDocAddress {
                segment_ord: 0,
                doc_id: ordinal,
            },
        };
        RankedHit {
            doc_id: doc_id.to_owned(),
            score_bits: score.to_bits(),
            native_tie_key,
        }
    }

    fn oversized_query_signature() -> String {
        mismatch_signature(
            RankClass::RankExact,
            &Divergence {
                class: DivergenceClass::OversizedQueryToken,
                pointer: "/comparison/subject/ast_differences/0".to_owned(),
                oracle: "BooleanQuery(TermQuery(content:oversized))".to_owned(),
                subject: "MatchNone(oversized-query-token)".to_owned(),
            },
        )
    }

    fn subject_descriptor() -> EngineDescriptor {
        EngineDescriptor {
            family: EngineFamily::Quill,
            implementation: "scripted-quill-subject".to_owned(),
            crate_version: env!("CARGO_PKG_VERSION").to_owned(),
            source_revision: "runner-test-subject".to_owned(),
            source_dirty: false,
            config_hash: "runner-test-quill-config".to_owned(),
        }
    }

    fn oracle_descriptor() -> EngineDescriptor {
        let version = oracle_version_contract().expect("oracle version contract");
        EngineDescriptor {
            family: EngineFamily::Tantivy,
            implementation: "frankensearch-lexical/tantivy-index".to_owned(),
            crate_version: version.lexical_package_version,
            source_revision: version.lexical_git_revision,
            source_dirty: false,
            config_hash: TANTIVY_ORACLE_CONFIG_HASH.to_owned(),
        }
    }

    struct Fixture {
        documents: Vec<GeneratedDocument>,
        corpus_manifest: CorpusManifest,
        corpus_hash: String,
        query_suite: GeneratedQuerySuite,
    }

    struct DriftingReplay {
        calls: AtomicUsize,
        first: Vec<GeneratedDocument>,
        second: Vec<GeneratedDocument>,
    }

    impl GeneratedCorpusReplay for DriftingReplay {
        fn replay(&self) -> Box<dyn Iterator<Item = GeneratedDocument> + Send + '_> {
            let call = self.calls.fetch_add(1, Ordering::Relaxed);
            if call == 0 {
                Box::new(self.first.clone().into_iter())
            } else {
                Box::new(self.second.clone().into_iter())
            }
        }
    }

    fn make_fixture() -> Fixture {
        let corpus = SyntheticCorpus::new(SyntheticCorpusSpec {
            seed: 0x6200,
            document_count: 12,
            vocabulary_size: 128,
            zipf_exponent: ZipfExponent::S11,
            max_document_bytes: 512,
        })
        .expect("synthetic corpus");
        let documents = corpus.iter().collect::<Vec<_>>();
        let corpus_manifest = corpus.manifest().expect("corpus manifest");
        let corpus_hash = corpus_manifest.manifest_hash().expect("corpus hash");
        let shared = SharedFixtureSuite::load().expect("shared fixtures");
        let query_suite = GeneratedQuerySuite::generate(
            QueryGeneratorSpec {
                seed: 0x6201,
                default_limit: 20,
                include_shared_relevance_queries: false,
            },
            &corpus_hash,
            &shared,
        )
        .expect("query suite");
        Fixture {
            documents,
            corpus_manifest,
            corpus_hash,
            query_suite,
        }
    }

    #[cfg(feature = "tantivy-oracle")]
    fn make_cass_activation_fixture() -> Fixture {
        let corpus = SyntheticCorpus::new(SyntheticCorpusSpec {
            seed: 0x6200,
            document_count: 24,
            vocabulary_size: 128,
            zipf_exponent: ZipfExponent::S11,
            max_document_bytes: 512,
        })
        .expect("CASS activation corpus");
        let documents = corpus.iter().collect::<Vec<_>>();
        let corpus_manifest = corpus.manifest().expect("CASS corpus manifest");
        let corpus_hash = corpus_manifest.manifest_hash().expect("CASS corpus hash");
        let query_suite = GeneratedQuerySuite::generate(
            QueryGeneratorSpec {
                seed: 0x6201,
                default_limit: 20,
                include_shared_relevance_queries: false,
            },
            &corpus_hash,
            &SharedFixtureSuite::load().expect("shared fixtures"),
        )
        .expect("CASS query suite");
        Fixture {
            documents,
            corpus_manifest,
            corpus_hash,
            query_suite,
        }
    }

    #[cfg(feature = "tantivy-oracle")]
    fn make_scalar_g1a_regression_fixture() -> Fixture {
        let shared = SharedFixtureSuite::load().expect("shared fixtures");
        let documents = shared
            .documents(crate::generator::SharedCorpusView::Core100)
            .to_vec();
        let corpus_manifest = shared
            .manifest(crate::generator::SharedCorpusView::Core100)
            .expect("shared corpus manifest");
        let corpus_hash = corpus_manifest.manifest_hash().expect("corpus hash");
        let query_suite = GeneratedQuerySuite::generate(
            QueryGeneratorSpec {
                seed: 0x6201,
                default_limit: 20,
                include_shared_relevance_queries: true,
            },
            &corpus_hash,
            &shared,
        )
        .expect("query suite");
        Fixture {
            documents,
            corpus_manifest,
            corpus_hash,
            query_suite,
        }
    }

    fn semantic_contract() -> SemanticContract {
        SemanticContract::shipping_default()
    }

    fn fixture_provenance(
        fixture: &Fixture,
        config: &CampaignConfig,
        semantic_contract: &SemanticContract,
        subject: &EngineDescriptor,
        oracle: &EngineDescriptor,
    ) -> CampaignProvenance {
        CampaignProvenance {
            subject_git_revision: subject.source_revision.clone(),
            subject_source_dirty: subject.source_dirty,
            oracle_git_revision: oracle.source_revision.clone(),
            oracle_source_dirty: oracle.source_dirty,
            cargo_lock_sha256: hash_workspace_lockfile().expect("Cargo.lock hash"),
            rustc_version_verbose: collect_rustc_verbose().expect("rustc provenance"),
            rust_toolchain_channel: collect_dated_toolchain_channel()
                .expect("dated nightly provenance"),
            unicode_version: format!(
                "{}.{}.{}",
                char::UNICODE_VERSION.0,
                char::UNICODE_VERSION.1,
                char::UNICODE_VERSION.2
            ),
            unicode_normalization_version: locked_crate_version("unicode-normalization")
                .expect("locked normalization version"),
            unicode_normalization_table_version: unicode_normalization_table_version(),
            query_generator_id: fixture.query_suite.manifest.generator_id.clone(),
            query_generator_schema_version: fixture.query_suite.manifest.schema_version,
            query_seed: fixture.query_suite.manifest.spec.seed,
            query_source_identity_sha256: fixture
                .query_suite
                .manifest
                .source_identity_sha256
                .clone(),
            query_profile_sha256: query_profile_sha256(
                &fixture.query_suite.manifest,
                &config.selection,
                semantic_contract,
            )
            .expect("query profile hash"),
            analyzer_contract_hash: semantic_contract.analyzer_contract_hash.clone(),
            schema_contract_hash: semantic_contract.schema_contract_hash.clone(),
            corpus_manifest_hash: fixture
                .corpus_manifest
                .manifest_hash()
                .expect("corpus manifest hash"),
            query_manifest_hash: fixture
                .query_suite
                .manifest
                .manifest_hash()
                .expect("query manifest hash"),
            corpus_seed: Some(0x6200),
        }
    }

    #[test]
    fn core_lexical_coverage_requires_nonempty_and_empty_search_paths() {
        let searched = ProbeCoverageCounts {
            success: 1,
            empty: 1,
            ..ProbeCoverageCounts::default()
        };
        let hydrated = ProbeCoverageCounts {
            success: 1,
            ..ProbeCoverageCounts::default()
        };
        let mut side = LexicalSideCoverageCounts {
            full_search: searched.clone(),
            fusion_candidates: searched,
            all_lexical_winners_hydration: hydrated.clone(),
            strict_hybrid_winners_hydration: hydrated.clone(),
            semantic_only_hydration: hydrated.clone(),
            mixed_winners_hydration: hydrated,
            metadata_deferred_cases: 0,
        };
        assert!(lexical_side_coverage_is_admissible(&side));

        side.full_search.empty = 0;
        assert!(
            !lexical_side_coverage_is_admissible(&side),
            "a campaign that never observes empty full search is vacuous"
        );
        side.full_search.empty = 1;
        side.fusion_candidates.empty = 0;
        assert!(
            !lexical_side_coverage_is_admissible(&side),
            "a campaign that never observes empty candidates is vacuous"
        );
    }

    #[test]
    fn core_lexical_coverage_requires_every_deferred_metadata_restoration_shape() {
        let searched = ProbeCoverageCounts {
            success: 1,
            empty: 1,
            ..ProbeCoverageCounts::default()
        };
        let restored = ProbeCoverageCounts {
            restoration: 1,
            ..ProbeCoverageCounts::default()
        };
        let semantic_control = ProbeCoverageCounts {
            success: 1,
            ..ProbeCoverageCounts::default()
        };
        let side = LexicalSideCoverageCounts {
            full_search: searched.clone(),
            fusion_candidates: searched,
            all_lexical_winners_hydration: restored.clone(),
            strict_hybrid_winners_hydration: restored.clone(),
            semantic_only_hydration: semantic_control,
            mixed_winners_hydration: restored,
            metadata_deferred_cases: 1,
        };
        assert!(lexical_side_coverage_is_admissible(&side));

        let mut missing_all = side.clone();
        missing_all.all_lexical_winners_hydration.restoration = 0;
        missing_all.all_lexical_winners_hydration.success = 1;
        let mut missing_strict = side.clone();
        missing_strict.strict_hybrid_winners_hydration.restoration = 0;
        missing_strict.strict_hybrid_winners_hydration.success = 1;
        let mut missing_mixed = side;
        missing_mixed.mixed_winners_hydration.restoration = 0;
        missing_mixed.mixed_winners_hydration.success = 1;
        for (shape, missing_shape) in [
            ("all lexical winners", missing_all),
            ("strict hybrid winners", missing_strict),
            ("mixed winners", missing_mixed),
        ] {
            assert!(
                !lexical_side_coverage_is_admissible(&missing_shape),
                "a deferred-metadata campaign must restore {shape}"
            );
        }
    }

    #[test]
    fn cass_selection_is_complete_profile_specific_and_manifest_ordered() {
        let fixture = make_fixture();
        let selected = CampaignSelection::CassSyntax
            .select(&fixture.query_suite.cases)
            .expect("CASS selection");
        assert!(
            selected.iter().all(|case| case.syntax == QuerySyntax::Cass),
            "the CASS campaign must not admit default-parser cases"
        );
        let selected_ids = selected
            .iter()
            .map(|case| case.id.as_str())
            .collect::<Vec<_>>();
        for required in [
            "boolean-cass",
            "boolean-cass-and",
            "boolean-cass-or",
            "boolean-cass-not",
            "glob-exact",
            "glob-prefix",
            "glob-suffix",
            "glob-substring",
            "glob-complex",
            "range-inclusive",
            "range-from",
            "range-to",
            "filter-agent",
            "filter-workspace",
            "filter-local",
            "filter-remote",
            "filter-source-id",
            "filter-combined",
        ] {
            assert!(
                selected_ids.contains(&required),
                "the CASS campaign dropped required case {required}"
            );
        }
        assert_eq!(
            selected
                .iter()
                .find(|case| case.id == "boolean-cass")
                .expect("CASS precedence case")
                .query,
            "auth OR token cache"
        );
        let manifest_positions = selected_ids
            .iter()
            .map(|id| {
                fixture
                    .query_suite
                    .cases
                    .iter()
                    .position(|case| case.id == *id)
                    .expect("selected case is in manifest")
            })
            .collect::<Vec<_>>();
        assert!(
            manifest_positions.windows(2).all(|pair| pair[0] < pair[1]),
            "CASS selection must retain manifest order"
        );
    }

    #[test]
    fn cass_semantic_contract_is_canonical_and_profile_distinct() {
        let contract = SemanticContract::cass();
        contract.validate().expect("canonical CASS hashes");
        assert_eq!(
            contract.analyzer_contract_hash,
            sha256_text(CASS_ANALYZER_CONTRACT_PREIMAGE)
        );
        assert_eq!(
            contract.schema_contract_hash,
            sha256_text(CASS_SCHEMA_CONTRACT_PREIMAGE)
        );
        assert_ne!(contract, SemanticContract::scalar_g1a());
        assert_ne!(contract, SemanticContract::shipping_default());
    }

    #[test]
    fn divergence_register_accepts_only_reviewed_semantic_taxonomy() {
        let entry = |class| DivergenceRegisterEntry {
            id: "DIV-999".to_owned(),
            class,
            fixture_id: "reviewed-fixture".to_owned(),
            mismatch_signatures: vec!["0".repeat(64)],
            decision: DivergenceRegisterDecision::Accept,
            root_cause: "reviewed root cause".to_owned(),
            consumer_impact: "reviewed consumer impact".to_owned(),
            reviewer: "second-agent".to_owned(),
            reviewed_at: "2026-07-26".to_owned(),
        };
        for class in [
            DivergenceClass::SnippetWindow,
            DivergenceClass::GlobExpansionLimit,
            DivergenceClass::QueryCanonicalization,
            DivergenceClass::OracleBug,
            DivergenceClass::StatsSemantics,
            DivergenceClass::UnicodeEdge,
            DivergenceClass::OversizedQueryToken,
        ] {
            entry(class)
                .validate()
                .unwrap_or_else(|error| panic!("{class:?} must be registerable: {error}"));
        }
        assert!(
            entry(DivergenceClass::RankMismatch).validate().is_err(),
            "generic result mismatch must never become a register wildcard"
        );
        assert!(
            entry(DivergenceClass::PostingRecordSemantics)
                .validate()
                .is_err(),
            "posting record semantics are fix-only and must never become an accepted wildcard"
        );
    }

    #[test]
    fn divergence_ledger_schema_fixture_round_trips() {
        let schema: serde_json::Value =
            serde_json::from_str(DIVERGENCE_REGISTER_SCHEMA_JSON).expect("parse register schema");
        let fixture: serde_json::Value =
            serde_json::from_str(DIVERGENCE_REGISTER_FIXTURE_JSON).expect("parse register fixture");
        let validator =
            jsonschema::draft202012::new(&schema).expect("compile Divergence Register schema");
        validator
            .validate(&fixture)
            .expect("fixture satisfies JSON Schema");

        let ledger = divergence_ledger_fixture();
        ledger
            .validate()
            .expect("fixture satisfies semantic contract");
        let round_trip =
            serde_json::to_value(&ledger).expect("serialize typed Divergence Register fixture");
        assert_eq!(round_trip, fixture);
        assert!(is_lower_sha256(
            &ledger.ledger_hash().expect("register ledger hash")
        ));

        let signature = "e".repeat(64);
        let census = ledger
            .require_terminal_census(&[signature])
            .expect("reviewed observed fixture is terminal");
        assert!(census.flip_ready);
        assert_eq!(census.mismatch_count, 1);
        assert_eq!(census.registered_mismatch_count, 1);
        let table = ledger.review_table().expect("render review table");
        assert!(table.contains(
            "| DIV-900 | oversized_query_token | divergence-register-contract-fixture | accepted | contract-fixture-reviewer |"
        ));

        let mut unknown_field = fixture;
        unknown_field
            .as_object_mut()
            .expect("fixture object")
            .insert("unreviewed".to_owned(), serde_json::Value::Bool(true));
        assert!(!validator.is_valid(&unknown_field));
        assert!(
            serde_json::from_value::<DivergenceRegisterLedger>(unknown_field).is_err(),
            "typed decoder must also reject unknown fields"
        );
    }

    #[test]
    fn divergence_ledger_enforces_append_only_corrections() {
        let ledger = divergence_ledger_fixture();
        let mut previous = ledger.clone();
        previous.events.truncate(2);
        previous.validate().expect("observation plus disposition");
        ledger
            .validate_append_only_successor(&previous)
            .expect("later prediction events are append-only");

        let mut successor = ledger.clone();
        let DivergenceRegisterEvent::Observation(corrected_observation) = ledger.events[0].clone()
        else {
            panic!("first fixture event is the observation");
        };
        let mut corrected_observation = *corrected_observation;
        corrected_observation.header.sequence = 5;
        corrected_observation.header.supersedes = Some(1);
        corrected_observation.header.recorded_at = "2026-07-27T00:04:00Z".to_owned();
        corrected_observation.root_cause =
            "The reviewed correction clarifies the symmetric AST admission boundary.".to_owned();
        successor
            .events
            .push(DivergenceRegisterEvent::Observation(Box::new(
                corrected_observation,
            )));

        let DivergenceRegisterEvent::Disposition(mut corrected_disposition) =
            ledger.events[1].clone()
        else {
            panic!("second fixture event is the disposition");
        };
        corrected_disposition.header.sequence = 6;
        corrected_disposition.header.supersedes = Some(2);
        corrected_disposition.header.recorded_at = "2026-07-27T00:05:00Z".to_owned();
        successor
            .events
            .push(DivergenceRegisterEvent::Disposition(corrected_disposition));
        successor.validate().expect("corrected append-only ledger");
        successor
            .validate_append_only_successor(&ledger)
            .expect("correction appends observation and renewed review");

        let mut rewritten = successor.clone();
        let DivergenceRegisterEvent::Observation(observation) = &mut rewritten.events[0] else {
            panic!("first fixture event is the observation");
        };
        observation.root_cause = "rewritten history".to_owned();
        rewritten
            .validate()
            .expect("rewritten snapshot remains internally shaped");
        assert!(
            rewritten.validate_append_only_successor(&ledger).is_err(),
            "editing any prior event must fail the append-only proof"
        );

        let mut missing_review = ledger.clone();
        let DivergenceRegisterEvent::Observation(observation) = ledger.events[0].clone() else {
            panic!("first fixture event is the observation");
        };
        let mut observation = *observation;
        observation.header.sequence = 5;
        observation.header.supersedes = Some(1);
        observation.header.recorded_at = "2026-07-27T00:04:00Z".to_owned();
        missing_review
            .events
            .push(DivergenceRegisterEvent::Observation(Box::new(observation)));
        assert!(
            missing_review.validate().is_err(),
            "corrected evidence requires a later disposition review"
        );
    }

    #[test]
    fn divergence_ledger_rejects_orphans_duplicate_signatures_and_unsafe_accepts() {
        let ledger = divergence_ledger_fixture();
        let mut duplicate = ledger.clone();
        let DivergenceRegisterEvent::Observation(observation) = ledger.events[0].clone() else {
            panic!("first fixture event is the observation");
        };
        let mut observation = *observation;
        observation.header.sequence = 5;
        observation.header.supersedes = None;
        observation.header.recorded_at = "2026-07-27T00:04:00Z".to_owned();
        observation.divergence_id = "DIV-901".to_owned();
        duplicate
            .events
            .push(DivergenceRegisterEvent::Observation(Box::new(observation)));
        let DivergenceRegisterEvent::Disposition(mut disposition) = ledger.events[1].clone() else {
            panic!("second fixture event is the disposition");
        };
        disposition.header.sequence = 6;
        disposition.header.supersedes = None;
        disposition.header.recorded_at = "2026-07-27T00:05:00Z".to_owned();
        disposition.divergence_id = "DIV-901".to_owned();
        duplicate
            .events
            .push(DivergenceRegisterEvent::Disposition(disposition));
        assert!(
            duplicate.validate().is_err(),
            "one mismatch signature cannot resolve to two active entries"
        );

        let mut unsafe_accept = ledger.clone();
        unsafe_accept.events.truncate(2);
        let DivergenceRegisterEvent::Observation(observation) = &mut unsafe_accept.events[0] else {
            panic!("first fixture event is the observation");
        };
        observation.class = DivergenceClass::RankMismatch;
        assert!(
            unsafe_accept.validate().is_err(),
            "raw result failures cannot be accepted as equivalence classes"
        );

        let mut self_reviewed = ledger;
        let DivergenceRegisterEvent::Disposition(disposition) = &mut self_reviewed.events[1] else {
            panic!("second fixture event is the disposition");
        };
        let DivergenceDisposition::Accepted { reviewer, .. } = &mut disposition.disposition else {
            panic!("fixture disposition is accepted");
        };
        *reviewer = "contract-fixture-author".to_owned();
        assert!(
            self_reviewed.validate().is_err(),
            "accepted divergences require a fresh-eyes reviewer"
        );
    }

    #[test]
    fn divergence_ledger_census_fails_closed_for_every_unresolved_state() {
        let ledger = divergence_ledger_fixture();
        let signature = "e".repeat(64);
        let unknown = "0".repeat(64);
        let unclassified = ledger
            .census(std::slice::from_ref(&unknown))
            .expect("census");
        assert!(!unclassified.flip_ready);
        assert_eq!(unclassified.unclassified_signatures, vec![unknown.clone()]);
        assert!(
            ledger
                .require_terminal_census(std::slice::from_ref(&unknown))
                .is_err()
        );

        let mut fixed = ledger.clone();
        let DivergenceRegisterEvent::Disposition(disposition) = &mut fixed.events[1] else {
            panic!("second fixture event is the disposition");
        };
        disposition.disposition = DivergenceDisposition::Fixed {
            fixing_commit: "3".repeat(40),
            regression_test: "runner::tests::fixed_divergence_regression".to_owned(),
            reviewer: "contract-fixture-reviewer".to_owned(),
            reviewed_at: "2026-07-27T00:00:30Z".to_owned(),
        };
        let fixed_census = fixed
            .census(std::slice::from_ref(&signature))
            .expect("fixed census");
        assert_eq!(
            fixed_census.fixed_regression_divergence_ids,
            vec!["DIV-900"]
        );
        assert!(!fixed_census.flip_ready);

        let mut blocking = ledger.clone();
        let DivergenceRegisterEvent::Disposition(disposition) = &mut blocking.events[1] else {
            panic!("second fixture event is the disposition");
        };
        disposition.disposition = DivergenceDisposition::Blocking {
            bead_id: "bd-quill-e6-gauntlet-scale-rm3q.8".to_owned(),
            rationale: "The mismatch remains an explicit flip blocker.".to_owned(),
            reviewer: "contract-fixture-reviewer".to_owned(),
            reviewed_at: "2026-07-27T00:00:30Z".to_owned(),
        };
        let blocking_census = blocking.census(&[]).expect("blocking census");
        assert_eq!(blocking_census.blocking_divergence_ids, vec!["DIV-900"]);
        assert!(!blocking_census.flip_ready);

        let mut unresolved_prediction = ledger;
        unresolved_prediction.events.truncate(3);
        unresolved_prediction
            .validate()
            .expect("declared prediction is valid during an active campaign");
        let prediction_census = unresolved_prediction
            .census(std::slice::from_ref(&signature))
            .expect("prediction census");
        assert_eq!(
            prediction_census.unresolved_prediction_ids,
            vec!["PRED-900"]
        );
        assert!(!prediction_census.flip_ready);
    }

    #[test]
    fn divergence_ledger_redaction_canaries_reject_plaintext_leaks() {
        let ledger = divergence_ledger_fixture();
        ledger
            .validate_redaction_canaries(&["TOP_SECRET_CANARY"])
            .expect("safe fixture does not contain the canary");

        let mut leaked = ledger.clone();
        let DivergenceRegisterEvent::Observation(observation) = &mut leaked.events[0] else {
            panic!("first fixture event is the observation");
        };
        observation.observed_behavior.push_str(" TOP_SECRET_CANARY");
        assert!(
            leaked
                .validate_redaction_canaries(&["TOP_SECRET_CANARY"])
                .is_err(),
            "source-sensitive plaintext must not survive committed serialization"
        );

        let DivergenceRegisterEvent::Observation(observation) = &mut leaked.events[0] else {
            panic!("first fixture event is the observation");
        };
        observation.observed_behavior =
            "The subject emits only a redacted diagnostic marker.".to_owned();
        observation.diagnostic.marker = "raw query text".to_owned();
        assert!(
            leaked.validate().is_err(),
            "diagnostic excerpts must use canonical redaction markers"
        );
    }

    fn runner(
        root: &std::path::Path,
        selection: CampaignSelection,
        registry: DivergenceRegistry,
    ) -> DifferentialCampaignRunner {
        DifferentialCampaignRunner::new(
            ArtifactStore::new(root),
            semantic_contract(),
            CampaignConfig {
                selection,
                index_batch_size: 5,
                ..CampaignConfig::default()
            },
            registry,
        )
        .expect("campaign runner")
    }

    #[cfg(feature = "tantivy-oracle")]
    async fn run_scalar_g1a_deterministic_regression(
        cx: &Cx,
        root: &std::path::Path,
        fixture: &Fixture,
    ) -> Result<CampaignReport, GauntletError> {
        // The fixed subject label and contract-sourced oracle revision make
        // repeated report bytes comparable. This self-contained test is
        // deterministic regression coverage, not independently observed live
        // Git provenance.
        let lexical_revision = oracle_version_contract()
            .expect("oracle version contract")
            .lexical_git_revision;
        let config = frankensearch_quill::QuillConfig {
            deterministic_ingest: true,
            ..frankensearch_quill::QuillConfig::default()
        };
        let mut subject = crate::engine::QuillSubject::in_memory(
            config,
            "g1a-deterministic-regression-not-live-provenance",
            false,
        )
        .expect("fresh scalar Quill subject");
        let mut oracle =
            crate::engine::TantivyOracle::in_memory_scalar_g1a(&lexical_revision, false)
                .expect("fresh scalar G1a Tantivy oracle");
        let campaign = DifferentialCampaignRunner::new(
            ArtifactStore::new(root),
            SemanticContract::scalar_g1a(),
            CampaignConfig {
                selection: CampaignSelection::DefaultSyntax,
                index_batch_size: 5,
                snippet_max_chars: None,
                ..CampaignConfig::default()
            },
            DivergenceRegistry::default(),
        )
        .expect("deterministic scalar G1a regression campaign");

        campaign
            .run(
                cx,
                "scalar-g1a-deterministic-regression",
                &mut subject,
                &mut oracle,
                &fixture.documents,
                &fixture.corpus_manifest,
                &fixture.query_suite,
            )
            .await
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    const UNION_HORIZON_QUERY: &str = "content:alpha OR content:beta OR content:gamma";
    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    const UNION_HORIZON_DOCUMENT_COUNT: usize = 9_001;
    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    const UNION_HORIZON_TWO_SEGMENT_SPLIT: usize = 257;
    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    const UNION_HORIZON_LATE_TIE_EXPANSION: u64 = 1;
    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    const UNION_HORIZON_MATRIX_TIE_EXPANSION: u64 = 4;
    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    const UNION_HORIZON_RANKED_DOCUMENT_TOKENS: usize = 128;

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    const UNION_HORIZON_ARTIFACT_SCHEMA_VERSION: &str =
        "frankensearch.salej-union-horizon-diagnostic.v3";
    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    const UNION_HORIZON_ARTIFACT_HASH_DOMAIN: &[u8] =
        b"frankensearch/quill/salej-union-horizon-diagnostic/v3\0";
    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    const UNION_HORIZON_COMPLETION_SCHEMA_VERSION: &str =
        "frankensearch.salej-union-horizon-diagnostic-completion.v2";
    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    const UNION_HORIZON_COMPLETION_HASH_DOMAIN: &[u8] =
        b"frankensearch/quill/salej-union-horizon-diagnostic-completion/v2\0";

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum UnionHorizonSegmentLayout {
        Single,
        Two,
        Eight,
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    impl UnionHorizonSegmentLayout {
        const ALL: [Self; 3] = [Self::Single, Self::Two, Self::Eight];

        const fn label(self) -> &'static str {
            match self {
                Self::Single => "single",
                Self::Two => "two",
                Self::Eight => "eight",
            }
        }

        const fn expected_execution_mode(
            self,
        ) -> frankensearch_quill::ConformancePruningExecutionMode {
            match self {
                Self::Single | Self::Two => {
                    frankensearch_quill::ConformancePruningExecutionMode::Serial
                }
                Self::Eight => frankensearch_quill::ConformancePruningExecutionMode::Rayon,
            }
        }

        fn ranges(self) -> Vec<std::ops::Range<usize>> {
            let boundaries: &[usize] = match self {
                Self::Single => &[0, UNION_HORIZON_DOCUMENT_COUNT],
                Self::Two => &[
                    0,
                    UNION_HORIZON_TWO_SEGMENT_SPLIT,
                    UNION_HORIZON_DOCUMENT_COUNT,
                ],
                // Seven uniquely sized prefix leaves retain the exact
                // 257-document prefix and 8,744-document target tail while
                // crossing the shipping eight-segment fan-out threshold
                // naturally. Unique cardinalities make every Tantivy native
                // segment address independently resolvable.
                Self::Eight => &[
                    0,
                    1,
                    3,
                    6,
                    10,
                    15,
                    21,
                    UNION_HORIZON_TWO_SEGMENT_SPLIT,
                    UNION_HORIZON_DOCUMENT_COUNT,
                ],
            };
            boundaries.windows(2).map(|pair| pair[0]..pair[1]).collect()
        }

        fn range_for_ordinal(self, ordinal: usize) -> std::ops::Range<usize> {
            self.ranges()
                .into_iter()
                .find(|range| range.contains(&ordinal))
                .expect("UNION_HORIZON ordinal belongs to one segment")
        }

        const fn target_segment_start(self) -> usize {
            match self {
                Self::Single => 0,
                Self::Two | Self::Eight => UNION_HORIZON_TWO_SEGMENT_SPLIT,
            }
        }
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct UnionHorizonBuildIdentity {
        source_git_revision: String,
        source_git_dirty: bool,
        source_verification: UnionHorizonSourceVerification,
        cargo_lock_sha256: String,
        rustc_version_verbose: String,
        target_triple: String,
        cargo_profile: String,
        enabled_features: Vec<String>,
        enabled_features_sha256: String,
        test_executable_sha256: String,
        test_executable_byte_len: u64,
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum UnionHorizonSourceVerification {
        /// `build.rs` rediscovered the exact repository root and verified any
        /// explicit identity against its live Git revision and dirty state.
        GitCheckoutVerified,
        /// A Git-less build accepted caller-supplied revision metadata only
        /// for diagnostic execution. This state can never publish evidence.
        ExplicitUnverified,
        /// Neither an exact checkout nor explicit diagnostic identity existed.
        Unavailable,
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct UnionHorizonOracleDependencyIdentity {
        tantivy_version: String,
        tantivy_checksum_sha256: String,
        lexical_package: String,
        lexical_package_version: String,
        pinned_lexical_contract_revision: String,
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct UnionHorizonTraceReceipt {
        limit: u64,
        segment_doc_start: u32,
        segment_doc_count: u64,
        refills: Vec<frankensearch_quill::ConformancePruningRefillReceipt>,
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct UnionHorizonTracedHitReceipt {
        document_id: String,
        global_docid: u32,
        score_bits: u32,
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct UnionHorizonTracedResultReceipt {
        limit: u64,
        hits: Vec<UnionHorizonTracedHitReceipt>,
        total_count: Option<u64>,
        doc_count: u64,
        diagnostic_count: u64,
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct UnionHorizonTantivySegmentReceipt {
        segment_ord: u32,
        max_doc: u32,
        num_docs: u32,
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct UnionHorizonTopologyReceipt {
        quill_segment_doc_counts: Vec<u32>,
        tantivy_segments: Vec<UnionHorizonTantivySegmentReceipt>,
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct UnionHorizonProof {
        layout: UnionHorizonSegmentLayout,
        build_identity: UnionHorizonBuildIdentity,
        oracle_dependency: UnionHorizonOracleDependencyIdentity,
        comparisons: Vec<HarnessRun>,
        traced_results: Vec<UnionHorizonTracedResultReceipt>,
        target_traces: Vec<UnionHorizonTraceReceipt>,
        complete_pruning_traces: Vec<frankensearch_quill::ConformancePruningTraceReceipt>,
        topology: UnionHorizonTopologyReceipt,
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum UnionHorizonProofKind {
        LateWinner,
        TieMatrix,
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    impl UnionHorizonProofKind {
        const fn label(self) -> &'static str {
            match self {
                Self::LateWinner => "late-winner",
                Self::TieMatrix => "tie-matrix",
            }
        }
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn union_horizon_quill_config() -> frankensearch_quill::QuillConfig {
        frankensearch_quill::QuillConfig {
            deterministic_ingest: true,
            // The proof needs eight independently searchable sealed leaves to
            // exercise the shipping Rayon query branch. The default fanout of
            // eight would merge on the eighth commit; nine is the smallest
            // production-valid value that preserves this exact topology.
            tier_fanout: 9,
            ..frankensearch_quill::QuillConfig::default()
        }
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn union_horizon_expected_corpus_hash(proof_kind: UnionHorizonProofKind) -> &'static str {
        static LATE_WINNER: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        static TIE_MATRIX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        match proof_kind {
            UnionHorizonProofKind::LateWinner => LATE_WINNER
                .get_or_init(|| make_union_horizon_fixture().corpus_hash)
                .as_str(),
            UnionHorizonProofKind::TieMatrix => TIE_MATRIX
                .get_or_init(|| make_union_horizon_tie_fixture().corpus_hash)
                .as_str(),
        }
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct UnionHorizonDiagnosticArtifact {
        schema_version: String,
        run_id: String,
        proof_kind: UnionHorizonProofKind,
        proofs: Vec<UnionHorizonProof>,
        artifact_sha256: String,
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct PublishedUnionHorizonDiagnostic {
        path: std::path::PathBuf,
        raw_file_sha256: String,
        byte_len: u64,
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct UnionHorizonCompletionEntry {
        proof_kind: UnionHorizonProofKind,
        filename: String,
        semantic_sha256: String,
        raw_file_sha256: String,
        byte_len: u64,
        artifact: UnionHorizonDiagnosticArtifact,
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct UnionHorizonCompletionManifest {
        schema_version: String,
        run_id: String,
        build_identity: UnionHorizonBuildIdentity,
        artifacts: Vec<UnionHorizonCompletionEntry>,
        manifest_sha256: String,
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    impl UnionHorizonCompletionManifest {
        fn seal(run_id: String, mut artifacts: Vec<UnionHorizonCompletionEntry>) -> Self {
            artifacts.sort_by_key(|entry| entry.proof_kind);
            let mut manifest = Self {
                schema_version: UNION_HORIZON_COMPLETION_SCHEMA_VERSION.to_owned(),
                run_id,
                build_identity: union_horizon_build_identity(),
                artifacts,
                manifest_sha256: String::new(),
            };
            manifest.validate_structure();
            manifest.manifest_sha256 = manifest.preimage_sha256();
            manifest.verify();
            manifest
        }

        fn preimage_sha256(&self) -> String {
            let mut preimage = self.clone();
            preimage.manifest_sha256.clear();
            let bytes = serde_json::to_vec(&preimage)
                .expect("serialize canonical UNION_HORIZON completion preimage");
            let mut hasher = Sha256::new();
            hasher.update(UNION_HORIZON_COMPLETION_HASH_DOMAIN);
            hasher.update(bytes);
            lower_hex(&hasher.finalize())
        }

        fn validate_structure(&self) {
            assert_eq!(
                self.schema_version, UNION_HORIZON_COMPLETION_SCHEMA_VERSION,
                "UNION_HORIZON completion schema drifted",
            );
            assert!(
                !self.run_id.is_empty()
                    && self.run_id.bytes().all(|byte| {
                        byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')
                    }),
                "UNION_HORIZON completion run ID must be path-safe",
            );
            assert_eq!(
                self.build_identity,
                union_horizon_build_identity(),
                "UNION_HORIZON completion must bind the executing test binary",
            );
            assert!(
                union_horizon_identity_is_publishable(&self.build_identity),
                "UNION_HORIZON completion requires clean Git-verified source",
            );
            assert_eq!(
                self.artifacts
                    .iter()
                    .map(|entry| entry.proof_kind)
                    .collect::<Vec<_>>(),
                vec![
                    UnionHorizonProofKind::LateWinner,
                    UnionHorizonProofKind::TieMatrix,
                ],
                "UNION_HORIZON completion requires exactly both proof kinds",
            );
            for entry in &self.artifacts {
                assert!(
                    !entry.filename.is_empty()
                        && entry.filename.bytes().all(|byte| {
                            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.')
                        })
                        && entry.filename.ends_with(".json"),
                    "UNION_HORIZON completion contains an unsafe artifact filename",
                );
                assert_lower_hex(
                    &entry.semantic_sha256,
                    64,
                    "UNION_HORIZON completion semantic identity",
                );
                assert_lower_hex(
                    &entry.raw_file_sha256,
                    64,
                    "UNION_HORIZON completion raw identity",
                );
                assert!(entry.byte_len > 0);
                entry.artifact.verify();
                assert_eq!(
                    entry.artifact.run_id, self.run_id,
                    "UNION_HORIZON completion cannot embed a proof from another run",
                );
                assert_eq!(
                    entry.artifact.proof_kind, entry.proof_kind,
                    "UNION_HORIZON completion proof kind does not match its embedded artifact",
                );
                assert_eq!(
                    entry.artifact.artifact_sha256, entry.semantic_sha256,
                    "UNION_HORIZON completion semantic identity does not match its embedded artifact",
                );
                assert!(
                    entry
                        .artifact
                        .proofs
                        .iter()
                        .all(|proof| proof.build_identity == self.build_identity),
                    "UNION_HORIZON completion artifact does not bind the manifest executable",
                );
                let canonical_artifact_bytes = serde_json::to_vec_pretty(&entry.artifact)
                    .expect("serialize embedded UNION_HORIZON artifact");
                assert_eq!(
                    u64::try_from(canonical_artifact_bytes.len())
                        .expect("embedded UNION_HORIZON artifact length fits u64"),
                    entry.byte_len,
                    "UNION_HORIZON completion artifact length is not canonical",
                );
                assert_eq!(
                    sha256_hex(&canonical_artifact_bytes),
                    entry.raw_file_sha256,
                    "UNION_HORIZON completion raw identity is not canonical",
                );
                assert_eq!(
                    entry.filename,
                    format!(
                        "{}-{}-{}-{}.json",
                        entry.artifact.run_id,
                        entry.artifact.proof_kind.label(),
                        entry.artifact.artifact_sha256,
                        entry.raw_file_sha256,
                    ),
                    "UNION_HORIZON completion artifact filename is not canonical",
                );
            }
        }

        fn verify(&self) {
            self.validate_structure();
            assert_lower_hex(
                &self.manifest_sha256,
                64,
                "UNION_HORIZON completion manifest identity",
            );
            assert_eq!(
                self.manifest_sha256,
                self.preimage_sha256(),
                "UNION_HORIZON completion identity does not match its canonical preimage",
            );
        }
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    impl UnionHorizonDiagnosticArtifact {
        fn seal(
            run_id: String,
            proof_kind: UnionHorizonProofKind,
            proofs: Vec<UnionHorizonProof>,
        ) -> Self {
            let mut artifact = Self {
                schema_version: UNION_HORIZON_ARTIFACT_SCHEMA_VERSION.to_owned(),
                run_id,
                proof_kind,
                proofs,
                artifact_sha256: String::new(),
            };
            artifact.validate_structure();
            artifact.artifact_sha256 = artifact.preimage_sha256();
            artifact.verify();
            artifact
        }

        fn preimage_sha256(&self) -> String {
            let mut preimage = self.clone();
            // `run_id` is publication metadata, not semantic evidence.  Two
            // invocations that prove the same layouts against the same binary
            // must therefore have the same content identity.
            preimage.run_id.clear();
            preimage.artifact_sha256.clear();
            let bytes = serde_json::to_vec(&preimage)
                .expect("serialize canonical UNION_HORIZON artifact preimage");
            let mut hasher = Sha256::new();
            hasher.update(UNION_HORIZON_ARTIFACT_HASH_DOMAIN);
            hasher.update(bytes);
            lower_hex(&hasher.finalize())
        }

        fn validate_structure(&self) {
            assert_eq!(
                self.schema_version, UNION_HORIZON_ARTIFACT_SCHEMA_VERSION,
                "UNION_HORIZON artifact schema drifted",
            );
            assert!(
                !self.run_id.is_empty()
                    && self.run_id.bytes().all(|byte| {
                        byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')
                    }),
                "UNION_HORIZON run ID must be a non-empty path-safe ASCII token",
            );
            assert_eq!(
                self.proofs.len(),
                UnionHorizonSegmentLayout::ALL.len(),
                "UNION_HORIZON artifact must contain every segment layout",
            );
            let expected_limits = [1_u64, 20, 100];
            let expected_seed = match self.proof_kind {
                UnionHorizonProofKind::LateWinner => 0x6202_4096,
                UnionHorizonProofKind::TieMatrix => 0x6202_71E5,
            };
            let expected_fixture_prefix = match self.proof_kind {
                UnionHorizonProofKind::LateWinner => "union-horizon-k",
                UnionHorizonProofKind::TieMatrix => "union-horizon-ties-k",
            };
            let expected_corpus_hash = union_horizon_expected_corpus_hash(self.proof_kind);
            let expected_subject_config_hash =
                crate::engine::quill_config_hash(&union_horizon_quill_config());
            for (proof, expected_layout) in self.proofs.iter().zip(UnionHorizonSegmentLayout::ALL) {
                assert_eq!(
                    proof.layout, expected_layout,
                    "UNION_HORIZON artifact layouts must be complete and canonical",
                );
                let expected_doc_counts = proof
                    .layout
                    .ranges()
                    .iter()
                    .map(|range| {
                        u32::try_from(range.len())
                            .expect("UNION_HORIZON expected segment count fits u32")
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    proof.topology.quill_segment_doc_counts, expected_doc_counts,
                    "UNION_HORIZON Quill topology must equal the declared layout",
                );
                let mut expected_tantivy_doc_counts = expected_doc_counts.clone();
                expected_tantivy_doc_counts.sort_unstable_by(|left, right| right.cmp(left));
                assert!(
                    proof
                        .topology
                        .tantivy_segments
                        .iter()
                        .enumerate()
                        .all(|(ordinal, segment)| {
                            u32::try_from(ordinal).is_ok_and(|expected_ordinal| {
                                segment.segment_ord == expected_ordinal
                                    && segment.max_doc == segment.num_docs
                            })
                        }),
                    "UNION_HORIZON Tantivy topology must retain dense native ordinals with no deletes",
                );
                assert_eq!(
                    proof
                        .topology
                        .tantivy_segments
                        .iter()
                        .map(|segment| segment.num_docs)
                        .collect::<Vec<_>>(),
                    expected_tantivy_doc_counts,
                    "UNION_HORIZON Tantivy topology must equal the declared layout",
                );
                assert_eq!(
                    proof.comparisons.len(),
                    expected_limits.len(),
                    "UNION_HORIZON comparison matrix must contain every canonical limit",
                );
                assert_eq!(
                    proof.traced_results.len(),
                    expected_limits.len(),
                    "UNION_HORIZON traced-result matrix must contain every canonical limit",
                );
                assert_eq!(
                    proof.target_traces.len(),
                    expected_limits.len(),
                    "UNION_HORIZON target trace matrix must contain every canonical limit",
                );
                assert_eq!(
                    proof.complete_pruning_traces.len(),
                    expected_limits.len(),
                    "UNION_HORIZON complete trace matrix must contain every canonical limit",
                );
                for ((((run, traced_result), target_trace), trace), expected_limit) in proof
                    .comparisons
                    .iter()
                    .zip(&proof.traced_results)
                    .zip(&proof.target_traces)
                    .zip(&proof.complete_pruning_traces)
                    .zip(expected_limits)
                {
                    run.engines
                        .validate_gauntlet_contract()
                        .expect("UNION_HORIZON engine identity contract");
                    assert_eq!(run.engines.comparison_mode, ComparisonMode::CrossEngine);
                    assert_eq!(
                        run.engines.semantic_contract.as_ref(),
                        Some(&SemanticContract::scalar_g1a()),
                        "UNION_HORIZON engines must bind the scalar G1a semantic contract",
                    );
                    assert_eq!(
                        run.engines.subject.family,
                        EngineFamily::Quill,
                        "UNION_HORIZON subject must be Quill",
                    );
                    assert_eq!(
                        run.engines.subject.implementation,
                        "frankensearch-quill/scalar-index",
                    );
                    assert_eq!(
                        run.engines.subject.crate_version,
                        frankensearch_quill::FRANKENSEARCH_QUILL_CRATE_VERSION,
                        "UNION_HORIZON subject wrapper package version drifted",
                    );
                    assert_eq!(
                        run.engines.subject.config_hash, expected_subject_config_hash,
                        "UNION_HORIZON subject configuration drifted from deterministic fanout nine",
                    );
                    assert_eq!(
                        run.engines.oracle.family,
                        EngineFamily::Tantivy,
                        "UNION_HORIZON oracle must be Tantivy",
                    );
                    assert_eq!(
                        run.engines.oracle.implementation,
                        "frankensearch-lexical/tantivy-index",
                    );
                    assert_eq!(
                        run.engines.oracle.crate_version,
                        proof.oracle_dependency.lexical_package_version,
                        "UNION_HORIZON oracle wrapper package version drifted",
                    );
                    assert_eq!(
                        run.engines.oracle.config_hash, TANTIVY_ORACLE_CONFIG_HASH,
                        "UNION_HORIZON oracle configuration drifted",
                    );
                    assert_eq!(
                        run.engines.subject.source_revision,
                        proof.build_identity.source_git_revision,
                        "UNION_HORIZON subject descriptor must identify the compiled wrapper build",
                    );
                    assert_eq!(
                        run.engines.subject.source_dirty, proof.build_identity.source_git_dirty,
                        "UNION_HORIZON subject dirty state must identify the compiled wrapper build",
                    );
                    assert_eq!(
                        run.engines.oracle.source_revision,
                        proof.oracle_dependency.pinned_lexical_contract_revision,
                        "UNION_HORIZON oracle descriptor must retain the committed lexical baseline",
                    );
                    assert!(
                        !run.engines.oracle.source_dirty,
                        "UNION_HORIZON committed oracle baseline cannot be dirty",
                    );
                    assert_eq!(run.case.limit, expected_limit);
                    assert_eq!(
                        run.case.fixture_id,
                        format!("{expected_fixture_prefix}{expected_limit}"),
                    );
                    assert_eq!(run.case.query, UNION_HORIZON_QUERY);
                    assert_eq!(run.case.offset, 0);
                    assert_eq!(
                        run.case.tie_expansion_limit,
                        match self.proof_kind {
                            UnionHorizonProofKind::LateWinner => {
                                UNION_HORIZON_LATE_TIE_EXPANSION
                            }
                            UnionHorizonProofKind::TieMatrix => {
                                UNION_HORIZON_MATRIX_TIE_EXPANSION
                            }
                        },
                    );
                    assert!(!run.case.count_requested);
                    assert_eq!(run.case.snippet_max_chars, None);
                    assert_eq!(
                        run.case.metadata.generator_id.as_deref(),
                        Some(GENERATOR_ID),
                    );
                    assert_eq!(run.case.metadata.generator_seed, Some(expected_seed));
                    assert_eq!(
                        run.case.metadata.corpus_hash.as_deref(),
                        Some(expected_corpus_hash),
                        "UNION_HORIZON case must bind the exact deterministic 9,001-document fixture",
                    );
                    assert_eq!(run.comparator_config, ComparatorConfig::default());
                    run.case
                        .validate_observations(
                            &run.engines,
                            &run.comparison.subject,
                            &run.comparison.oracle,
                        )
                        .expect("UNION_HORIZON observation contract");
                    assert_eq!(
                        compare_observations(
                            run.comparison.subject.clone(),
                            run.comparison.oracle.clone(),
                            run.comparator_config,
                        )
                        .expect("recompute UNION_HORIZON comparison"),
                        run.comparison,
                        "UNION_HORIZON comparison must equal its sealed observations",
                    );
                    match self.proof_kind {
                        UnionHorizonProofKind::LateWinner => {
                            assert_union_horizon_late_comparison(
                                run,
                                expected_limit,
                                proof.layout,
                                &proof.topology.tantivy_segments,
                            );
                        }
                        UnionHorizonProofKind::TieMatrix => {
                            assert_union_horizon_tie_comparison(
                                run,
                                expected_limit,
                                proof.layout,
                                &proof.topology.tantivy_segments,
                            );
                        }
                    }
                    assert_union_horizon_traced_result(run, expected_limit, traced_result);
                    assert_union_horizon_complete_trace_semantics(proof.layout, trace);
                    assert_eq!(
                        trace.execution_mode(),
                        proof.layout.expected_execution_mode(),
                        "UNION_HORIZON artifact recorded the wrong shipping execution branch",
                    );
                    assert_eq!(
                        trace
                            .segments()
                            .iter()
                            .map(|segment| {
                                u32::try_from(segment.segment_doc_count())
                                    .expect("UNION_HORIZON segment count fits u32")
                            })
                            .collect::<Vec<_>>(),
                        proof.topology.quill_segment_doc_counts,
                        "UNION_HORIZON artifact trace is not a complete topology receipt",
                    );
                    assert!(
                        trace
                            .segments()
                            .iter()
                            .enumerate()
                            .all(|(ordinal, segment)| {
                                u64::try_from(ordinal)
                                    .is_ok_and(|expected| expected == segment.segment_ordinal())
                            }),
                        "UNION_HORIZON complete trace segment ordinals must be dense and ordered",
                    );
                    let target_segment = trace
                        .segments()
                        .last()
                        .expect("UNION_HORIZON trace contains the target segment");
                    assert_eq!(target_trace.limit, expected_limit);
                    assert_eq!(
                        target_trace.segment_doc_start,
                        u32::try_from(proof.layout.target_segment_start())
                            .expect("UNION_HORIZON target start fits u32"),
                    );
                    assert_eq!(
                        target_trace.segment_doc_count,
                        target_segment.segment_doc_count(),
                    );
                    assert_eq!(
                        target_trace.refills,
                        target_segment.refills(),
                        "UNION_HORIZON target trace must be an exact projection of the complete receipt",
                    );
                    let revalidated_target = match self.proof_kind {
                        UnionHorizonProofKind::LateWinner => union_horizon_late_trace_receipt(
                            expected_limit,
                            target_trace.segment_doc_start,
                            target_trace.segment_doc_count,
                            trace.segments(),
                        ),
                        UnionHorizonProofKind::TieMatrix => union_horizon_tie_trace_receipt(
                            expected_limit,
                            target_trace.segment_doc_start,
                            target_trace.segment_doc_count,
                            trace.segments(),
                        ),
                    };
                    assert_eq!(
                        *target_trace, revalidated_target,
                        "UNION_HORIZON target receipt failed semantic revalidation",
                    );
                }
            }
            let first = self
                .proofs
                .first()
                .expect("UNION_HORIZON artifact contains proofs");
            assert_eq!(
                first.build_identity,
                union_horizon_build_identity(),
                "UNION_HORIZON artifact must identify the verifier's compiled producer build",
            );
            assert_eq!(
                first.oracle_dependency,
                union_horizon_oracle_dependency_identity(),
                "UNION_HORIZON artifact must bind the compiled oracle dependency contract",
            );
            assert!(
                self.proofs.iter().all(|proof| {
                    proof.build_identity == first.build_identity
                        && proof.oracle_dependency == first.oracle_dependency
                }),
                "UNION_HORIZON artifact must bind one subject binary and one oracle dependency",
            );
            let first_engines = &first
                .comparisons
                .first()
                .expect("UNION_HORIZON proof contains comparisons")
                .engines;
            assert!(
                self.proofs.iter().all(|proof| {
                    proof
                        .comparisons
                        .iter()
                        .all(|run| run.engines == *first_engines)
                }),
                "UNION_HORIZON artifact must bind one exact engine pair across every layout and query",
            );
        }

        fn verify(&self) {
            self.validate_structure();
            assert!(
                self.artifact_sha256.len() == 64
                    && self
                        .artifact_sha256
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
                "UNION_HORIZON artifact identity must be lowercase SHA-256",
            );
            assert_eq!(
                self.artifact_sha256,
                self.preimage_sha256(),
                "UNION_HORIZON artifact identity does not match its canonical preimage",
            );
        }
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn union_horizon_ranked_content(gamma_frequency: usize) -> Vec<u8> {
        assert!(
            gamma_frequency < UNION_HORIZON_RANKED_DOCUMENT_TOKENS,
            "UNION_HORIZON ranked-document frequency must leave room for alpha",
        );
        let mut tokens = Vec::with_capacity(UNION_HORIZON_RANKED_DOCUMENT_TOKENS);
        tokens.push("alpha");
        tokens.extend(std::iter::repeat_n("gamma", gamma_frequency));
        tokens.extend(std::iter::repeat_n(
            "padding",
            UNION_HORIZON_RANKED_DOCUMENT_TOKENS - gamma_frequency - 1,
        ));
        assert_eq!(tokens.len(), UNION_HORIZON_RANKED_DOCUMENT_TOKENS);
        tokens.join(" ").into_bytes()
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn make_union_horizon_fixture() -> Fixture {
        let snapshot = RepositorySnapshot::from_entries(
            "union-horizon-late-winner",
            (0..UNION_HORIZON_DOCUMENT_COUNT).map(|ordinal| {
                let bytes = match ordinal {
                    // These 99 documents give the public top-100 comparison a
                    // strictly ordered prefix. Their fixed field length makes
                    // increasing gamma frequency the only score-changing
                    // variable, so a segment split cannot expose Tantivy's
                    // native DocAddress tie order against Quill's global
                    // document-id tie order.
                    0..=98 => union_horizon_ranked_content(ordinal + 10),
                    99..=4_095 => b"alpha beta".to_vec(),
                    4_096..=8_999 => b"alpha".to_vec(),
                    // The late winner retains the same fixed field length as
                    // the ranked prefix and has the unique largest gamma
                    // frequency. It remains beyond the first empty
                    // competitive window in both segment layouts.
                    9_000 => union_horizon_ranked_content(120),
                    _ => unreachable!("UNION_HORIZON fixture ordinal is bounded"),
                };
                RepositoryEntry {
                    relative_path: std::path::PathBuf::from(format!("docs/{ordinal:05}.txt")),
                    bytes,
                }
            }),
        )
        .expect("UNION_HORIZON repository snapshot");
        let corpus_hash = snapshot
            .manifest
            .manifest_hash()
            .expect("UNION_HORIZON corpus hash");
        let query_suite = GeneratedQuerySuite::from_cases(
            QueryGeneratorSpec {
                seed: 0x6202_4096,
                default_limit: 100,
                include_shared_relevance_queries: false,
            },
            &corpus_hash,
            [1_u64, 20, 100]
                .into_iter()
                .map(|limit| GeneratedQueryCase {
                    id: format!("union-horizon-k{limit}"),
                    syntax: QuerySyntax::Default,
                    query_kind: GeneratedQueryKind::Boolean,
                    query: UNION_HORIZON_QUERY.to_owned(),
                    limit,
                    offset: 0,
                    count_requested: false,
                    filters: crate::generator::GeneratedQueryFilters::default(),
                    expected_divergence: None,
                    source: "runner.rs UNION_HORIZON late-winner regression".to_owned(),
                })
                .collect(),
        )
        .expect("UNION_HORIZON query suite");
        Fixture {
            documents: snapshot.documents,
            corpus_manifest: snapshot.manifest,
            corpus_hash,
            query_suite,
        }
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn union_horizon_tie_frequency(ordinal: usize) -> Option<usize> {
        match ordinal {
            0 | 9_000 => Some(120),
            1..=17 | 19..=95 => Some(120 - ordinal),
            18 | 8_500 | 8_501 => Some(102),
            96 | 8_502 | 8_503 => Some(24),
            97 => Some(23),
            _ => None,
        }
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn make_union_horizon_tie_fixture() -> Fixture {
        let snapshot = RepositorySnapshot::from_entries(
            "union-horizon-tie-matrix",
            (0..UNION_HORIZON_DOCUMENT_COUNT).map(|ordinal| {
                let bytes = union_horizon_tie_frequency(ordinal).map_or_else(
                    || {
                        if ordinal < 4_096 {
                            b"alpha beta".to_vec()
                        } else {
                            b"alpha".to_vec()
                        }
                    },
                    union_horizon_ranked_content,
                );
                RepositoryEntry {
                    relative_path: std::path::PathBuf::from(format!("docs/{ordinal:05}.txt")),
                    bytes,
                }
            }),
        )
        .expect("UNION_HORIZON tie repository snapshot");
        let corpus_hash = snapshot
            .manifest
            .manifest_hash()
            .expect("UNION_HORIZON tie corpus hash");
        let query_suite = GeneratedQuerySuite::from_cases(
            QueryGeneratorSpec {
                seed: 0x6202_71E5,
                default_limit: 100,
                include_shared_relevance_queries: false,
            },
            &corpus_hash,
            [1_u64, 20, 100]
                .into_iter()
                .map(|limit| GeneratedQueryCase {
                    id: format!("union-horizon-ties-k{limit}"),
                    syntax: QuerySyntax::Default,
                    query_kind: GeneratedQueryKind::Boolean,
                    query: UNION_HORIZON_QUERY.to_owned(),
                    limit,
                    offset: 0,
                    count_requested: false,
                    filters: crate::generator::GeneratedQueryFilters::default(),
                    expected_divergence: None,
                    source: "runner.rs UNION_HORIZON cutoff-tie matrix".to_owned(),
                })
                .collect(),
        )
        .expect("UNION_HORIZON tie query suite");
        Fixture {
            documents: snapshot.documents,
            corpus_manifest: snapshot.manifest,
            corpus_hash,
            query_suite,
        }
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn union_horizon_target_refills(
        limit: u64,
        segment_doc_count: u64,
        segments: &[frankensearch_quill::ConformanceSegmentPruningReceipt],
    ) -> Vec<frankensearch_quill::ConformancePruningRefillReceipt> {
        let matching = segments
            .iter()
            .filter(|receipt| receipt.segment_doc_count().eq(&segment_doc_count))
            .collect::<Vec<_>>();
        assert_eq!(
            matching.len(),
            1,
            "UNION_HORIZON limit={limit} must expose exactly one target-segment receipt: \
             segment_doc_count={segment_doc_count} receipts={segments:#?}",
        );
        matching[0].refills().to_vec()
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn assert_union_horizon_refill_geometry(
        limit: u64,
        segment_doc_start: u32,
        refills: &[frankensearch_quill::ConformancePruningRefillReceipt],
    ) {
        assert_eq!(
            refills.len(),
            3,
            "UNION_HORIZON limit={limit} target segment must execute one exhaustive and two \
             competitive refills: {refills:#?}",
        );
        for (index, refill) in refills.iter().enumerate() {
            let expected_start = u64::from(segment_doc_start)
                + u64::try_from(index).expect("refill index fits u64") * 4_096;
            assert_eq!(u64::from(refill.window_start()), expected_start);
            assert_eq!(
                refill.ordinal(),
                u64::try_from(index).expect("refill index fits u64") + 1,
            );
            assert_eq!(
                refill.horizon_end(),
                u64::from(refill.window_start()) + 4_096,
            );
        }
        let late = refills[2];
        assert!(
            u64::from(late.window_start()) <= 9_000 && 9_000 < late.horizon_end(),
            "UNION_HORIZON final refill must contain global document 9000: {late:?}",
        );
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn union_horizon_late_trace_receipt(
        limit: u64,
        segment_doc_start: u32,
        segment_doc_count: u64,
        segments: &[frankensearch_quill::ConformanceSegmentPruningReceipt],
    ) -> UnionHorizonTraceReceipt {
        use frankensearch_quill::ConformancePruningStrategy;

        let refills = union_horizon_target_refills(limit, segment_doc_count, segments);
        assert_union_horizon_refill_geometry(limit, segment_doc_start, &refills);
        let initial = refills[0];
        assert_eq!(initial.strategy(), ConformancePruningStrategy::Exhaustive);
        assert_eq!(initial.cutoff_bits(), None);
        assert!(!initial.buffer_empty());
        assert!(initial.live_work_remains());

        let empty_competitive = refills[1];
        let late_competitive = refills[2];
        for refill in [empty_competitive, late_competitive] {
            assert_eq!(refill.strategy(), ConformancePruningStrategy::MaxScore);
            let cutoff = f32::from_bits(
                refill
                    .cutoff_bits()
                    .expect("competitive refill must bind cutoff bits"),
            );
            assert!(cutoff.is_finite() && cutoff > 0.0);
        }
        assert_eq!(empty_competitive.candidate_docs(), 0);
        assert!(empty_competitive.buffer_empty());
        assert!(empty_competitive.live_work_remains());
        assert_eq!(late_competitive.candidate_docs(), 1);
        assert!(!late_competitive.buffer_empty());
        assert!(!late_competitive.live_work_remains());
        let empty_cutoff =
            f32::from_bits(empty_competitive.cutoff_bits().expect("empty cutoff bits"));
        let late_cutoff = f32::from_bits(late_competitive.cutoff_bits().expect("late cutoff bits"));
        assert!(
            !matches!(
                late_cutoff.total_cmp(&empty_cutoff),
                std::cmp::Ordering::Less
            ),
            "UNION_HORIZON competitive cutoff regressed from {empty_cutoff:?} to {late_cutoff:?}",
        );

        UnionHorizonTraceReceipt {
            limit,
            segment_doc_start,
            segment_doc_count,
            refills,
        }
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn union_horizon_tie_trace_receipt(
        limit: u64,
        segment_doc_start: u32,
        segment_doc_count: u64,
        segments: &[frankensearch_quill::ConformanceSegmentPruningReceipt],
    ) -> UnionHorizonTraceReceipt {
        use frankensearch_quill::ConformancePruningStrategy;

        let refills = union_horizon_target_refills(limit, segment_doc_count, segments);
        assert_union_horizon_refill_geometry(limit, segment_doc_start, &refills);
        let initial = refills[0];
        assert_eq!(initial.strategy(), ConformancePruningStrategy::Exhaustive);
        assert_eq!(initial.cutoff_bits(), None);
        assert!(!initial.buffer_empty());
        assert!(initial.live_work_remains());

        let competitive = &refills[1..];
        let cutoffs = competitive
            .iter()
            .map(|refill| {
                assert_eq!(refill.strategy(), ConformancePruningStrategy::MaxScore);
                let cutoff = f32::from_bits(
                    refill
                        .cutoff_bits()
                        .expect("tie-matrix competitive refill must bind cutoff bits"),
                );
                assert!(cutoff.is_finite() && cutoff > 0.0);
                cutoff
            })
            .collect::<Vec<_>>();
        assert!(
            cutoffs
                .windows(2)
                .all(|pair| !matches!(pair[1].total_cmp(&pair[0]), std::cmp::Ordering::Less)),
            "UNION_HORIZON tie-matrix cutoff regressed: {cutoffs:?}",
        );
        assert!(competitive[0].live_work_remains());
        let late = competitive[1];
        assert!(u64::from(late.window_start()) >= 2 * 4_096);
        assert!(
            late.candidate_docs() > 0,
            "UNION_HORIZON tie-matrix final horizon must emit a competitive candidate",
        );
        assert!(!late.buffer_empty());
        assert!(!late.live_work_remains());

        UnionHorizonTraceReceipt {
            limit,
            segment_doc_start,
            segment_doc_count,
            refills,
        }
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn assert_union_horizon_complete_trace_semantics(
        layout: UnionHorizonSegmentLayout,
        trace: &frankensearch_quill::ConformancePruningTraceReceipt,
    ) {
        use frankensearch_quill::ConformancePruningStrategy;

        assert_eq!(trace.execution_mode(), layout.expected_execution_mode());
        let ranges = layout.ranges();
        assert_eq!(trace.segments().len(), ranges.len());
        for (segment_ordinal, (segment, range)) in trace.segments().iter().zip(ranges).enumerate() {
            assert_eq!(
                segment.segment_ordinal(),
                u64::try_from(segment_ordinal).expect("UNION_HORIZON segment ordinal fits u64"),
            );
            assert_eq!(
                segment.segment_doc_count(),
                u64::try_from(range.len()).expect("UNION_HORIZON segment length fits u64"),
            );
            let expected_refill_count = range.len().div_ceil(4_096);
            assert_eq!(
                segment.refills().len(),
                expected_refill_count,
                "UNION_HORIZON segment {segment_ordinal} must witness every union horizon",
            );
            for (refill_ordinal, refill) in segment.refills().iter().enumerate() {
                let window_start = range.start + refill_ordinal * 4_096;
                let window_doc_count = (range.end - window_start).min(4_096);
                assert_eq!(
                    refill.ordinal(),
                    u64::try_from(refill_ordinal + 1)
                        .expect("UNION_HORIZON refill ordinal fits u64"),
                );
                assert_eq!(
                    refill.window_start(),
                    u32::try_from(window_start).expect("UNION_HORIZON window start fits u32"),
                );
                assert_eq!(
                    refill.horizon_end(),
                    u64::try_from(window_start + 4_096)
                        .expect("UNION_HORIZON horizon end fits u64"),
                );
                assert!(
                    refill.candidate_docs()
                        <= u64::try_from(window_doc_count)
                            .expect("UNION_HORIZON window count fits u64"),
                    "UNION_HORIZON refill admitted more candidates than its physical window",
                );
                assert_eq!(
                    refill.buffer_empty(),
                    refill.candidate_docs() == 0,
                    "UNION_HORIZON direct-term fixture must bind candidate and buffer emptiness",
                );
                assert_eq!(
                    refill.live_work_remains(),
                    refill_ordinal + 1 < expected_refill_count,
                    "UNION_HORIZON refill liveness must end exactly with the segment",
                );
                if refill_ordinal == 0 {
                    assert_eq!(refill.strategy(), ConformancePruningStrategy::Exhaustive);
                    assert_eq!(refill.cutoff_bits(), None);
                    assert_eq!(
                        refill.candidate_docs(),
                        u64::try_from(window_doc_count)
                            .expect("UNION_HORIZON initial window count fits u64"),
                    );
                } else {
                    assert_eq!(refill.strategy(), ConformancePruningStrategy::MaxScore);
                    let cutoff = f32::from_bits(
                        refill
                            .cutoff_bits()
                            .expect("UNION_HORIZON competitive refill binds cutoff bits"),
                    );
                    assert!(cutoff.is_finite() && cutoff > 0.0);
                }
            }
        }
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn union_horizon_traced_result_receipt(
        limit: u64,
        traced: &frankensearch_quill::QuillSearchResult,
    ) -> UnionHorizonTracedResultReceipt {
        UnionHorizonTracedResultReceipt {
            limit,
            hits: traced
                .hits
                .iter()
                .map(|hit| UnionHorizonTracedHitReceipt {
                    document_id: hit.document_id.clone(),
                    global_docid: hit.global_docid,
                    score_bits: hit.score.to_bits(),
                })
                .collect(),
            total_count: traced.total_count,
            doc_count: traced.doc_count,
            diagnostic_count: u64::try_from(traced.diagnostics.len())
                .expect("UNION_HORIZON diagnostic count fits u64"),
        }
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn assert_union_horizon_traced_result(
        run: &HarnessRun,
        expected_limit: u64,
        traced: &UnionHorizonTracedResultReceipt,
    ) {
        assert_eq!(traced.limit, expected_limit);
        assert_eq!(traced.total_count, None);
        assert_eq!(
            traced.doc_count,
            u64::try_from(UNION_HORIZON_DOCUMENT_COUNT)
                .expect("UNION_HORIZON document count fits u64"),
        );
        assert_eq!(traced.diagnostic_count, 0);
        assert_eq!(
            traced.hits.len(),
            usize::try_from(expected_limit).expect("UNION_HORIZON limit fits usize"),
        );
        assert_eq!(traced.hits.len(), run.comparison.subject.hits.len());
        for (traced_hit, observed_hit) in traced.hits.iter().zip(&run.comparison.subject.hits) {
            assert_eq!(traced_hit.document_id, observed_hit.doc_id);
            assert_eq!(traced_hit.score_bits, observed_hit.score_bits);
            assert_eq!(
                observed_hit.native_tie_key,
                NativeTieKey::QuillDocId {
                    doc_id: traced_hit.global_docid,
                },
                "UNION_HORIZON traced global document ID must equal the sealed subject tie key",
            );
        }
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn union_horizon_document_ordinal(document_id: &str) -> usize {
        document_id
            .strip_prefix("repo:docs/")
            .and_then(|value| value.strip_suffix(".txt"))
            .expect("UNION_HORIZON document identity shape")
            .parse()
            .expect("UNION_HORIZON document ordinal")
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn union_horizon_build_identity() -> UnionHorizonBuildIdentity {
        static IDENTITY: std::sync::OnceLock<UnionHorizonBuildIdentity> =
            std::sync::OnceLock::new();
        IDENTITY
            .get_or_init(|| {
                let source_git_revision = env!("QUILL_PERF_PRODUCER_GIT_REVISION").to_owned();
                let source_git_dirty = match env!("QUILL_PERF_PRODUCER_GIT_DIRTY") {
                    "true" => true,
                    "false" => false,
                    other => {
                        assert!(
                            matches!(other, "true" | "false"),
                            "embedded UNION_HORIZON dirty state must be true or false",
                        );
                        true
                    }
                };
                let source_verification = match env!("QUILL_PERF_PRODUCER_SOURCE_VERIFICATION") {
                    "git_checkout_verified" => UnionHorizonSourceVerification::GitCheckoutVerified,
                    "explicit_unverified" => UnionHorizonSourceVerification::ExplicitUnverified,
                    "unavailable" => UnionHorizonSourceVerification::Unavailable,
                    other => {
                        panic!("unknown embedded UNION_HORIZON source verification mode {other:?}")
                    }
                };
                let cargo_lock_sha256 = env!("QUILL_PERF_PRODUCER_CARGO_LOCK_SHA256").to_owned();
                let rustc_version_verbose = String::from_utf8(decode_lower_hex(
                    env!("QUILL_PERF_PRODUCER_RUSTC_VV_HEX"),
                    "embedded rustc -Vv",
                ))
                .expect("embedded rustc -Vv identity must be UTF-8");
                let target_triple = env!("QUILL_PERF_PRODUCER_TARGET_TRIPLE").to_owned();
                let cargo_profile = env!("QUILL_PERF_PRODUCER_CARGO_PROFILE").to_owned();
                let enabled_features = env!("QUILL_PERF_PRODUCER_ENABLED_FEATURES")
                    .split(',')
                    .filter(|feature| !feature.is_empty())
                    .map(str::to_owned)
                    .collect::<Vec<_>>();
                let enabled_features_sha256 =
                    env!("QUILL_PERF_PRODUCER_ENABLED_FEATURES_SHA256").to_owned();
                let (test_executable_sha256, test_executable_byte_len) =
                    union_horizon_current_executable_identity();

                assert_lower_hex(
                    &source_git_revision,
                    40,
                    "embedded UNION_HORIZON Git revision",
                );
                assert_lower_hex(
                    &cargo_lock_sha256,
                    64,
                    "embedded UNION_HORIZON Cargo.lock identity",
                );
                assert_lower_hex(
                    &enabled_features_sha256,
                    64,
                    "embedded UNION_HORIZON feature-set identity",
                );
                assert_eq!(
                    enabled_features_sha256,
                    sha256_hex(enabled_features.join("\n").as_bytes()),
                    "embedded UNION_HORIZON feature set does not match its digest",
                );
                assert!(
                    enabled_features
                        .windows(2)
                        .all(|pair| pair[0].as_str() < pair[1].as_str()),
                    "embedded UNION_HORIZON feature set must be sorted and unique",
                );
                assert!(
                    enabled_features
                        .iter()
                        .any(|feature| feature == "pruning_conformance")
                        && enabled_features
                            .iter()
                            .any(|feature| feature == "tantivy_oracle"),
                    "UNION_HORIZON executable must enable both proof features",
                );
                assert!(
                    !rustc_version_verbose.is_empty()
                        && rustc_version_verbose.len() <= 16 * 1024
                        && rustc_version_verbose.contains("release:")
                        && rustc_version_verbose.contains("host:"),
                    "embedded UNION_HORIZON rustc identity is incomplete",
                );
                assert!(
                    !target_triple.is_empty() && !cargo_profile.is_empty(),
                    "embedded UNION_HORIZON target and Cargo profile must be present",
                );
                assert_lower_hex(
                    &test_executable_sha256,
                    64,
                    "UNION_HORIZON test executable identity",
                );
                assert!(
                    test_executable_byte_len > 0,
                    "UNION_HORIZON test executable must be nonempty",
                );

                UnionHorizonBuildIdentity {
                    source_git_revision,
                    source_git_dirty,
                    source_verification,
                    cargo_lock_sha256,
                    rustc_version_verbose,
                    target_triple,
                    cargo_profile,
                    enabled_features,
                    enabled_features_sha256,
                    test_executable_sha256,
                    test_executable_byte_len,
                }
            })
            .clone()
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn assert_lower_hex(value: &str, expected_len: usize, label: &str) {
        assert!(
            value.len() == expected_len
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
            "{label} must be {expected_len} lowercase hexadecimal characters",
        );
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn decode_lower_hex(value: &str, label: &str) -> Vec<u8> {
        assert!(
            value.len().is_multiple_of(2)
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
            "{label} must be canonical lowercase hexadecimal",
        );
        value
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let high = char::from(pair[0])
                    .to_digit(16)
                    .expect("validated hexadecimal nibble");
                let low = char::from(pair[1])
                    .to_digit(16)
                    .expect("validated hexadecimal nibble");
                u8::try_from(high * 16 + low).expect("two hexadecimal nibbles fit u8")
            })
            .collect()
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn union_horizon_current_executable_identity() -> (String, u64) {
        #[cfg(target_os = "linux")]
        let path = std::path::PathBuf::from("/proc/self/exe");
        #[cfg(not(target_os = "linux"))]
        let path =
            std::env::current_exe().expect("resolve current UNION_HORIZON test executable path");
        let mut file = std::fs::File::open(&path).unwrap_or_else(|error| {
            panic!(
                "open current UNION_HORIZON test executable {}: {error}",
                path.display(),
            )
        });
        let metadata = file
            .metadata()
            .expect("stat current UNION_HORIZON test executable");
        assert!(
            metadata.is_file(),
            "current UNION_HORIZON test executable must be a regular file",
        );
        let mut hasher = Sha256::new();
        let mut byte_len = 0_u64;
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let read = file
                .read(&mut buffer)
                .expect("stream current UNION_HORIZON test executable");
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            byte_len = byte_len
                .checked_add(u64::try_from(read).expect("read length fits u64"))
                .expect("UNION_HORIZON executable length cannot overflow u64");
        }
        assert_eq!(
            byte_len,
            metadata.len(),
            "UNION_HORIZON executable changed while its identity was captured",
        );
        (lower_hex(&hasher.finalize()), byte_len)
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn assert_union_horizon_executable_still_matches(identity: &UnionHorizonBuildIdentity) {
        let (sha256, byte_len) = union_horizon_current_executable_identity();
        assert_eq!(
            (sha256, byte_len),
            (
                identity.test_executable_sha256.clone(),
                identity.test_executable_byte_len,
            ),
            "the executing UNION_HORIZON binary changed after its build identity was sealed",
        );
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn union_horizon_identity_is_publishable(identity: &UnionHorizonBuildIdentity) -> bool {
        cfg!(target_os = "linux")
            && !identity.source_git_dirty
            && matches!(
                identity.source_verification,
                UnionHorizonSourceVerification::GitCheckoutVerified
            )
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn reject_union_horizon_oracle_identity_overrides(
        revision: Option<&std::ffi::OsStr>,
        dirty: Option<&std::ffi::OsStr>,
    ) -> Result<(), GauntletError> {
        if revision.is_some() || dirty.is_some() {
            return Err(campaign_error(
                "Salej UNION_HORIZON uses the committed oracle dependency contract; GAUNTLET_ORACLE_REVISION and GAUNTLET_ORACLE_DIRTY cannot override it",
            ));
        }
        Ok(())
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn union_horizon_validated_build_identity() -> UnionHorizonBuildIdentity {
        const REVISION_ENV: &str = "GAUNTLET_SUBJECT_REVISION";
        const DIRTY_ENV: &str = "GAUNTLET_SUBJECT_DIRTY";
        reject_union_horizon_oracle_identity_overrides(
            std::env::var_os("GAUNTLET_ORACLE_REVISION").as_deref(),
            std::env::var_os("GAUNTLET_ORACLE_DIRTY").as_deref(),
        )
        .unwrap_or_else(|error| panic!("invalid UNION_HORIZON legacy identity input: {error}"));
        let embedded = union_horizon_build_identity();
        let revision = match std::env::var(REVISION_ENV) {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => {
                panic!("{REVISION_ENV} must be valid Unicode")
            }
        };
        let dirty = match std::env::var(DIRTY_ENV) {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => {
                panic!("{DIRTY_ENV} must be valid Unicode")
            }
        };
        validate_union_horizon_runtime_identity_override(
            &embedded,
            revision.as_deref(),
            dirty.as_deref(),
        )
        .unwrap_or_else(|reason| panic!("invalid UNION_HORIZON runtime identity: {reason}"));
        embedded
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn validate_union_horizon_runtime_identity_override(
        embedded: &UnionHorizonBuildIdentity,
        revision: Option<&str>,
        dirty: Option<&str>,
    ) -> Result<(), String> {
        match (revision, dirty) {
            (None, None) => Ok(()),
            (Some(revision), Some(dirty)) => {
                let dirty = match dirty {
                    "true" => true,
                    "false" => false,
                    _ => {
                        return Err(
                            "GAUNTLET_SUBJECT_DIRTY must be exactly true or false".to_owned()
                        );
                    }
                };
                if revision != embedded.source_git_revision {
                    return Err(
                        "runtime revision does not equal the compiled producer revision".to_owned(),
                    );
                }
                if dirty != embedded.source_git_dirty {
                    return Err(
                        "runtime dirty state does not equal the compiled producer dirty state"
                            .to_owned(),
                    );
                }
                Ok(())
            }
            _ => Err(
                "GAUNTLET_SUBJECT_REVISION and GAUNTLET_SUBJECT_DIRTY must be supplied together"
                    .to_owned(),
            ),
        }
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn union_horizon_oracle_dependency_identity() -> UnionHorizonOracleDependencyIdentity {
        let contract = oracle_version_contract().expect("UNION_HORIZON oracle version contract");
        UnionHorizonOracleDependencyIdentity {
            tantivy_version: contract.tantivy_version,
            tantivy_checksum_sha256: contract.tantivy_checksum_sha256,
            lexical_package: contract.lexical_package,
            lexical_package_version: contract.lexical_package_version,
            pinned_lexical_contract_revision: contract.lexical_git_revision,
        }
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn union_horizon_observed_envelope(observation: &EngineObservation) -> BTreeMap<String, u32> {
        let mut envelope = BTreeMap::new();
        for hit in observation.hits.iter().chain(&observation.cutoff_tie_group) {
            if let Some(prior_bits) = envelope.insert(hit.doc_id.clone(), hit.score_bits) {
                assert_eq!(
                    prior_bits, hit.score_bits,
                    "UNION_HORIZON observation changed score bits across ranked and tie evidence",
                );
            }
        }
        envelope
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn assert_union_horizon_late_comparison(
        run: &HarnessRun,
        limit: u64,
        layout: UnionHorizonSegmentLayout,
        tantivy_segments: &[UnionHorizonTantivySegmentReceipt],
    ) {
        assert_eq!(run.comparison.status, ComparisonStatus::Exact);
        assert_eq!(run.comparison.rank_class, RankClass::RankExact);
        assert!(run.comparison.divergences.is_empty());
        assert_eq!(run.comparison.subject.match_count, CountState::NotRequested);
        assert_eq!(run.comparison.oracle.match_count, CountState::NotRequested);
        assert_eq!(
            run.comparison.subject.doc_count,
            u64::try_from(UNION_HORIZON_DOCUMENT_COUNT)
                .expect("UNION_HORIZON document count fits u64"),
        );
        assert_eq!(
            run.comparison.oracle.doc_count,
            u64::try_from(UNION_HORIZON_DOCUMENT_COUNT)
                .expect("UNION_HORIZON document count fits u64"),
        );
        assert!(run.comparison.subject.snippets.is_empty());
        assert!(run.comparison.oracle.snippets.is_empty());

        let subject_rows = run
            .comparison
            .subject
            .hits
            .iter()
            .map(|hit| (hit.doc_id.as_str(), hit.score_bits))
            .collect::<Vec<_>>();
        let oracle_rows = run
            .comparison
            .oracle
            .hits
            .iter()
            .map(|hit| (hit.doc_id.as_str(), hit.score_bits))
            .collect::<Vec<_>>();
        assert_eq!(
            subject_rows.len(),
            usize::try_from(limit).expect("UNION_HORIZON limit fits usize"),
        );
        assert_eq!(subject_rows, oracle_rows);
        assert_eq!(
            subject_rows.first().map(|(document_id, _)| *document_id),
            Some("repo:docs/09000.txt"),
            "UNION_HORIZON late winner must rank first",
        );
        assert!(
            subject_rows.windows(2).all(|rows| {
                matches!(
                    f32::from_bits(rows[0].1).total_cmp(&f32::from_bits(rows[1].1)),
                    std::cmp::Ordering::Greater
                )
            }),
            "UNION_HORIZON limit={limit} ranked prefix must have strictly descending unique scores",
        );
        if limit == 100 {
            assert_eq!(subject_rows.len(), 100);
            assert_eq!(subject_rows[0].0, "repo:docs/09000.txt");
            for (rank, ordinal) in (0..=98).rev().enumerate() {
                assert_eq!(
                    subject_rows[rank + 1].0,
                    format!("repo:docs/{ordinal:05}.txt"),
                    "UNION_HORIZON ranked-anchor identity drifted at rank {}",
                    rank + 1,
                );
            }
        }

        let subject_ties = run
            .comparison
            .subject
            .cutoff_tie_group
            .iter()
            .map(|hit| (hit.doc_id.as_str(), hit.score_bits))
            .collect::<Vec<_>>();
        let oracle_ties = run
            .comparison
            .oracle
            .cutoff_tie_group
            .iter()
            .map(|hit| (hit.doc_id.as_str(), hit.score_bits))
            .collect::<Vec<_>>();
        assert_eq!(subject_ties, oracle_ties);
        assert!(
            run.comparison.subject.cutoff_tie_complete && run.comparison.oracle.cutoff_tie_complete,
            "UNION_HORIZON limit={limit} requires complete cutoff-tie evidence",
        );
        assert_union_horizon_native_addresses(run, layout, tantivy_segments);
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn assert_union_horizon_tie_comparison(
        run: &HarnessRun,
        limit: u64,
        layout: UnionHorizonSegmentLayout,
        tantivy_segments: &[UnionHorizonTantivySegmentReceipt],
    ) {
        let expected_tie_count = match limit {
            1 => 2,
            _ => 3,
        };
        assert_eq!(run.comparison.subject.match_count, CountState::NotRequested,);
        assert_eq!(run.comparison.oracle.match_count, CountState::NotRequested,);
        assert_eq!(
            run.comparison.subject.doc_count,
            u64::try_from(UNION_HORIZON_DOCUMENT_COUNT)
                .expect("UNION_HORIZON document count fits u64"),
        );
        assert_eq!(
            run.comparison.oracle.doc_count,
            u64::try_from(UNION_HORIZON_DOCUMENT_COUNT)
                .expect("UNION_HORIZON document count fits u64"),
        );
        assert!(run.comparison.subject.snippets.is_empty());
        assert!(run.comparison.oracle.snippets.is_empty());
        assert_eq!(
            run.comparison.subject.hits.len(),
            usize::try_from(limit).expect("UNION_HORIZON limit fits usize"),
        );
        assert_eq!(
            run.comparison.oracle.hits.len(),
            usize::try_from(limit).expect("UNION_HORIZON limit fits usize"),
        );
        if matches!(layout, UnionHorizonSegmentLayout::Single) {
            assert_eq!(run.comparison.status, ComparisonStatus::Exact);
            assert_eq!(run.comparison.rank_class, RankClass::RankExact);
            assert!(run.comparison.divergences.is_empty());
            let subject_rows = run
                .comparison
                .subject
                .hits
                .iter()
                .map(|hit| (hit.doc_id.as_str(), hit.score_bits))
                .collect::<Vec<_>>();
            let oracle_rows = run
                .comparison
                .oracle
                .hits
                .iter()
                .map(|hit| (hit.doc_id.as_str(), hit.score_bits))
                .collect::<Vec<_>>();
            assert_eq!(subject_rows, oracle_rows);
        } else {
            assert_eq!(run.comparison.status, ComparisonStatus::Classified);
            assert_eq!(run.comparison.rank_class, RankClass::TieOrder);
            assert!(
                !run.comparison.divergences.is_empty()
                    && run.comparison.divergences.iter().all(|divergence| {
                        matches!(divergence.class, DivergenceClass::TieOrder)
                    }),
                "UNION_HORIZON two-segment tie proof admitted a non-TieOrder divergence: {:#?}",
                run.comparison,
            );
        }

        assert!(
            run.comparison.subject.cutoff_tie_complete && run.comparison.oracle.cutoff_tie_complete,
            "UNION_HORIZON tie matrix requires a lower-score completion sentinel",
        );
        assert_eq!(
            run.comparison.subject.cutoff_tie_group.len(),
            expected_tie_count,
        );
        assert_eq!(
            run.comparison.oracle.cutoff_tie_group.len(),
            expected_tie_count,
        );
        let subject_ties = run
            .comparison
            .subject
            .cutoff_tie_group
            .iter()
            .map(|hit| (hit.doc_id.as_str(), hit.score_bits))
            .collect::<BTreeMap<_, _>>();
        let oracle_ties = run
            .comparison
            .oracle
            .cutoff_tie_group
            .iter()
            .map(|hit| (hit.doc_id.as_str(), hit.score_bits))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(subject_ties, oracle_ties);
        assert_eq!(
            subject_ties
                .values()
                .copied()
                .collect::<BTreeSet<_>>()
                .len(),
            1,
            "UNION_HORIZON cutoff group must be an exact-score tie",
        );
        let tie_ordinals = subject_ties
            .keys()
            .map(|document_id| union_horizon_document_ordinal(document_id))
            .collect::<Vec<_>>();
        assert!(
            [1, 20, 100].contains(&limit),
            "UNION_HORIZON tie matrix has an unsupported limit {limit}",
        );
        let expected_tie_ordinals = match limit {
            1 => vec![0, 9_000],
            20 => vec![18, 8_500, 8_501],
            100 => vec![96, 8_502, 8_503],
            _ => Vec::new(),
        };
        assert_eq!(
            tie_ordinals, expected_tie_ordinals,
            "UNION_HORIZON cutoff tie membership drifted",
        );
        assert!(
            tie_ordinals.iter().any(|ordinal| *ordinal < 4_096)
                && tie_ordinals.iter().any(|ordinal| *ordinal > 4_095),
            "UNION_HORIZON tie group must cross the refill boundary",
        );
        if !matches!(layout, UnionHorizonSegmentLayout::Single) {
            assert!(
                tie_ordinals
                    .iter()
                    .map(|ordinal| layout.range_for_ordinal(*ordinal).start)
                    .collect::<BTreeSet<_>>()
                    .len()
                    > 1,
                "UNION_HORIZON tie group must cross the explicit segment split",
            );
        }
        assert_union_horizon_native_addresses(run, layout, tantivy_segments);
        assert_eq!(
            union_horizon_observed_envelope(&run.comparison.subject),
            union_horizon_observed_envelope(&run.comparison.oracle),
            "UNION_HORIZON ranked plus expanded tie envelope diverged",
        );
        let observed_envelope = union_horizon_observed_envelope(&run.comparison.subject);
        let mut score_cardinality = BTreeMap::<u32, usize>::new();
        for score_bits in observed_envelope.values() {
            *score_cardinality.entry(*score_bits).or_default() += 1;
        }
        let subject_cutoff_bits = run
            .comparison
            .subject
            .hits
            .last()
            .expect("UNION_HORIZON subject cutoff hit")
            .score_bits;
        let oracle_cutoff_bits = run
            .comparison
            .oracle
            .hits
            .last()
            .expect("UNION_HORIZON oracle cutoff hit")
            .score_bits;
        assert_eq!(subject_cutoff_bits, oracle_cutoff_bits);
        let subject_singleton_strata = run
            .comparison
            .subject
            .hits
            .iter()
            .filter(|hit| {
                score_cardinality
                    .get(&hit.score_bits)
                    .is_some_and(|count| matches!(count, &1))
            })
            .map(|hit| (&hit.doc_id, hit.score_bits))
            .collect::<Vec<_>>();
        let oracle_singleton_strata = run
            .comparison
            .oracle
            .hits
            .iter()
            .filter(|hit| {
                score_cardinality
                    .get(&hit.score_bits)
                    .is_some_and(|count| matches!(count, &1))
            })
            .map(|hit| (&hit.doc_id, hit.score_bits))
            .collect::<Vec<_>>();
        assert_eq!(
            subject_singleton_strata, oracle_singleton_strata,
            "UNION_HORIZON every singleton score stratum must remain RankExact",
        );
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn assert_union_horizon_native_addresses(
        run: &HarnessRun,
        layout: UnionHorizonSegmentLayout,
        tantivy_segments: &[UnionHorizonTantivySegmentReceipt],
    ) {
        for hit in run
            .comparison
            .subject
            .hits
            .iter()
            .chain(&run.comparison.subject.cutoff_tie_group)
        {
            let ordinal = union_horizon_document_ordinal(&hit.doc_id);
            let NativeTieKey::QuillDocId { doc_id } = &hit.native_tie_key else {
                assert!(
                    matches!(&hit.native_tie_key, NativeTieKey::QuillDocId { .. }),
                    "UNION_HORIZON subject must preserve its native global document ID",
                );
                continue;
            };
            assert_eq!(
                usize::try_from(*doc_id).expect("Quill global document ID fits usize"),
                ordinal,
                "UNION_HORIZON Quill global document ID must equal the external ordinal",
            );
        }

        for hit in run
            .comparison
            .oracle
            .hits
            .iter()
            .chain(&run.comparison.oracle.cutoff_tie_group)
        {
            let ordinal = union_horizon_document_ordinal(&hit.doc_id);
            let expected_range = layout.range_for_ordinal(ordinal);
            let NativeTieKey::TantivyDocAddress {
                segment_ord,
                doc_id,
            } = &hit.native_tie_key
            else {
                assert!(
                    matches!(&hit.native_tie_key, NativeTieKey::TantivyDocAddress { .. }),
                    "UNION_HORIZON oracle must preserve native Tantivy DocAddress",
                );
                continue;
            };
            let matching_segments = tantivy_segments
                .iter()
                .filter(|segment| {
                    usize::try_from(segment.num_docs)
                        .is_ok_and(|count| count == expected_range.len())
                })
                .collect::<Vec<_>>();
            assert_eq!(
                matching_segments.len(),
                1,
                "UNION_HORIZON explicit ingest range must resolve to exactly one Tantivy segment",
            );
            let expected_segment = matching_segments[0];
            assert_eq!(
                expected_segment.max_doc, expected_segment.num_docs,
                "UNION_HORIZON oracle topology must contain no deletes",
            );
            assert_eq!(
                usize::try_from(expected_segment.num_docs)
                    .expect("segment document count fits usize"),
                expected_range.len(),
                "UNION_HORIZON hit resolved through a segment with the wrong cardinality",
            );
            assert_eq!(
                segment_ord, &expected_segment.segment_ord,
                "UNION_HORIZON Tantivy native segment ordinal must match the explicit ingest range",
            );
            assert_eq!(
                usize::try_from(*doc_id).expect("Tantivy local document ID fits usize"),
                ordinal - expected_range.start,
                "UNION_HORIZON Tantivy local document ID must match the explicit ingest range",
            );
        }
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    async fn run_union_horizon_proof(
        cx: &Cx,
        fixture: &Fixture,
        layout: UnionHorizonSegmentLayout,
        proof_kind: UnionHorizonProofKind,
    ) -> UnionHorizonProof {
        let build_identity = union_horizon_validated_build_identity();
        let oracle_dependency = union_horizon_oracle_dependency_identity();
        let config = union_horizon_quill_config();
        let mut subject = crate::engine::QuillSubject::in_memory(
            config,
            build_identity.source_git_revision.clone(),
            build_identity.source_git_dirty,
        )
        .expect("fresh UNION_HORIZON Quill subject");
        assert_eq!(
            subject.config().tier_fanout,
            9,
            "UNION_HORIZON eight-leaf fixture must not trigger the default eight-way merge",
        );
        let mut oracle = crate::engine::TantivyOracle::in_memory_scalar_g1a(
            &oracle_dependency.pinned_lexical_contract_revision,
            false,
        )
        .expect("fresh UNION_HORIZON pinned Tantivy oracle");
        oracle
            .index()
            .oracle_disable_auto_merge(cx)
            .await
            .expect("disable UNION_HORIZON Tantivy auto-merge");
        subject
            .claim_fresh_campaign()
            .expect("claim UNION_HORIZON Quill subject");
        oracle
            .claim_fresh_campaign()
            .expect("claim UNION_HORIZON Tantivy oracle");

        let ranges = layout.ranges();
        for range in &ranges {
            let documents = fixture.documents[range.clone()]
                .iter()
                .cloned()
                .map(frankensearch_core::IndexableDocument::from)
                .collect::<Vec<_>>();
            subject
                .index_mut()
                .expect("UNION_HORIZON Quill index")
                .index_documents(cx, &documents)
                .await
                .expect("index UNION_HORIZON Quill segment");
            subject
                .index_mut()
                .expect("UNION_HORIZON Quill index")
                .commit(cx)
                .await
                .expect("commit UNION_HORIZON Quill segment");
            oracle
                .index()
                .index_documents(cx, &documents)
                .await
                .expect("index UNION_HORIZON Tantivy segment");
            oracle
                .index()
                .commit(cx)
                .await
                .expect("commit UNION_HORIZON Tantivy segment");
        }
        subject
            .mark_committed()
            .expect("commit UNION_HORIZON Quill campaign");
        oracle
            .mark_committed()
            .expect("commit UNION_HORIZON Tantivy campaign");

        let expected_segment_doc_counts = ranges
            .iter()
            .map(|range| u32::try_from(range.len()).expect("UNION_HORIZON segment length fits u32"))
            .collect::<Vec<_>>();
        let snapshot = subject
            .index()
            .expect("UNION_HORIZON Quill index")
            .snapshot();
        let actual_segment_doc_counts = snapshot
            .segments()
            .iter()
            .map(|segment| segment.doc_count())
            .collect::<Vec<_>>();
        assert_eq!(
            actual_segment_doc_counts, expected_segment_doc_counts,
            "UNION_HORIZON Quill segment shape drifted",
        );
        let tantivy_segments = oracle
            .index()
            .oracle_segment_layout()
            .expect("UNION_HORIZON Tantivy segment layout")
            .into_iter()
            .map(|segment| UnionHorizonTantivySegmentReceipt {
                segment_ord: segment.segment_ord,
                max_doc: segment.max_doc,
                num_docs: segment.num_docs,
            })
            .collect::<Vec<_>>();
        let mut expected_tantivy_doc_counts = expected_segment_doc_counts.clone();
        expected_tantivy_doc_counts.sort_unstable_by(|left, right| right.cmp(left));
        let actual_tantivy_doc_counts = tantivy_segments
            .iter()
            .map(|segment| {
                assert_eq!(
                    segment.max_doc, segment.num_docs,
                    "UNION_HORIZON oracle topology must contain no deletes",
                );
                segment.num_docs
            })
            .collect::<Vec<_>>();
        assert_eq!(
            actual_tantivy_doc_counts, expected_tantivy_doc_counts,
            "UNION_HORIZON Tantivy native searchable-segment order drifted",
        );
        assert!(
            tantivy_segments
                .iter()
                .enumerate()
                .all(|(ordinal, segment)| {
                    u32::try_from(ordinal).is_ok_and(|observed| observed == segment.segment_ord)
                }),
            "UNION_HORIZON Tantivy segment ordinals must be dense and ordered",
        );
        let topology = UnionHorizonTopologyReceipt {
            quill_segment_doc_counts: actual_segment_doc_counts,
            tantivy_segments: tantivy_segments.clone(),
        };
        if !matches!(layout, UnionHorizonSegmentLayout::Single) {
            assert_eq!(9_000_usize - layout.target_segment_start(), 8_743);
            assert!(9_000_usize - layout.target_segment_start() >= 2 * 4_096);
        }

        let target_id = fixture
            .documents
            .last()
            .expect("UNION_HORIZON target document")
            .id
            .as_str();
        let harness = crate::engine::DifferentialHarness::new(
            ComparisonMode::CrossEngine,
            ComparatorConfig::default(),
        );
        let mut comparisons = Vec::new();
        for query in &fixture.query_suite.cases {
            let case = DifferentialCase {
                fixture_id: query.id.clone(),
                query: query.query.clone(),
                limit: query.limit,
                offset: query.offset,
                tie_expansion_limit: match proof_kind {
                    UnionHorizonProofKind::LateWinner => UNION_HORIZON_LATE_TIE_EXPANSION,
                    UnionHorizonProofKind::TieMatrix => UNION_HORIZON_MATRIX_TIE_EXPANSION,
                },
                count_requested: query.count_requested,
                snippet_max_chars: None,
                metadata: DifferentialCaseMetadata {
                    generator_id: Some(GENERATOR_ID.to_owned()),
                    generator_seed: Some(fixture.query_suite.manifest.spec.seed),
                    corpus_hash: Some(fixture.corpus_hash.clone()),
                },
            };
            let mut run = harness
                .run(cx, &subject, &oracle, &case)
                .await
                .unwrap_or_else(|error| {
                    panic!(
                        "UNION_HORIZON layout={} limit={} failed: {error}",
                        layout.label(),
                        query.limit,
                    )
                });
            run.engines
                .bind_semantic_contract(SemanticContract::scalar_g1a())
                .expect("bind UNION_HORIZON scalar G1a contract");
            match proof_kind {
                UnionHorizonProofKind::LateWinner => {
                    assert_union_horizon_late_comparison(
                        &run,
                        query.limit,
                        layout,
                        &tantivy_segments,
                    );
                }
                UnionHorizonProofKind::TieMatrix => {
                    assert_union_horizon_tie_comparison(
                        &run,
                        query.limit,
                        layout,
                        &tantivy_segments,
                    );
                }
            }
            comparisons.push(run);
        }

        let segment_doc_start = u32::try_from(layout.target_segment_start())
            .expect("UNION_HORIZON segment start fits u32");
        let segment_doc_count =
            u64::try_from(UNION_HORIZON_DOCUMENT_COUNT - layout.target_segment_start())
                .expect("UNION_HORIZON target segment count fits u64");
        let mut traced_results = Vec::new();
        let mut target_traces = Vec::new();
        let mut complete_pruning_traces = Vec::new();
        for (query_ordinal, query) in fixture.query_suite.cases.iter().enumerate() {
            let index = subject.index().expect("UNION_HORIZON Quill index");
            let (traced, trace_receipt) = index
                .search_paginated_with_conformance_pruning_trace(
                    cx,
                    UNION_HORIZON_QUERY,
                    usize::try_from(query.limit).expect("UNION_HORIZON limit fits usize"),
                    0,
                    false,
                )
                .expect("trace UNION_HORIZON Quill search");
            let traced_result = union_horizon_traced_result_receipt(query.limit, &traced);
            assert_union_horizon_traced_result(
                &comparisons[query_ordinal],
                query.limit,
                &traced_result,
            );
            assert_union_horizon_complete_trace_semantics(layout, &trace_receipt);
            assert_eq!(
                trace_receipt.execution_mode(),
                layout.expected_execution_mode(),
                "UNION_HORIZON layout={} used the wrong shipping collection branch",
                layout.label(),
            );
            assert_eq!(
                trace_receipt
                    .segments()
                    .iter()
                    .map(|receipt| {
                        u32::try_from(receipt.segment_doc_count())
                            .expect("UNION_HORIZON receipt doc count fits u32")
                    })
                    .collect::<Vec<_>>(),
                topology.quill_segment_doc_counts,
                "UNION_HORIZON complete pruning receipt must match the full Quill topology",
            );
            assert_eq!(
                traced.hits.len(),
                usize::try_from(query.limit).expect("UNION_HORIZON limit fits usize"),
            );
            assert_eq!(
                traced.total_count, None,
                "UNION_HORIZON typed path proof must remain count-free",
            );
            if matches!(proof_kind, UnionHorizonProofKind::LateWinner) {
                assert_eq!(traced.hits[0].document_id, target_id);
                assert_eq!(traced.hits[0].global_docid, 9_000);
            }
            traced_results.push(traced_result);
            target_traces.push(match proof_kind {
                UnionHorizonProofKind::LateWinner => union_horizon_late_trace_receipt(
                    query.limit,
                    segment_doc_start,
                    segment_doc_count,
                    trace_receipt.segments(),
                ),
                UnionHorizonProofKind::TieMatrix => union_horizon_tie_trace_receipt(
                    query.limit,
                    segment_doc_start,
                    segment_doc_count,
                    trace_receipt.segments(),
                ),
            });
            complete_pruning_traces.push(trace_receipt);
        }

        UnionHorizonProof {
            layout,
            build_identity,
            oracle_dependency,
            comparisons,
            traced_results,
            target_traces,
            complete_pruning_traces,
            topology,
        }
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn union_horizon_artifact_run_id() -> String {
        const ARTIFACT_ROOT_ENV: &str = "GAUNTLET_UNION_HORIZON_ARTIFACT_ROOT";
        const RUN_ID_ENV: &str = "GAUNTLET_UNION_HORIZON_RUN_ID";
        let artifact_root = std::env::var_os(ARTIFACT_ROOT_ENV);
        let run_id = std::env::var(RUN_ID_ENV);
        match (artifact_root, run_id) {
            (Some(_), Ok(run_id)) => run_id,
            (None, Err(std::env::VarError::NotPresent)) => {
                let identity = union_horizon_build_identity();
                format!(
                    "local-{}",
                    identity
                        .source_git_revision
                        .get(..12)
                        .expect("validated UNION_HORIZON revision has 12 characters"),
                )
            }
            (Some(_), Err(error)) => {
                panic!(
                    "{RUN_ID_ENV} must be valid Unicode when {ARTIFACT_ROOT_ENV} is configured: {error}",
                )
            }
            (None, Ok(_) | Err(std::env::VarError::NotUnicode(_))) => {
                panic!("{ARTIFACT_ROOT_ENV} must be configured when {RUN_ID_ENV} is present",)
            }
        }
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn decode_union_horizon_artifact_bytes(
        bytes: &[u8],
        expected_raw_file_sha256: &str,
    ) -> UnionHorizonDiagnosticArtifact {
        assert!(
            expected_raw_file_sha256.len() == 64
                && expected_raw_file_sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
            "UNION_HORIZON raw-file identity must be lowercase SHA-256",
        );
        assert_eq!(
            sha256_hex(bytes),
            expected_raw_file_sha256,
            "UNION_HORIZON uploaded bytes do not match their publication receipt",
        );
        let artifact: UnionHorizonDiagnosticArtifact =
            serde_json::from_slice(bytes).expect("decode published UNION_HORIZON artifact");
        let canonical_bytes = serde_json::to_vec_pretty(&artifact)
            .expect("re-encode canonical UNION_HORIZON artifact bytes");
        assert_eq!(
            bytes, canonical_bytes,
            "UNION_HORIZON strict decoder requires the publisher's exact canonical bytes; unknown, duplicate, reordered, or alternate-spelling JSON is inadmissible",
        );
        artifact.verify();
        artifact
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn assert_union_horizon_artifact_rejected(
        artifact: &UnionHorizonDiagnosticArtifact,
        mutate: impl FnOnce(&mut UnionHorizonDiagnosticArtifact),
    ) {
        let mut tampered = artifact.clone();
        mutate(&mut tampered);
        assert_ne!(
            tampered, *artifact,
            "UNION_HORIZON hostile mutation must change the artifact",
        );
        tampered.artifact_sha256 = tampered.preimage_sha256();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tampered.verify())).is_err(),
            "tampered UNION_HORIZON artifact unexpectedly verified",
        );
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn persist_union_horizon_artifact(
        artifact: &UnionHorizonDiagnosticArtifact,
    ) -> Option<PublishedUnionHorizonDiagnostic> {
        const ARTIFACT_ROOT_ENV: &str = "GAUNTLET_UNION_HORIZON_ARTIFACT_ROOT";
        let root = std::env::var_os(ARTIFACT_ROOT_ENV).map(std::path::PathBuf::from)?;
        assert_union_horizon_executable_still_matches(
            &artifact
                .proofs
                .first()
                .expect("UNION_HORIZON artifact contains proofs")
                .build_identity,
        );
        assert!(
            artifact
                .proofs
                .first()
                .is_some_and(|proof| union_horizon_identity_is_publishable(&proof.build_identity)),
            "persisted UNION_HORIZON evidence requires a clean Git-verified compiled producer build",
        );
        assert!(
            root.is_relative()
                && root.starts_with("target/coverage")
                && !root
                    .components()
                    .any(|component| component == std::path::Component::ParentDir),
            "UNION_HORIZON artifacts must use a relative target/coverage path so RCH returns them",
        );
        let bytes =
            serde_json::to_vec_pretty(artifact).expect("serialize sealed UNION_HORIZON artifact");
        let raw_file_sha256 = sha256_hex(&bytes);
        let filename = format!(
            "{}-{}-{}-{}.json",
            artifact.run_id,
            artifact.proof_kind.label(),
            artifact.artifact_sha256,
            raw_file_sha256,
        );
        let target_name = std::ffi::OsString::from(&filename);
        let temporary_name =
            std::ffi::OsString::from(format!(".tmp-{}-{filename}", std::process::id()));
        let target = root.join(&target_name);
        let directory = crate::artifact::PinnedDirectory::ensure_path(&root)
            .expect("pin UNION_HORIZON artifact directory without following symlinks");
        directory
            .publish_unique_no_clobber(&temporary_name, &target_name, &bytes)
            .unwrap_or_else(|error| {
                panic!(
                    "atomically publish UNION_HORIZON artifact without replacing {}: {error}",
                    target.display(),
                )
            });
        let published_bytes = directory
            .read_regular_bounded(
                &target_name,
                u64::try_from(bytes.len()).expect("UNION_HORIZON artifact length fits u64"),
            )
            .expect("reload published UNION_HORIZON artifact through its pinned directory");
        let reloaded = decode_union_horizon_artifact_bytes(&published_bytes, &raw_file_sha256);
        assert_eq!(
            &reloaded, artifact,
            "published UNION_HORIZON artifact changed during durable round-trip",
        );
        assert_union_horizon_executable_still_matches(
            &artifact
                .proofs
                .first()
                .expect("UNION_HORIZON artifact contains proofs")
                .build_identity,
        );
        Some(PublishedUnionHorizonDiagnostic {
            path: target,
            raw_file_sha256,
            byte_len: u64::try_from(published_bytes.len())
                .expect("UNION_HORIZON artifact byte length fits u64"),
        })
    }

    #[cfg(all(
        feature = "tantivy-oracle",
        feature = "pruning-conformance",
        any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "tvos",
            target_os = "watchos"
        )
    ))]
    fn validate_union_horizon_completed_bundle(
        directory: &crate::artifact::PinnedDirectory,
        manifest: &UnionHorizonCompletionManifest,
        completion_name: &std::ffi::OsStr,
    ) {
        manifest.verify();
        assert_union_horizon_executable_still_matches(&manifest.build_identity);

        let canonical_completion_bytes = serde_json::to_vec_pretty(manifest)
            .expect("serialize canonical UNION_HORIZON completion manifest");
        let completion_raw_sha256 = sha256_hex(&canonical_completion_bytes);
        let expected_completion_name = std::ffi::OsString::from(format!(
            "completion-{}-{}.json",
            manifest.manifest_sha256, completion_raw_sha256,
        ));
        assert_eq!(
            completion_name, expected_completion_name,
            "UNION_HORIZON completion filename is not bound to its canonical bytes",
        );

        let mut expected_names = manifest
            .artifacts
            .iter()
            .map(|entry| std::ffi::OsString::from(&entry.filename))
            .collect::<std::collections::BTreeSet<_>>();
        assert!(
            expected_names.insert(expected_completion_name),
            "UNION_HORIZON completion filename must be distinct from both proofs",
        );
        assert_eq!(
            directory
                .entry_names(expected_names.len())
                .expect("enumerate exact completed UNION_HORIZON bundle"),
            expected_names,
            "completed UNION_HORIZON bundle contains a missing, extra, renamed, or substituted entry",
        );

        let reloaded_completion = directory
            .read_regular_bounded(
                completion_name,
                u64::try_from(canonical_completion_bytes.len())
                    .expect("completion manifest length fits u64"),
            )
            .expect("reread completed manifest through pinned directory");
        assert_eq!(
            reloaded_completion, canonical_completion_bytes,
            "completed UNION_HORIZON manifest changed after sealing",
        );

        for entry in &manifest.artifacts {
            let name = std::ffi::OsStr::new(&entry.filename);
            let bytes = directory
                .read_regular_bounded(name, entry.byte_len)
                .expect("reread completed proof through pinned directory");
            assert_eq!(
                u64::try_from(bytes.len()).expect("completed proof length fits u64"),
                entry.byte_len,
                "completed UNION_HORIZON proof length changed after manifest sealing",
            );
            assert_eq!(
                sha256_hex(&bytes),
                entry.raw_file_sha256,
                "completed UNION_HORIZON proof bytes changed after manifest sealing",
            );
            let artifact = decode_union_horizon_artifact_bytes(&bytes, &entry.raw_file_sha256);
            assert_eq!(
                artifact, entry.artifact,
                "completed UNION_HORIZON proof does not equal its self-contained manifest copy",
            );
        }

        assert_eq!(
            directory
                .entry_names(expected_names.len())
                .expect("re-enumerate exact completed UNION_HORIZON bundle"),
            expected_names,
            "completed UNION_HORIZON bundle changed during final verification",
        );
        assert_union_horizon_executable_still_matches(&manifest.build_identity);
    }

    #[cfg(all(
        feature = "tantivy-oracle",
        feature = "pruning-conformance",
        any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "tvos",
            target_os = "watchos"
        )
    ))]
    fn publish_union_horizon_completion_manifest() -> (
        PublishedUnionHorizonDiagnostic,
        UnionHorizonCompletionManifest,
    ) {
        const ARTIFACT_ROOT_ENV: &str = "GAUNTLET_UNION_HORIZON_ARTIFACT_ROOT";
        const MAX_ARTIFACT_BYTES: u64 = 16 * 1024 * 1024;

        let root = std::env::var_os(ARTIFACT_ROOT_ENV)
            .map(std::path::PathBuf::from)
            .expect("completion requires an explicit UNION_HORIZON artifact root");
        let run_id = union_horizon_artifact_run_id();
        let build_identity = union_horizon_build_identity();
        assert_union_horizon_executable_still_matches(&build_identity);
        assert!(
            root.ends_with(&run_id),
            "UNION_HORIZON admitted evidence root must be isolated by exact run ID",
        );
        let directory = crate::artifact::PinnedDirectory::ensure_path(&root)
            .expect("pin UNION_HORIZON completion directory");
        let names = directory
            .entry_names(3)
            .expect("enumerate bounded UNION_HORIZON proof bundle");
        assert_eq!(
            names.len(),
            2,
            "completion requires exactly the late-winner and tie-matrix artifacts",
        );

        let mut entries = Vec::with_capacity(2);
        for name in names {
            let filename = name
                .to_str()
                .expect("UNION_HORIZON artifact filenames must be UTF-8")
                .to_owned();
            let raw_file_sha256 = filename
                .strip_suffix(".json")
                .and_then(|stem| stem.rsplit_once('-').map(|(_, raw)| raw))
                .expect("UNION_HORIZON artifact filename carries its raw SHA-256")
                .to_owned();
            let bytes = directory
                .read_regular_bounded(&name, MAX_ARTIFACT_BYTES)
                .expect("read proof artifact through pinned completion directory");
            let artifact = decode_union_horizon_artifact_bytes(&bytes, &raw_file_sha256);
            assert_eq!(
                artifact.run_id, run_id,
                "completion cannot combine artifacts from different CI invocations",
            );
            assert_eq!(
                filename,
                format!(
                    "{}-{}-{}-{}.json",
                    artifact.run_id,
                    artifact.proof_kind.label(),
                    artifact.artifact_sha256,
                    raw_file_sha256,
                ),
                "UNION_HORIZON artifact filename is not bound to its canonical evidence",
            );
            let proof_kind = artifact.proof_kind;
            let semantic_sha256 = artifact.artifact_sha256.clone();
            entries.push(UnionHorizonCompletionEntry {
                proof_kind,
                filename,
                semantic_sha256,
                raw_file_sha256,
                byte_len: u64::try_from(bytes.len())
                    .expect("UNION_HORIZON proof artifact length fits u64"),
                artifact,
            });
        }

        let manifest = UnionHorizonCompletionManifest::seal(run_id, entries);
        let bytes = serde_json::to_vec_pretty(&manifest)
            .expect("serialize sealed UNION_HORIZON completion manifest");
        let raw_file_sha256 = sha256_hex(&bytes);
        let filename = format!(
            "completion-{}-{}.json",
            manifest.manifest_sha256, raw_file_sha256,
        );
        let target_name = std::ffi::OsString::from(&filename);
        let temporary_name =
            std::ffi::OsString::from(format!(".tmp-{}-{filename}", std::process::id()));
        directory
            .publish_unique_no_clobber(&temporary_name, &target_name, &bytes)
            .expect("publish no-clobber UNION_HORIZON completion manifest");
        let reloaded_bytes = directory
            .read_regular_bounded(
                &target_name,
                u64::try_from(bytes.len()).expect("completion manifest length fits u64"),
            )
            .expect("reread completion manifest through pinned directory");
        assert_eq!(sha256_hex(&reloaded_bytes), raw_file_sha256);
        let reloaded: UnionHorizonCompletionManifest = serde_json::from_slice(&reloaded_bytes)
            .expect("strictly decode UNION_HORIZON completion manifest");
        assert_eq!(
            reloaded_bytes,
            serde_json::to_vec_pretty(&reloaded)
                .expect("re-encode canonical UNION_HORIZON completion manifest"),
            "completion manifest must retain its exact canonical bytes",
        );
        reloaded.verify();
        assert_eq!(reloaded, manifest);
        assert_union_horizon_executable_still_matches(&manifest.build_identity);
        validate_union_horizon_completed_bundle(&directory, &manifest, &target_name);
        eprintln!(
            "{}",
            serde_json::json!({
                "event": "salej_union_horizon_completion",
                "schema_version": manifest.schema_version,
                "run_id": manifest.run_id,
                "manifest_sha256": manifest.manifest_sha256,
                "raw_file_sha256": raw_file_sha256,
                "artifact_count": manifest.artifacts.len(),
                "test_executable_sha256": manifest.build_identity.test_executable_sha256,
            }),
        );
        (
            PublishedUnionHorizonDiagnostic {
                path: root.join(target_name),
                raw_file_sha256,
                byte_len: u64::try_from(reloaded_bytes.len())
                    .expect("completion manifest byte length fits u64"),
            },
            manifest,
        )
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn log_union_horizon_evidence(
        artifact: &UnionHorizonDiagnosticArtifact,
        publication: Option<&PublishedUnionHorizonDiagnostic>,
    ) {
        for proof in &artifact.proofs {
            let execution_mode = match proof.layout.expected_execution_mode() {
                frankensearch_quill::ConformancePruningExecutionMode::Serial => "serial",
                frankensearch_quill::ConformancePruningExecutionMode::Rayon => "rayon",
            };
            eprintln!(
                "{}",
                serde_json::json!({
                    "event": "salej_union_horizon_layout_proof",
                    "schema_version": artifact.schema_version,
                    "run_id": artifact.run_id,
                    "proof_kind": artifact.proof_kind.label(),
                    "layout": proof.layout.label(),
                    "execution_mode": execution_mode,
                    "query_count": proof.comparisons.len(),
                    "quill_segment_doc_counts": proof.topology.quill_segment_doc_counts,
                    "tantivy_segment_doc_counts": proof
                        .topology
                        .tantivy_segments
                        .iter()
                        .map(|segment| segment.num_docs)
                        .collect::<Vec<_>>(),
                    "complete_trace_count": proof.complete_pruning_traces.len(),
                    "artifact_sha256": artifact.artifact_sha256,
                }),
            );
        }
        eprintln!(
            "{}",
            serde_json::json!({
                "event": "salej_union_horizon_artifact",
                "schema_version": artifact.schema_version,
                "run_id": artifact.run_id,
                "proof_kind": artifact.proof_kind.label(),
                "proof_count": artifact.proofs.len(),
                "artifact_sha256": artifact.artifact_sha256,
                "raw_file_sha256": publication.map(|receipt| &receipt.raw_file_sha256),
                "byte_len": publication.map(|receipt| receipt.byte_len),
                "persisted": publication.is_some(),
                "path": publication.map(|receipt| receipt.path.display().to_string()),
            }),
        );
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    fn union_horizon_trace_geometry(
        trace: &UnionHorizonTraceReceipt,
    ) -> Vec<(
        u64,
        u32,
        u64,
        frankensearch_quill::ConformancePruningStrategy,
        bool,
        bool,
    )> {
        trace
            .refills
            .iter()
            .map(|refill| {
                (
                    refill.ordinal(),
                    refill.window_start(),
                    refill.horizon_end(),
                    refill.strategy(),
                    refill.buffer_empty(),
                    refill.live_work_remains(),
                )
            })
            .collect()
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    async fn run_union_horizon_layout_matrix(
        cx: &Cx,
        fixture: &Fixture,
        proof_kind: UnionHorizonProofKind,
    ) -> UnionHorizonDiagnosticArtifact {
        let mut proofs = Vec::with_capacity(UnionHorizonSegmentLayout::ALL.len());
        for layout in UnionHorizonSegmentLayout::ALL {
            let first = run_union_horizon_proof(cx, fixture, layout, proof_kind).await;
            let second = run_union_horizon_proof(cx, fixture, layout, proof_kind).await;
            assert_eq!(
                first,
                second,
                "UNION_HORIZON proof kind={} layout={} changed across fresh rebuilds",
                proof_kind.label(),
                layout.label(),
            );
            proofs.push(first);
        }

        let serial = proofs
            .iter()
            .find(|proof| matches!(proof.layout, UnionHorizonSegmentLayout::Two))
            .expect("UNION_HORIZON matrix contains the two-segment serial proof");
        let rayon = proofs
            .iter()
            .find(|proof| matches!(proof.layout, UnionHorizonSegmentLayout::Eight))
            .expect("UNION_HORIZON matrix contains the eight-segment Rayon proof");
        assert_eq!(serial.target_traces.len(), rayon.target_traces.len());
        for (serial_trace, rayon_trace) in serial.target_traces.iter().zip(&rayon.target_traces) {
            assert_eq!(serial_trace.limit, rayon_trace.limit);
            assert_eq!(
                serial_trace.segment_doc_start,
                rayon_trace.segment_doc_start
            );
            assert_eq!(
                serial_trace.segment_doc_count,
                rayon_trace.segment_doc_count
            );
            assert_eq!(
                union_horizon_trace_geometry(serial_trace),
                union_horizon_trace_geometry(rayon_trace),
                "UNION_HORIZON refill geometry and liveness must be invariant across serial and Rayon execution",
            );
            // The serial collector reaches the target tail with a global heap
            // populated by prefix segments. Rayon intentionally gives each
            // segment a fresh local heap before merging. Exact cutoff bits and
            // admitted-candidate counts are therefore branch-local evidence,
            // not a valid cross-branch equality contract.
        }
        assert_eq!(
            serial.comparisons.len(),
            rayon.comparisons.len(),
            "UNION_HORIZON serial and Rayon query matrices must have equal cardinality",
        );
        for (serial_run, rayon_run) in serial.comparisons.iter().zip(&rayon.comparisons) {
            assert_eq!(
                serial_run.comparison.subject, rayon_run.comparison.subject,
                "UNION_HORIZON subject observation must be invariant across serial and Rayon execution",
            );
        }

        let artifact = UnionHorizonDiagnosticArtifact::seal(
            union_horizon_artifact_run_id(),
            proof_kind,
            proofs,
        );
        let publication = persist_union_horizon_artifact(&artifact);
        log_union_horizon_evidence(&artifact, publication.as_ref());
        artifact
    }

    #[test]
    fn quill_subject_rejects_calls_outside_its_one_shot_lifecycle() {
        let fixture = make_fixture();
        let contract = SemanticContract::scalar_g1a();
        let selected = CampaignSelection::DefaultSyntax
            .select(&fixture.query_suite.cases)
            .expect("scalar G1a query selection");
        let query = (*selected[0]).clone();
        let mut evidence_case =
            DifferentialCase::new("lifecycle-observe-before-commit", &query.query, query.limit);
        evidence_case.offset = query.offset;
        evidence_case.count_requested = query.count_requested;
        evidence_case.snippet_max_chars = None;
        let deterministic_config = frankensearch_quill::QuillConfig {
            deterministic_ingest: true,
            ..frankensearch_quill::QuillConfig::default()
        };

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let mut before_begin = crate::engine::QuillSubject::in_memory(
                deterministic_config.clone(),
                "lifecycle-before-begin",
                false,
            )
            .expect("subject before begin");
            let index_before_begin = DifferentialCampaignEngine::index_batch(
                &mut before_begin,
                &cx,
                &fixture.documents[..1],
            )
            .await
            .expect_err("indexing before begin must fail");
            assert!(matches!(
                index_before_begin,
                GauntletError::InvalidCampaign { .. }
            ));

            let mut before_commit = crate::engine::QuillSubject::in_memory(
                deterministic_config.clone(),
                "lifecycle-before-commit",
                false,
            )
            .expect("subject before commit");
            DifferentialCampaignEngine::begin_corpus(
                &mut before_commit,
                &cx,
                &fixture.corpus_manifest,
                &contract,
            )
            .await
            .expect("begin ingest session");
            let observe_before_commit = DifferentialCampaignEngine::observe_generated(
                &mut before_commit,
                &cx,
                &query,
                &evidence_case,
            )
            .await
            .expect_err("observation before commit must fail");
            assert!(matches!(
                observe_before_commit,
                GauntletError::InvalidCampaign { .. }
            ));

            let mut after_commit = crate::engine::QuillSubject::in_memory(
                deterministic_config,
                "lifecycle-after-commit",
                false,
            )
            .expect("subject after commit");
            DifferentialCampaignEngine::begin_corpus(
                &mut after_commit,
                &cx,
                &fixture.corpus_manifest,
                &contract,
            )
            .await
            .expect("begin ingest session");
            DifferentialCampaignEngine::index_batch(&mut after_commit, &cx, &fixture.documents)
                .await
                .expect("index fixture corpus");
            DifferentialCampaignEngine::commit_corpus(
                &mut after_commit,
                &cx,
                &fixture.corpus_manifest,
                &contract,
            )
            .await
            .expect("commit fixture corpus");
            let index_after_commit = DifferentialCampaignEngine::index_batch(
                &mut after_commit,
                &cx,
                &fixture.documents[..1],
            )
            .await
            .expect_err("indexing after commit must fail");
            assert!(matches!(
                index_after_commit,
                GauntletError::InvalidCampaign { .. }
            ));
        });
    }

    #[test]
    fn replay_and_identity_fail_before_either_engine_ingests() {
        let fixture = make_fixture();
        let temp = tempfile::tempdir().expect("tempdir");
        let mut subject = ScriptedEngine::new(subject_descriptor(), BTreeMap::new());
        let mut oracle = ScriptedEngine::new(oracle_descriptor(), BTreeMap::new());
        let mut changed_documents = fixture.documents.clone();
        changed_documents[0].content.push_str("tampered");
        let campaign = runner(
            temp.path(),
            CampaignSelection::All,
            DivergenceRegistry::default(),
        );
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            assert!(
                campaign
                    .run(
                        &cx,
                        "tampered-replay",
                        &mut subject,
                        &mut oracle,
                        &changed_documents,
                        &fixture.corpus_manifest,
                        &fixture.query_suite,
                    )
                    .await
                    .is_err()
            );
            assert_eq!(subject.index_calls.load(Ordering::Relaxed), 0);
            assert_eq!(oracle.index_calls.load(Ordering::Relaxed), 0);
        });

        let fixture = make_fixture();
        let mut first = ScriptedEngine::new(oracle_descriptor(), BTreeMap::new());
        let mut second = ScriptedEngine::new(oracle_descriptor(), BTreeMap::new());
        let temp = tempfile::tempdir().expect("tempdir");
        let campaign = runner(
            temp.path(),
            CampaignSelection::All,
            DivergenceRegistry::default(),
        );
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            assert!(
                campaign
                    .run(
                        &cx,
                        "identity-collision",
                        &mut first,
                        &mut second,
                        &fixture.documents,
                        &fixture.corpus_manifest,
                        &fixture.query_suite,
                    )
                    .await
                    .is_err()
            );
            assert_eq!(first.index_calls.load(Ordering::Relaxed), 0);
            assert_eq!(second.index_calls.load(Ordering::Relaxed), 0);
        });
    }

    #[test]
    fn production_campaign_missing_provenance_fails_before_ingest() {
        let fixture = make_fixture();
        let mut subject = ScriptedEngine::new(subject_descriptor(), BTreeMap::new());
        let mut oracle = ScriptedEngine::new(oracle_descriptor(), BTreeMap::new());
        let temp = tempfile::tempdir().expect("tempdir");
        let campaign = DifferentialCampaignRunner::new(
            ArtifactStore::new(temp.path()),
            semantic_contract(),
            CampaignConfig {
                selection: CampaignSelection::DefaultSyntax,
                contract_mode: CampaignContractMode::CoreLexicalV3,
                require_provenance: true,
                ..CampaignConfig::default()
            },
            DivergenceRegistry::default(),
        )
        .expect("production campaign policy");
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let error = campaign
                .run(
                    &cx,
                    "missing-production-provenance",
                    &mut subject,
                    &mut oracle,
                    &fixture.documents,
                    &fixture.corpus_manifest,
                    &fixture.query_suite,
                )
                .await
                .expect_err("production provenance is mandatory");
            assert!(
                error.to_string().contains("missing required provenance"),
                "unexpected fail-closed reason: {error}"
            );
            assert_eq!(subject.index_calls.load(Ordering::Relaxed), 0);
            assert_eq!(oracle.index_calls.load(Ordering::Relaxed), 0);
        });
    }

    #[test]
    fn production_default_rank_envelope_is_rejected_before_ingest_with_valid_provenance() {
        let fixture = make_fixture();
        let semantic_contract = semantic_contract();
        let subject_descriptor = EngineDescriptor {
            source_revision: "1".repeat(40),
            ..subject_descriptor()
        };
        let oracle_descriptor = oracle_descriptor();
        let config = CampaignConfig {
            selection: CampaignSelection::DefaultSyntax,
            contract_mode: CampaignContractMode::RankEnvelopeOnly,
            require_provenance: true,
            ..CampaignConfig::default()
        };
        let provenance = fixture_provenance(
            &fixture,
            &config,
            &semantic_contract,
            &subject_descriptor,
            &oracle_descriptor,
        );
        let mut engines = EnginePairIdentity::new(
            ComparisonMode::CrossEngine,
            subject_descriptor.clone(),
            oracle_descriptor.clone(),
        )
        .expect("distinct engines");
        engines
            .bind_semantic_contract(semantic_contract.clone())
            .expect("semantic contract");
        provenance
            .validate_for_campaign(
                &engines,
                &semantic_contract,
                &config,
                &fixture.corpus_manifest,
                &fixture.query_suite.manifest,
            )
            .expect("the policy regression must carry otherwise-valid provenance");

        let mut subject = ScriptedEngine::new(subject_descriptor, BTreeMap::new());
        let mut oracle = ScriptedEngine::new(oracle_descriptor, BTreeMap::new());
        let temp = tempfile::tempdir().expect("tempdir");
        let campaign = DifferentialCampaignRunner::new(
            ArtifactStore::new(temp.path()),
            semantic_contract,
            config,
            DivergenceRegistry::default(),
        )
        .expect("production rank-envelope policy")
        .with_provenance(provenance);
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let error = campaign
                .run(
                    &cx,
                    "default-rank-envelope-is-not-replacement-evidence",
                    &mut subject,
                    &mut oracle,
                    &fixture.documents,
                    &fixture.corpus_manifest,
                    &fixture.query_suite,
                )
                .await
                .expect_err("default production evidence requires core lexical v3");
            assert!(
                error
                    .to_string()
                    .contains("require core lexical v3 evidence"),
                "unexpected fail-closed reason: {error}"
            );
            assert_eq!(subject.index_calls.load(Ordering::Relaxed), 0);
            assert_eq!(oracle.index_calls.load(Ordering::Relaxed), 0);
        });
    }

    #[test]
    fn production_cass_rank_envelope_remains_admissible_with_valid_provenance() {
        let fixture = make_fixture();
        let semantic_contract = SemanticContract::cass();
        let subject_descriptor = EngineDescriptor {
            source_revision: "1".repeat(40),
            ..subject_descriptor()
        };
        let oracle_descriptor = EngineDescriptor {
            config_hash: crate::engine::CASS_TANTIVY_ORACLE_CONFIG_HASH.to_owned(),
            ..oracle_descriptor()
        };
        let config = CampaignConfig {
            selection: CampaignSelection::CassSyntax,
            contract_mode: CampaignContractMode::RankEnvelopeOnly,
            require_provenance: true,
            index_batch_size: 5,
            ..CampaignConfig::default()
        };
        let provenance = fixture_provenance(
            &fixture,
            &config,
            &semantic_contract,
            &subject_descriptor,
            &oracle_descriptor,
        );
        let mut engines = EnginePairIdentity::new(
            ComparisonMode::CrossEngine,
            subject_descriptor.clone(),
            oracle_descriptor.clone(),
        )
        .expect("distinct CASS engines");
        engines
            .bind_semantic_contract(semantic_contract.clone())
            .expect("CASS semantic contract");
        provenance
            .validate_for_campaign(
                &engines,
                &semantic_contract,
                &config,
                &fixture.corpus_manifest,
                &fixture.query_suite.manifest,
            )
            .expect("the CASS positive control must carry valid provenance");
        let mut subject = ScriptedEngine::new(subject_descriptor, BTreeMap::new())
            .with_semantic_contract(semantic_contract.clone());
        let mut oracle = ScriptedEngine::new(oracle_descriptor, BTreeMap::new())
            .with_semantic_contract(semantic_contract.clone());
        let temp = tempfile::tempdir().expect("tempdir");
        let campaign = DifferentialCampaignRunner::new(
            ArtifactStore::new(temp.path()),
            semantic_contract,
            config,
            DivergenceRegistry::default(),
        )
        .expect("CASS production rank-envelope policy")
        .with_provenance(provenance);
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let report = campaign
                .run(
                    &cx,
                    "cass-rank-envelope-remains-supported",
                    &mut subject,
                    &mut oracle,
                    &fixture.documents,
                    &fixture.corpus_manifest,
                    &fixture.query_suite,
                )
                .await
                .expect("CASS retains its profile-specific rank-envelope contract");
            assert!(report.passed);
            assert!(!report.cases.is_empty());
            assert!(report.cases.iter().all(|case| {
                case.lexical_contract == CampaignLexicalCaseSummary::RankEnvelopeOnly
            }));
            assert!(subject.index_calls.load(Ordering::Relaxed) > 0);
            assert!(oracle.index_calls.load(Ordering::Relaxed) > 0);
        });
    }

    #[test]
    fn provenance_matches_every_engine_toolchain_and_query_pin() {
        let fixture = make_fixture();
        let semantic_contract = semantic_contract();
        let config = CampaignConfig {
            selection: CampaignSelection::All,
            require_provenance: true,
            ..CampaignConfig::default()
        };
        let subject_revision = "1".repeat(40);
        let subject = EngineDescriptor {
            source_revision: subject_revision.clone(),
            ..subject_descriptor()
        };
        let oracle = oracle_descriptor();
        let mut engines =
            EnginePairIdentity::new(ComparisonMode::CrossEngine, subject, oracle.clone())
                .expect("engine identity");
        engines
            .bind_semantic_contract(semantic_contract.clone())
            .expect("semantic contract");
        let provenance = CampaignProvenance {
            subject_git_revision: subject_revision,
            subject_source_dirty: false,
            oracle_git_revision: oracle.source_revision,
            oracle_source_dirty: oracle.source_dirty,
            cargo_lock_sha256: hash_workspace_lockfile().expect("Cargo.lock hash"),
            rustc_version_verbose: collect_rustc_verbose().expect("rustc provenance"),
            rust_toolchain_channel: collect_dated_toolchain_channel()
                .expect("dated nightly provenance"),
            unicode_version: format!(
                "{}.{}.{}",
                char::UNICODE_VERSION.0,
                char::UNICODE_VERSION.1,
                char::UNICODE_VERSION.2
            ),
            unicode_normalization_version: locked_crate_version("unicode-normalization")
                .expect("locked normalization version"),
            unicode_normalization_table_version: unicode_normalization_table_version(),
            query_generator_id: fixture.query_suite.manifest.generator_id.clone(),
            query_generator_schema_version: fixture.query_suite.manifest.schema_version,
            query_seed: fixture.query_suite.manifest.spec.seed,
            query_source_identity_sha256: fixture
                .query_suite
                .manifest
                .source_identity_sha256
                .clone(),
            query_profile_sha256: query_profile_sha256(
                &fixture.query_suite.manifest,
                &config.selection,
                &semantic_contract,
            )
            .expect("query profile hash"),
            analyzer_contract_hash: semantic_contract.analyzer_contract_hash.clone(),
            schema_contract_hash: semantic_contract.schema_contract_hash.clone(),
            corpus_manifest_hash: fixture
                .corpus_manifest
                .manifest_hash()
                .expect("corpus manifest hash"),
            query_manifest_hash: fixture
                .query_suite
                .manifest
                .manifest_hash()
                .expect("query manifest hash"),
            corpus_seed: Some(0x6200),
        };
        provenance
            .validate_for_campaign(
                &engines,
                &semantic_contract,
                &config,
                &fixture.corpus_manifest,
                &fixture.query_suite.manifest,
            )
            .expect("all exact provenance pins validate");

        let serialized = serde_json::to_value(&provenance).expect("serialize provenance");
        for field in [
            "subject_git_revision",
            "subject_source_dirty",
            "oracle_git_revision",
            "oracle_source_dirty",
            "cargo_lock_sha256",
            "rustc_version_verbose",
            "rust_toolchain_channel",
            "unicode_version",
            "unicode_normalization_version",
            "unicode_normalization_table_version",
            "query_generator_id",
            "query_generator_schema_version",
            "query_seed",
            "query_source_identity_sha256",
            "query_profile_sha256",
            "analyzer_contract_hash",
            "schema_contract_hash",
            "corpus_manifest_hash",
            "query_manifest_hash",
            "corpus_seed",
        ] {
            let mut missing = serialized.clone();
            missing
                .as_object_mut()
                .expect("provenance object")
                .remove(field);
            assert!(
                serde_json::from_value::<CampaignProvenance>(missing).is_err(),
                "missing provenance field {field} must fail closed"
            );
        }

        type CorruptProvenance = fn(&mut CampaignProvenance);
        let corruptions: [(&str, CorruptProvenance); 20] = [
            ("subject_git_revision", |value| {
                value.subject_git_revision = "2".repeat(40);
            }),
            ("subject_source_dirty", |value| {
                value.subject_source_dirty = !value.subject_source_dirty;
            }),
            ("oracle_git_revision", |value| {
                value.oracle_git_revision = "3".repeat(40);
            }),
            ("oracle_source_dirty", |value| {
                value.oracle_source_dirty = !value.oracle_source_dirty;
            }),
            ("cargo_lock_sha256", |value| {
                value.cargo_lock_sha256 = "0".repeat(64);
            }),
            ("rustc_version_verbose", |value| {
                value.rustc_version_verbose.push_str("mismatch");
            }),
            ("rust_toolchain_channel", |value| {
                value.rust_toolchain_channel = "nightly-1970-01-01".to_owned();
            }),
            ("unicode_version", |value| {
                value.unicode_version = "0.0.0".to_owned();
            }),
            ("unicode_normalization_version", |value| {
                value.unicode_normalization_version = "0.0.0".to_owned();
            }),
            ("unicode_normalization_table_version", |value| {
                value.unicode_normalization_table_version = "0.0.0".to_owned();
            }),
            ("query_generator_id", |value| {
                value.query_generator_id = "wrong-generator".to_owned();
            }),
            ("query_generator_schema_version", |value| {
                value.query_generator_schema_version =
                    value.query_generator_schema_version.saturating_add(1);
            }),
            ("query_seed", |value| {
                value.query_seed ^= 1;
            }),
            ("query_source_identity_sha256", |value| {
                value.query_source_identity_sha256 = "0".repeat(64);
            }),
            ("query_profile_sha256", |value| {
                value.query_profile_sha256 = "0".repeat(64);
            }),
            ("analyzer_contract_hash", |value| {
                value.analyzer_contract_hash = "0".repeat(64);
            }),
            ("schema_contract_hash", |value| {
                value.schema_contract_hash = "0".repeat(64);
            }),
            ("corpus_manifest_hash", |value| {
                value.corpus_manifest_hash = "0".repeat(64);
            }),
            ("query_manifest_hash", |value| {
                value.query_manifest_hash = "0".repeat(64);
            }),
            ("corpus_seed", |value| {
                value.corpus_seed = None;
            }),
        ];
        for (field, corrupt) in corruptions {
            let mut mismatched = provenance.clone();
            corrupt(&mut mismatched);
            assert!(
                mismatched
                    .validate_for_campaign(
                        &engines,
                        &semantic_contract,
                        &config,
                        &fixture.corpus_manifest,
                        &fixture.query_suite.manifest,
                    )
                    .is_err(),
                "mismatched provenance field {field} must fail closed"
            );
        }
    }

    #[test]
    fn invalid_manifests_semantics_and_deserialized_registers_fail_closed() {
        let mut fixture = make_fixture();
        fixture.query_suite.manifest.schema_version = 999;
        let mut subject = ScriptedEngine::new(subject_descriptor(), BTreeMap::new());
        let mut oracle = ScriptedEngine::new(oracle_descriptor(), BTreeMap::new());
        let temp = tempfile::tempdir().expect("tempdir");
        let campaign = runner(
            temp.path(),
            CampaignSelection::All,
            DivergenceRegistry::default(),
        );
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            assert!(
                campaign
                    .run(
                        &cx,
                        "invalid-query-manifest",
                        &mut subject,
                        &mut oracle,
                        &fixture.documents,
                        &fixture.corpus_manifest,
                        &fixture.query_suite,
                    )
                    .await
                    .is_err()
            );
            assert_eq!(subject.index_calls.load(Ordering::Relaxed), 0);
            assert_eq!(oracle.index_calls.load(Ordering::Relaxed), 0);
        });

        let fixture = make_fixture();
        let mut subject = ScriptedEngine::new(subject_descriptor(), BTreeMap::new())
            .with_semantic_contract(
                SemanticContract::new("c".repeat(64), "b".repeat(64)).expect("different contract"),
            );
        let mut oracle = ScriptedEngine::new(oracle_descriptor(), BTreeMap::new());
        let temp = tempfile::tempdir().expect("tempdir");
        let campaign = runner(
            temp.path(),
            CampaignSelection::All,
            DivergenceRegistry::default(),
        );
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            assert!(
                campaign
                    .run(
                        &cx,
                        "semantic-mismatch",
                        &mut subject,
                        &mut oracle,
                        &fixture.documents,
                        &fixture.corpus_manifest,
                        &fixture.query_suite,
                    )
                    .await
                    .is_err()
            );
            assert_eq!(subject.index_calls.load(Ordering::Relaxed), 0);
            assert_eq!(oracle.index_calls.load(Ordering::Relaxed), 0);
        });

        let unreviewed: DivergenceRegistry = serde_json::from_value(serde_json::json!({
            "entries": [{
                "id": "DIV-004",
                "class": "oversized_query_token",
                "fixture_id": "term",
                "mismatch_signatures": ["0000000000000000000000000000000000000000000000000000000000000000"],
                "decision": "pending",
                "root_cause": "known",
                "consumer_impact": "known",
                "reviewer": "",
                "reviewed_at": "2026-07-18"
            }]
        }))
        .expect("DTO deserializes before policy validation");
        let temp = tempfile::tempdir().expect("tempdir");
        assert!(
            DifferentialCampaignRunner::new(
                ArtifactStore::new(temp.path()),
                semantic_contract(),
                CampaignConfig::default(),
                unreviewed,
            )
            .is_err()
        );
    }

    #[test]
    fn indexing_replay_drift_and_batch_failure_abort_both_adapters() {
        let fixture = make_fixture();
        let mut drifted = fixture.documents.clone();
        drifted[0].content.push_str("drift");
        let replay = DriftingReplay {
            calls: AtomicUsize::new(0),
            first: fixture.documents.clone(),
            second: drifted,
        };
        let mut subject = ScriptedEngine::new(subject_descriptor(), BTreeMap::new());
        let mut oracle = ScriptedEngine::new(oracle_descriptor(), BTreeMap::new());
        let temp = tempfile::tempdir().expect("tempdir");
        let campaign = runner(
            temp.path(),
            CampaignSelection::All,
            DivergenceRegistry::default(),
        );
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            assert!(
                campaign
                    .run_replay(
                        &cx,
                        "drifting-replay",
                        &mut subject,
                        &mut oracle,
                        &replay,
                        &fixture.corpus_manifest,
                        &fixture.query_suite,
                    )
                    .await
                    .is_err()
            );
            assert_eq!(subject.abort_calls.load(Ordering::Relaxed), 1);
            assert_eq!(oracle.abort_calls.load(Ordering::Relaxed), 1);
            assert_eq!(subject.observe_calls.load(Ordering::Relaxed), 0);
            assert_eq!(oracle.observe_calls.load(Ordering::Relaxed), 0);
        });

        let fixture = make_fixture();
        let mut overlong = fixture.documents.clone();
        overlong.push(fixture.documents[0].clone());
        let replay = DriftingReplay {
            calls: AtomicUsize::new(0),
            first: overlong,
            second: fixture.documents.clone(),
        };
        let mut subject = ScriptedEngine::new(subject_descriptor(), BTreeMap::new());
        let mut oracle = ScriptedEngine::new(oracle_descriptor(), BTreeMap::new());
        let temp = tempfile::tempdir().expect("tempdir");
        let campaign = runner(
            temp.path(),
            CampaignSelection::All,
            DivergenceRegistry::default(),
        );
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            assert!(
                campaign
                    .run_replay(
                        &cx,
                        "overlong-first-replay",
                        &mut subject,
                        &mut oracle,
                        &replay,
                        &fixture.corpus_manifest,
                        &fixture.query_suite,
                    )
                    .await
                    .is_err()
            );
            assert_eq!(subject.index_calls.load(Ordering::Relaxed), 0);
            assert_eq!(oracle.index_calls.load(Ordering::Relaxed), 0);
        });

        let fixture = make_fixture();
        let mut subject =
            ScriptedEngine::new(subject_descriptor(), BTreeMap::new()).with_failing_index_batch();
        let mut oracle = ScriptedEngine::new(oracle_descriptor(), BTreeMap::new());
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().to_path_buf();
        let campaign = runner(&root, CampaignSelection::All, DivergenceRegistry::default());
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            assert!(
                campaign
                    .run(
                        &cx,
                        "batch-failure",
                        &mut subject,
                        &mut oracle,
                        &fixture.documents,
                        &fixture.corpus_manifest,
                        &fixture.query_suite,
                    )
                    .await
                    .is_err()
            );
            assert_eq!(subject.abort_calls.load(Ordering::Relaxed), 1);
            assert_eq!(oracle.abort_calls.load(Ordering::Relaxed), 1);
            assert!(
                root.join("campaigns/batch-failure/reservation.json")
                    .is_file()
            );
            assert!(!root.join("campaigns/batch-failure/report.json").exists());
        });

        let fixture = make_fixture();
        let mut subject =
            ScriptedEngine::new(subject_descriptor(), BTreeMap::new()).with_failing_begin();
        let mut oracle = ScriptedEngine::new(oracle_descriptor(), BTreeMap::new());
        let temp = tempfile::tempdir().expect("tempdir");
        let campaign = runner(
            temp.path(),
            CampaignSelection::All,
            DivergenceRegistry::default(),
        );
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            assert!(
                campaign
                    .run(
                        &cx,
                        "subject-begin-failure",
                        &mut subject,
                        &mut oracle,
                        &fixture.documents,
                        &fixture.corpus_manifest,
                        &fixture.query_suite,
                    )
                    .await
                    .is_err()
            );
            assert_eq!(subject.abort_calls.load(Ordering::Relaxed), 1);
            assert_eq!(oracle.abort_calls.load(Ordering::Relaxed), 0);
        });

        let fixture = make_fixture();
        let mut subject = ScriptedEngine::new(subject_descriptor(), BTreeMap::new())
            .with_failing_index_batch()
            .with_panicking_abort();
        let mut oracle = ScriptedEngine::new(oracle_descriptor(), BTreeMap::new());
        let temp = tempfile::tempdir().expect("tempdir");
        let campaign = runner(
            temp.path(),
            CampaignSelection::All,
            DivergenceRegistry::default(),
        );
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            assert!(
                campaign
                    .run(
                        &cx,
                        "panicking-subject-abort",
                        &mut subject,
                        &mut oracle,
                        &fixture.documents,
                        &fixture.corpus_manifest,
                        &fixture.query_suite,
                    )
                    .await
                    .is_err()
            );
            assert_eq!(subject.abort_calls.load(Ordering::Relaxed), 1);
            assert_eq!(oracle.abort_calls.load(Ordering::Relaxed), 1);
        });
    }

    #[test]
    fn indexing_batches_are_bounded_by_canonical_bytes_and_identical() {
        let query_fixture = make_fixture();
        let snapshot = RepositorySnapshot::from_entries(
            "byte-bounded-campaign",
            [
                RepositoryEntry {
                    relative_path: std::path::PathBuf::from("large-a.txt"),
                    bytes: vec![b'a'; 1024 * 1024],
                },
                RepositoryEntry {
                    relative_path: std::path::PathBuf::from("large-b.txt"),
                    bytes: vec![b'b'; 1024 * 1024],
                },
            ],
        )
        .expect("repository snapshot");
        let corpus_hash = snapshot.manifest.manifest_hash().expect("manifest hash");
        let query_suite = GeneratedQuerySuite::from_cases(
            query_fixture.query_suite.manifest.spec,
            &corpus_hash,
            vec![query_fixture.query_suite.cases[0].clone()],
        )
        .expect("query suite");
        let mut subject = ScriptedEngine::new(subject_descriptor(), BTreeMap::new());
        let mut oracle = ScriptedEngine::new(oracle_descriptor(), BTreeMap::new());
        let temp = tempfile::tempdir().expect("tempdir");
        let campaign = DifferentialCampaignRunner::new(
            ArtifactStore::new(temp.path()),
            semantic_contract(),
            CampaignConfig {
                selection: CampaignSelection::All,
                index_batch_size: 100,
                index_batch_max_bytes: 1_500_000,
                ..CampaignConfig::default()
            },
            DivergenceRegistry::default(),
        )
        .expect("campaign");
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            campaign
                .run(
                    &cx,
                    "byte-bounded",
                    &mut subject,
                    &mut oracle,
                    &snapshot.documents,
                    &snapshot.manifest,
                    &query_suite,
                )
                .await
                .expect("campaign report");
            let subject_batches = subject.indexed_payloads.lock().expect("subject batches");
            let oracle_batches = oracle.indexed_payloads.lock().expect("oracle batches");
            assert_eq!(subject_batches.as_slice(), oracle_batches.as_slice());
            assert_eq!(subject_batches.len(), 2);
        });
    }

    #[test]
    fn runner_preserves_rich_cases_and_persists_one_object_per_query() {
        let fixture = make_fixture();
        let corpus_hash = fixture.corpus_hash.clone();
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("gauntlet");
        let behaviors = BTreeMap::from([("counted".to_owned(), ScriptedBehavior::TieOrder)]);
        let mut subject = ScriptedEngine::new(subject_descriptor(), behaviors.clone());
        let mut oracle = ScriptedEngine::new(oracle_descriptor(), behaviors);
        let campaign = runner(
            &root,
            CampaignSelection::CaseIds {
                ids: vec![
                    "paginated".to_owned(),
                    "term".to_owned(),
                    "counted".to_owned(),
                ],
            },
            DivergenceRegistry::default(),
        );
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let _temp = temp;
            let report = campaign
                .run(
                    &cx,
                    "rich-fast",
                    &mut subject,
                    &mut oracle,
                    &fixture.documents,
                    &fixture.corpus_manifest,
                    &fixture.query_suite,
                )
                .await
                .expect("campaign report");
            assert!(report.passed);
            assert_eq!(report.selected_query_count, 3);
            assert_eq!(report.cases.len(), 3);
            assert_eq!(
                report
                    .cases
                    .iter()
                    .filter(|case| case.disposition == CampaignDisposition::Exact)
                    .count(),
                2
            );
            assert_eq!(
                report
                    .cases
                    .iter()
                    .filter(|case| case.disposition == CampaignDisposition::AutoClassified)
                    .count(),
                1
            );
            assert!(report.cases.iter().all(|case| case.artifact_hash.is_some()));
            assert_eq!(subject.index_calls.load(Ordering::Relaxed), 1);
            assert_eq!(oracle.index_calls.load(Ordering::Relaxed), 1);
            let subject_batches = subject.indexed_payloads.lock().expect("subject payload");
            let oracle_batches = oracle.indexed_payloads.lock().expect("oracle payload");
            assert_eq!(subject_batches.as_slice(), oracle_batches.as_slice());
            assert_eq!(subject_batches.len(), 3);
            let observed = subject.observed_queries.lock().expect("observed queries");
            assert_eq!(observed.len(), 3);
            assert!(observed.iter().any(|query| query.offset > 0));
            drop(observed);
            assert_eq!(
                std::fs::read_dir(root.join("objects"))
                    .expect("objects directory")
                    .count(),
                3
            );
            assert_eq!(
                std::fs::read_dir(root.join("campaigns/rich-fast/cases"))
                    .expect("campaign cases directory")
                    .count(),
                3
            );
            assert!(root.join("campaigns/rich-fast/reservation.json").is_file());
            let report_path = root.join("campaigns/rich-fast/report.json");
            assert!(report_path.is_file());
            assert_eq!(report.corpus_manifest_hash, corpus_hash);
            assert_eq!(report.report_hash().expect("report hash").len(), 64);
            let canonical = report.canonical_bytes().expect("canonical report");
            assert_eq!(
                std::fs::read(&report_path).expect("stored report"),
                canonical
            );
            let replayed: CampaignReport =
                serde_json::from_slice(&canonical).expect("report round-trip");
            assert_eq!(replayed, report);
            let verified = ArtifactStore::new(&root)
                .load_verified_campaign("rich-fast")
                .expect("evidence-backed campaign replay");
            assert_eq!(verified, report);
            ArtifactStore::new(&root)
                .complete_campaign(&replayed)
                .expect("idempotent campaign completion");
            let mut with_diagnostic = report.clone();
            with_diagnostic.cases[0].diagnostic = Some("/tmp/host-specific error".to_owned());
            assert_eq!(
                with_diagnostic.report_hash().expect("diagnostic-free hash"),
                report.report_hash().expect("report hash")
            );

            let mut wrong_pass = report.clone();
            wrong_pass.passed = false;
            assert!(wrong_pass.canonical_bytes().is_err());
            let mut wrong_count = report.clone();
            wrong_count.selected_query_count += 1;
            assert!(wrong_count.canonical_bytes().is_err());
            let mut wrong_summary = report.clone();
            wrong_summary.query_classes[0].total += 1;
            assert!(wrong_summary.canonical_bytes().is_err());
            for legacy_version in [3, 4] {
                let mut legacy = report.clone();
                legacy.schema_version = legacy_version;
                let error = legacy
                    .validate_contract()
                    .expect_err("pre-v5 report must require a campaign rerun");
                assert!(matches!(
                    error,
                    GauntletError::InvalidCampaign { ref reason }
                        if reason.contains("legacy campaign report")
                            && reason.contains("non-admissible")
                            && reason.contains("rerun")
                ));
            }
            let mut changed_query = report.clone();
            changed_query.query_suite.cases[0]
                .query
                .push_str(" tampered");
            assert!(changed_query.canonical_bytes().is_err());
            let mut legacy_address = report.clone();
            legacy_address.cases[0].artifact_hash = Some("0".repeat(16));
            assert!(
                legacy_address.canonical_bytes().is_err(),
                "current reports must reject legacy XXH3-64 object addresses"
            );
            let mut wrong_artifact = report.clone();
            wrong_artifact.cases[0].artifact_hash = Some("0".repeat(64));
            assert!(
                ArtifactStore::new(&root)
                    .complete_campaign(&wrong_artifact)
                    .is_err()
            );
            assert_eq!(
                std::fs::read(report_path).expect("unchanged report"),
                canonical
            );
        });
    }

    #[test]
    fn generated_default_suite_has_no_unexecutable_register_claims() {
        let fixture = make_fixture();
        assert!(
            fixture
                .query_suite
                .cases
                .iter()
                .all(|query| query.expected_divergence.is_none())
        );
        let temp = tempfile::tempdir().expect("tempdir");
        let mut subject = ScriptedEngine::new(subject_descriptor(), BTreeMap::new());
        let mut oracle = ScriptedEngine::new(oracle_descriptor(), BTreeMap::new());
        let campaign = runner(
            temp.path(),
            CampaignSelection::All,
            DivergenceRegistry::default(),
        );

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let report = campaign
                .run(
                    &cx,
                    "generated-exact",
                    &mut subject,
                    &mut oracle,
                    &fixture.documents,
                    &fixture.corpus_manifest,
                    &fixture.query_suite,
                )
                .await
                .expect("full generated suite");

            assert!(report.passed);
            assert_eq!(report.cases.len(), fixture.query_suite.cases.len());
            assert!(
                report
                    .cases
                    .iter()
                    .all(|case| case.disposition == CampaignDisposition::Exact)
            );
        });
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn scalar_g1a_harvested_default_syntax_is_exact_and_deterministic() {
        let fixture = make_scalar_g1a_regression_fixture();
        let first_root = tempfile::tempdir()
            .expect("first deterministic regression tempdir")
            .keep();
        let second_root = tempfile::tempdir()
            .expect("second deterministic regression tempdir")
            .keep();
        let trace_buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
        let writer_buffer = Arc::clone(&trace_buffer);
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_env_filter("off,frankensearch.quill=info")
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
            .with_writer(move || TraceLogWriter {
                buffer: Arc::clone(&writer_buffer),
            })
            .finish();

        tracing::subscriber::with_default(subscriber, || {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let first = run_scalar_g1a_deterministic_regression(&cx, &first_root, &fixture)
                    .await
                    .unwrap_or_else(|error| {
                        panic!(
                            "corpus_hash={} run=first campaign_error={error}",
                            fixture.corpus_hash
                        )
                    });
                let second = run_scalar_g1a_deterministic_regression(&cx, &second_root, &fixture)
                    .await
                    .unwrap_or_else(|error| {
                        panic!(
                            "corpus_hash={} run=second campaign_error={error}",
                            fixture.corpus_hash
                        )
                    });

                let selected_ids = first
                    .cases
                    .iter()
                    .map(|case| case.case_id.as_str())
                    .collect::<Vec<_>>();
                let expected_ids = CampaignSelection::DefaultSyntax
                    .select(&fixture.query_suite.cases)
                    .expect("scalar G1a default-syntax selection")
                    .into_iter()
                    .map(|case| case.id.as_str())
                    .collect::<Vec<_>>();
                assert_eq!(
                    selected_ids, expected_ids,
                    "the scalar G1a regression must execute the complete harvested default-parser corpus",
                );
                for required in [
                    "term",
                    "multi-term",
                    "phrase",
                    "same-position-phrase",
                    "boolean-default",
                    "paginated",
                    "uncounted",
                    "counted",
                ] {
                    assert!(
                        first.cases.iter().any(|case| case.case_id == required),
                        "the scalar G1a regression dropped owned parser class {required}",
                    );
                }
                assert_eq!(
                    first.selected_query_count,
                    u64::try_from(expected_ids.len()).expect("default query count fits u64"),
                );
                assert_eq!(
                    fixture
                        .query_suite
                        .cases
                        .iter()
                        .filter(|case| {
                            case.syntax == QuerySyntax::Default
                                && case.source == "tests/fixtures/queries.json"
                        })
                        .count(),
                    26,
                    "all committed harvested relevance queries must enter the live campaign",
                );
                let mut observed_regression_hit = false;
                let mut rank_cases = Vec::new();
                for case in &first.cases {
                    let object_hash = case
                        .artifact_hash
                        .as_deref()
                        .expect("regression case artifact hash");
                    let object_path = first_root
                        .join("objects")
                        .join(format!("{object_hash}.json"));
                    let object: ArtifactObject = serde_json::from_slice(
                        &std::fs::read(&object_path).expect("regression case artifact bytes"),
                    )
                    .expect("regression case artifact object");
                    assert_eq!(
                        case.disposition,
                        CampaignDisposition::Exact,
                        "corpus_hash={} query_seed={} query_id={} first_divergence={:?} reason={:?} divergences={:?} subject_hits={:?} oracle_hits={:?}",
                        first.corpus_manifest_hash,
                        fixture.query_suite.manifest.spec.seed,
                        case.case_id,
                        case.first_divergence,
                        case.reason,
                        object.comparison.divergences,
                        object.comparison.subject.hits,
                        object.comparison.oracle.hits,
                    );
                    assert_eq!(
                        case.comparison_status,
                        Some(ComparisonStatus::Exact),
                        "corpus_hash={} query_seed={} query_id={} first_divergence={:?}",
                        first.corpus_manifest_hash,
                        fixture.query_suite.manifest.spec.seed,
                        case.case_id,
                        case.first_divergence,
                    );
                    assert_eq!(
                        case.rank_class,
                        Some(RankClass::RankExact),
                        "corpus_hash={} query_seed={} query_id={} first_divergence={:?}",
                        first.corpus_manifest_hash,
                        fixture.query_suite.manifest.spec.seed,
                        case.case_id,
                        case.first_divergence,
                    );
                    assert!(
                        object.comparison.subject.snippets.is_empty()
                            && object.comparison.oracle.snippets.is_empty(),
                        "corpus_hash={} query_seed={} query_id={} unexpectedly emitted snippets",
                        first.corpus_manifest_hash,
                        fixture.query_suite.manifest.spec.seed,
                        case.case_id,
                    );
                    let generated = fixture
                        .query_suite
                        .cases
                        .iter()
                        .find(|query| query.id == case.case_id)
                        .expect("campaign report case comes from the generated suite");
                    if generated.source == "tests/fixtures/queries.json" {
                        rank_cases.push(E410RankCase {
                            query_id: case.case_id.clone(),
                            ranked_document_ids: object
                                .comparison
                                .subject
                                .hits
                                .iter()
                                .map(|hit| hit.doc_id.clone())
                                .collect(),
                        });
                    }
                    if generated.source == "tests/fixtures/queries.json"
                        && case.case_id == "harvested-22"
                    {
                        assert!(
                            !object.comparison.subject.hits.is_empty(),
                            "corpus_hash={} query_seed={} duplicate-term regression query_id={} was vacuous",
                            first.corpus_manifest_hash,
                            fixture.query_suite.manifest.spec.seed,
                            case.case_id,
                        );
                    }
                    observed_regression_hit |= !object.comparison.subject.hits.is_empty();
                }
                let actual_golden = E410RankGolden {
                    schema_version: 1,
                    corpus_manifest_hash: first.corpus_manifest_hash.clone(),
                    query_manifest_hash: first.query_manifest_hash.clone(),
                    query_seed: fixture.query_suite.manifest.spec.seed,
                    cases: rank_cases,
                };
                let expected_golden: E410RankGolden = serde_json::from_str(E410_RANK_GOLDEN_JSON)
                    .expect("parse committed E4.10 rank-list golden");
                let actual_golden_json = serde_json::to_string_pretty(&actual_golden)
                    .expect("serialize actual E4.10 rank-list golden");
                assert_eq!(
                    actual_golden.schema_version, expected_golden.schema_version,
                    "corpus_hash={} query_seed={} rank golden schema drifted; actual={actual_golden_json}",
                    first.corpus_manifest_hash, actual_golden.query_seed,
                );
                assert_eq!(
                    actual_golden.corpus_manifest_hash, expected_golden.corpus_manifest_hash,
                    "corpus_hash={} query_seed={} rank golden corpus binding drifted; actual={actual_golden_json}",
                    first.corpus_manifest_hash, actual_golden.query_seed,
                );
                assert_eq!(
                    actual_golden.query_manifest_hash, expected_golden.query_manifest_hash,
                    "corpus_hash={} query_seed={} rank golden query-manifest binding drifted; actual={actual_golden_json}",
                    first.corpus_manifest_hash, actual_golden.query_seed,
                );
                assert_eq!(
                    actual_golden.query_seed, expected_golden.query_seed,
                    "corpus_hash={} query_seed={} rank golden replay seed drifted; actual={actual_golden_json}",
                    first.corpus_manifest_hash, actual_golden.query_seed,
                );
                assert_eq!(
                    actual_golden.cases.len(),
                    expected_golden.cases.len(),
                    "corpus_hash={} query_seed={} rank golden case count drifted; actual={actual_golden_json}",
                    first.corpus_manifest_hash,
                    actual_golden.query_seed,
                );
                for (actual, expected) in actual_golden.cases.iter().zip(&expected_golden.cases) {
                    assert_eq!(
                        actual, expected,
                        "corpus_hash={} query_seed={} query_id={} rank-list golden drifted",
                        first.corpus_manifest_hash, actual_golden.query_seed, actual.query_id,
                    );
                }
                assert!(
                    observed_regression_hit,
                    "corpus_hash={} query_seed={} deterministic regression was vacuous",
                    first.corpus_manifest_hash, fixture.query_suite.manifest.spec.seed,
                );
                assert_eq!(
                    first.report_hash().expect("first report hash"),
                    second.report_hash().expect("second report hash")
                );
                assert_eq!(first, second, "repeated deterministic regression drifted");
            });
        });
        let logs = String::from_utf8(
            trace_buffer
                .lock()
                .expect("trace buffer lock is not poisoned")
                .clone(),
        )
        .expect("captured Quill trace is UTF-8");
        assert_scalar_g1a_trace_contract(&logs);
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    #[test]
    fn salej_runtime_identity_rejects_revision_or_dirty_state_rewrites() {
        assert!(
            reject_union_horizon_oracle_identity_overrides(
                Some(std::ffi::OsStr::new(
                    "062a5e5b2d41653b1c8b07888eda1a765e421f49"
                )),
                None,
            )
            .is_err(),
            "Salej must reject obsolete per-oracle provenance inputs before execution",
        );
        let compiled = union_horizon_build_identity();
        #[cfg(target_os = "linux")]
        let executable_path = std::path::PathBuf::from("/proc/self/exe");
        #[cfg(not(target_os = "linux"))]
        let executable_path =
            std::env::current_exe().expect("resolve independently hashed test executable path");
        let mut executable = std::fs::File::open(&executable_path)
            .unwrap_or_else(|error| panic!("open independently hashed test executable: {error}"));
        let mut independent_hasher = Sha256::new();
        let mut independent_len = 0_u64;
        let mut independent_buffer = [0_u8; 17 * 1024 + 3];
        loop {
            let read = executable
                .read(&mut independent_buffer)
                .expect("independently stream test executable");
            if read == 0 {
                break;
            }
            independent_hasher.update(&independent_buffer[..read]);
            independent_len = independent_len
                .checked_add(u64::try_from(read).expect("independent read length fits u64"))
                .expect("independent executable length cannot overflow u64");
        }
        assert_eq!(
            compiled.test_executable_sha256,
            lower_hex(&independent_hasher.finalize()),
            "streamed executable identity must match an independent chunking strategy",
        );
        assert_eq!(compiled.test_executable_byte_len, independent_len);

        let mut embedded = compiled.clone();
        embedded.source_git_revision = "a".repeat(40);
        embedded.source_git_dirty = true;
        embedded.source_verification = UnionHorizonSourceVerification::ExplicitUnverified;
        assert!(validate_union_horizon_runtime_identity_override(&embedded, None, None).is_ok());
        assert!(
            validate_union_horizon_runtime_identity_override(
                &embedded,
                Some(&embedded.source_git_revision),
                Some("true"),
            )
            .is_ok()
        );
        for (revision, dirty) in [
            (Some(embedded.source_git_revision.as_str()), None),
            (None, Some("true")),
            (
                Some("cccccccccccccccccccccccccccccccccccccccc"),
                Some("true"),
            ),
            (Some(embedded.source_git_revision.as_str()), Some("false")),
            (Some(embedded.source_git_revision.as_str()), Some("0")),
        ] {
            assert!(
                validate_union_horizon_runtime_identity_override(&embedded, revision, dirty)
                    .is_err(),
                "runtime identity rewrite unexpectedly passed: revision={revision:?} dirty={dirty:?}",
            );
        }
        assert!(
            !union_horizon_identity_is_publishable(&embedded),
            "plain explicit identity must remain diagnostic-only",
        );
        embedded.source_git_dirty = false;
        assert!(
            !union_horizon_identity_is_publishable(&embedded),
            "runtime clean metadata cannot upgrade an unverified source identity",
        );
        embedded.source_verification = UnionHorizonSourceVerification::GitCheckoutVerified;
        #[cfg(target_os = "linux")]
        assert!(
            union_horizon_identity_is_publishable(&embedded),
            "a clean build-time Git verification is publishable on Linux",
        );
        #[cfg(not(target_os = "linux"))]
        assert!(
            !union_horizon_identity_is_publishable(&embedded),
            "non-Linux builds remain diagnostic-only until executable identity is pinned without a path race",
        );
    }

    #[cfg(all(
        feature = "tantivy-oracle",
        feature = "pruning-conformance",
        any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "tvos",
            target_os = "watchos"
        )
    ))]
    #[test]
    fn salej_union_horizon_publication_never_replaces_an_existing_artifact() {
        let root = tempfile::tempdir()
            .expect("create UNION_HORIZON no-replace test directory")
            .keep();
        let first_temporary = std::ffi::OsStr::new(".tmp-first.json");
        let second_temporary = std::ffi::OsStr::new(".tmp-second.json");
        let target_name = std::ffi::OsStr::new("sealed.json");
        let directory = crate::artifact::PinnedDirectory::ensure_path(&root)
            .expect("pin no-replace test directory");
        directory
            .publish_unique_no_clobber(first_temporary, target_name, b"sealed")
            .expect("publish initial sealed artifact");

        assert!(
            directory
                .publish_unique_no_clobber(second_temporary, target_name, b"candidate")
                .is_err(),
            "UNION_HORIZON publication must fail when the destination already exists",
        );
        assert_eq!(
            directory
                .read_regular_bounded(target_name, 6)
                .expect("read preserved no-replace target"),
            b"sealed",
        );
        assert_eq!(
            directory
                .entry_names(4)
                .expect("list pinned no-replace directory"),
            [std::ffi::OsString::from(target_name)]
                .into_iter()
                .collect(),
            "failed no-replace publication must not create a staging entry",
        );
    }

    #[cfg(all(
        feature = "tantivy-oracle",
        feature = "pruning-conformance",
        any(target_os = "linux", target_os = "macos")
    ))]
    #[test]
    fn salej_union_horizon_publication_rejects_symlinks_and_survives_root_swap() {
        use std::os::unix::fs::symlink;

        let parent = tempfile::tempdir().expect("create pinned-directory test parent");
        let real = parent.path().join("real");
        let decoy = parent.path().join("decoy");
        std::fs::create_dir(&real).expect("create real evidence root");
        std::fs::create_dir(&decoy).expect("create decoy evidence root");

        let ancestor_link = parent.path().join("ancestor-link");
        symlink(&real, &ancestor_link).expect("create hostile ancestor symlink");
        assert!(
            crate::artifact::PinnedDirectory::ensure_path(&ancestor_link.join("child")).is_err(),
            "artifact roots must reject ancestor symlinks",
        );
        let final_target = real.join("final-target");
        std::fs::create_dir(&final_target).expect("create final symlink target");
        let final_link = real.join("final-link");
        symlink(&final_target, &final_link).expect("create hostile final symlink");
        assert!(
            crate::artifact::PinnedDirectory::ensure_path(&final_link).is_err(),
            "artifact roots must reject final-component symlinks",
        );

        let directory =
            crate::artifact::PinnedDirectory::ensure_path(&real).expect("pin real evidence root");
        let moved = parent.path().join("moved-real");
        std::fs::rename(&real, &moved).expect("move pinned root away from ambient path");
        std::fs::rename(&decoy, &real).expect("replace ambient root with decoy");
        let temporary_name = std::ffi::OsStr::new(".tmp-root-swap.json");
        let target_name = std::ffi::OsStr::new("sealed.json");
        directory
            .publish_unique_no_clobber(temporary_name, target_name, b"descriptor-bound")
            .expect("publish through the retained directory descriptor");
        assert_eq!(
            directory
                .read_regular_bounded(target_name, 16)
                .expect("reread through retained directory descriptor"),
            b"descriptor-bound",
        );
        assert!(
            !real.join(target_name).exists(),
            "ambient replacement root must not receive the sealed artifact",
        );
        assert_eq!(
            std::fs::read(moved.join(target_name)).expect("read descriptor-bound path witness"),
            b"descriptor-bound",
        );

        let external = parent.path().join("external.json");
        std::fs::write(&external, b"hostile").expect("write hostile symlink target");
        let symlink_name = std::ffi::OsStr::new("symlink.json");
        symlink(&external, moved.join(symlink_name)).expect("create hostile evidence symlink");
        assert!(
            directory.read_regular_bounded(symlink_name, 16).is_err(),
            "artifact reread must require a regular no-follow file",
        );
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    #[test]
    fn salej_union_horizon_late_winner_matches_tantivy_across_fresh_segment_shapes() {
        let fixture = make_union_horizon_fixture();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let artifact =
                run_union_horizon_layout_matrix(&cx, &fixture, UnionHorizonProofKind::LateWinner)
                    .await;
            artifact.verify();

            let mut stale_seal = artifact.clone();
            stale_seal.artifact_sha256 = "0".repeat(64);
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| stale_seal.verify()))
                    .is_err(),
                "UNION_HORIZON stale semantic seal must fail on an otherwise valid artifact",
            );

            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                tampered.proofs[0].comparisons[0].engines.semantic_contract = None;
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                tampered.proofs[0].comparisons[0]
                    .engines
                    .subject
                    .crate_version = "0.0.0".to_owned();
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                tampered.proofs[0].comparisons[0]
                    .engines
                    .subject
                    .config_hash = "alternate-fanout".to_owned();
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                tampered.proofs[0].build_identity.test_executable_sha256 = "0".repeat(64);
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                tampered.proofs[0].build_identity.test_executable_byte_len += 1;
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                tampered.proofs[0]
                    .build_identity
                    .rustc_version_verbose
                    .push_str("hostile compiler rewrite");
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                tampered.proofs[0].build_identity.target_triple = "unknown-target".to_owned();
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                tampered.proofs[0].build_identity.cargo_profile = "hostile".to_owned();
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                let identity = &mut tampered.proofs[0].build_identity;
                identity.enabled_features.push("hostile_feature".to_owned());
                identity.enabled_features.sort();
                identity.enabled_features_sha256 =
                    sha256_hex(identity.enabled_features.join("\n").as_bytes());
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                let source_verification =
                    &mut tampered.proofs[0].build_identity.source_verification;
                *source_verification = match source_verification {
                    UnionHorizonSourceVerification::GitCheckoutVerified => {
                        UnionHorizonSourceVerification::ExplicitUnverified
                    }
                    UnionHorizonSourceVerification::ExplicitUnverified
                    | UnionHorizonSourceVerification::Unavailable => {
                        UnionHorizonSourceVerification::GitCheckoutVerified
                    }
                };
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                tampered.proofs[0].comparisons[0].case.metadata.corpus_hash = Some("d".repeat(64));
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                let run = &mut tampered.proofs[0].comparisons[0];
                run.comparison.subject.doc_count = 42;
                run.comparison.oracle.doc_count = 42;
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                tampered.proofs[0].traced_results[0].hits[0].global_docid = 8_999;
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                let run = &mut tampered.proofs[0].comparisons[1];
                let (first, remaining) = run.comparison.subject.hits.split_at_mut(1);
                std::mem::swap(&mut first[0].doc_id, &mut remaining[0].doc_id);
                std::mem::swap(
                    &mut first[0].native_tie_key,
                    &mut remaining[0].native_tie_key,
                );
                run.comparison = compare_observations(
                    run.comparison.subject.clone(),
                    run.comparison.oracle.clone(),
                    run.comparator_config,
                )
                .expect("recompute hostile UNION_HORIZON rank mismatch");
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                tampered.proofs[1].topology.tantivy_segments[0].max_doc += 1;
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                let NativeTieKey::QuillDocId { doc_id } =
                    &mut tampered.proofs[0].comparisons[0].comparison.subject.hits[0]
                        .native_tie_key
                else {
                    panic!("UNION_HORIZON hostile fixture lost its Quill native key")
                };
                *doc_id += 1;
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                let mut trace =
                    serde_json::to_value(&tampered.proofs[2].complete_pruning_traces[0])
                        .expect("encode hostile UNION_HORIZON non-target trace");
                trace["segments"][0]["refills"][0]["candidate_docs"] =
                    serde_json::Value::from(0_u64);
                tampered.proofs[2].complete_pruning_traces[0] = serde_json::from_value(trace)
                    .expect("decode hostile UNION_HORIZON non-target trace");
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                let forged_revision = "c".repeat(40);
                for proof in &mut tampered.proofs {
                    for run in &mut proof.comparisons {
                        run.engines
                            .subject
                            .source_revision
                            .clone_from(&forged_revision);
                        run.engines
                            .oracle
                            .source_revision
                            .clone_from(&forged_revision);
                    }
                }
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                let producer_revision = tampered.proofs[0]
                    .build_identity
                    .source_git_revision
                    .clone();
                tampered.proofs[0].comparisons[0]
                    .engines
                    .oracle
                    .source_revision = producer_revision;
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                tampered.proofs[0].comparisons[0]
                    .engines
                    .oracle
                    .source_dirty = true;
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                tampered.proofs[0].oracle_dependency.tantivy_version = "0.0.0".to_owned();
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                tampered.proofs[0].oracle_dependency.tantivy_checksum_sha256 = "0".repeat(64);
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                tampered.proofs[0].oracle_dependency.lexical_package = "other".to_owned();
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                tampered.proofs[0].oracle_dependency.lexical_package_version = "0.0.0".to_owned();
            });
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                tampered.proofs[0]
                    .oracle_dependency
                    .pinned_lexical_contract_revision = "0".repeat(40);
            });

            let mut republished = artifact.clone();
            republished.run_id = "independent-publication".to_owned();
            assert_eq!(
                republished.preimage_sha256(),
                artifact.artifact_sha256,
                "UNION_HORIZON semantic identity must not depend on CI run metadata",
            );
            republished.artifact_sha256 = republished.preimage_sha256();
            republished.verify();

            let raw_bytes =
                serde_json::to_vec_pretty(&artifact).expect("serialize UNION_HORIZON seal test");
            let raw_file_sha256 = sha256_hex(&raw_bytes);
            assert_eq!(
                decode_union_horizon_artifact_bytes(&raw_bytes, &raw_file_sha256),
                artifact,
            );
            let mut tampered: serde_json::Value =
                serde_json::from_slice(&raw_bytes).expect("decode UNION_HORIZON tamper fixture");
            tampered["proofs"][0]["comparisons"][0]["engines"]["oracle"]["unsealed_extension"] =
                serde_json::Value::Bool(true);
            let tampered_bytes =
                serde_json::to_vec_pretty(&tampered).expect("encode UNION_HORIZON tamper fixture");
            assert!(
                std::panic::catch_unwind(|| {
                    decode_union_horizon_artifact_bytes(&tampered_bytes, &raw_file_sha256);
                })
                .is_err(),
                "UNION_HORIZON raw-file receipt must reject even a semantically ignored nested field",
            );
            let tampered_raw_file_sha256 = sha256_hex(&tampered_bytes);
            assert!(
                std::panic::catch_unwind(|| {
                    decode_union_horizon_artifact_bytes(&tampered_bytes, &tampered_raw_file_sha256);
                })
                .is_err(),
                "UNION_HORIZON strict typed reload must reject an unknown nested field even when its raw-file hash is recomputed",
            );
            let canonical = String::from_utf8(raw_bytes.clone())
                .expect("UNION_HORIZON canonical artifact is UTF-8");
            let encoded_run_id =
                serde_json::to_string(&artifact.run_id).expect("encode UNION_HORIZON run ID");
            let run_id_field = format!("\"run_id\": {encoded_run_id},");
            let duplicate_run_id = canonical.replacen(
                &run_id_field,
                &format!("{run_id_field}\n  {run_id_field}"),
                1,
            );
            assert_ne!(duplicate_run_id, canonical);
            let duplicate_run_id_bytes = duplicate_run_id.into_bytes();
            let duplicate_run_id_sha256 = sha256_hex(&duplicate_run_id_bytes);
            assert!(
                std::panic::catch_unwind(|| {
                    decode_union_horizon_artifact_bytes(
                        &duplicate_run_id_bytes,
                        &duplicate_run_id_sha256,
                    );
                })
                .is_err(),
                "UNION_HORIZON exact-byte reload must reject duplicate known JSON fields",
            );
            let compact_bytes =
                serde_json::to_vec(&artifact).expect("serialize compact UNION_HORIZON artifact");
            let compact_sha256 = sha256_hex(&compact_bytes);
            assert!(
                std::panic::catch_unwind(|| {
                    decode_union_horizon_artifact_bytes(&compact_bytes, &compact_sha256);
                })
                .is_err(),
                "UNION_HORIZON exact-byte reload must reject alternate JSON formatting",
            );
        });
    }

    #[cfg(all(feature = "tantivy-oracle", feature = "pruning-conformance"))]
    #[test]
    fn salej_union_horizon_cutoff_ties_are_exact_or_registered_across_fresh_segment_shapes() {
        let fixture = make_union_horizon_tie_fixture();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let artifact =
                run_union_horizon_layout_matrix(&cx, &fixture, UnionHorizonProofKind::TieMatrix)
                    .await;
            artifact.verify();
            assert_union_horizon_artifact_rejected(&artifact, |tampered| {
                let run = &mut tampered.proofs[1].comparisons[1];
                let original_score_bits = run.comparison.subject.hits[0].score_bits;
                let original_score = f32::from_bits(original_score_bits);
                assert!(
                    original_score.is_finite() && original_score.is_sign_positive(),
                    "UNION_HORIZON hostile tie-score fixture must be positive and finite",
                );
                run.comparison.subject.hits[0].score_bits = original_score_bits
                    .checked_add(1)
                    .expect("positive finite score bits have one higher finite encoding");
                run.comparison = compare_observations(
                    run.comparison.subject.clone(),
                    run.comparison.oracle.clone(),
                    run.comparator_config,
                )
                .expect("recompute hostile UNION_HORIZON tie-matrix score mismatch");
            });
        });
    }

    #[cfg(all(
        feature = "tantivy-oracle",
        feature = "pruning-conformance",
        any(target_os = "linux", target_os = "macos")
    ))]
    fn assert_union_horizon_completion_rejected(
        manifest: &UnionHorizonCompletionManifest,
        mutate: impl FnOnce(&mut UnionHorizonCompletionManifest),
    ) {
        let mut tampered = manifest.clone();
        mutate(&mut tampered);
        tampered.manifest_sha256 = tampered.preimage_sha256();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tampered.verify())).is_err(),
            "tampered UNION_HORIZON completion manifest unexpectedly verified",
        );
    }

    #[cfg(all(
        feature = "tantivy-oracle",
        feature = "pruning-conformance",
        any(target_os = "linux", target_os = "macos")
    ))]
    fn write_union_horizon_bundle_fixture(
        root: &std::path::Path,
        manifest: &UnionHorizonCompletionManifest,
    ) -> std::ffi::OsString {
        std::fs::create_dir(root).expect("create hostile completion-bundle fixture root");
        for entry in &manifest.artifacts {
            let bytes = serde_json::to_vec_pretty(&entry.artifact)
                .expect("serialize hostile completion-bundle proof fixture");
            assert_eq!(sha256_hex(&bytes), entry.raw_file_sha256);
            std::fs::write(root.join(&entry.filename), bytes)
                .expect("write hostile completion-bundle proof fixture");
        }
        let completion_bytes = serde_json::to_vec_pretty(manifest)
            .expect("serialize hostile completion-bundle manifest fixture");
        let completion_name = std::ffi::OsString::from(format!(
            "completion-{}-{}.json",
            manifest.manifest_sha256,
            sha256_hex(&completion_bytes),
        ));
        std::fs::write(root.join(&completion_name), completion_bytes)
            .expect("write hostile completion-bundle manifest fixture");
        completion_name
    }

    #[cfg(all(
        feature = "tantivy-oracle",
        feature = "pruning-conformance",
        any(target_os = "linux", target_os = "macos")
    ))]
    fn assert_union_horizon_bundle_validation_panics(
        directory: &crate::artifact::PinnedDirectory,
        manifest: &UnionHorizonCompletionManifest,
        completion_name: &std::ffi::OsStr,
        context: &str,
    ) {
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                validate_union_horizon_completed_bundle(directory, manifest, completion_name);
            }))
            .is_err(),
            "hostile UNION_HORIZON bundle unexpectedly verified: {context}",
        );
    }

    #[cfg(all(
        feature = "tantivy-oracle",
        feature = "pruning-conformance",
        any(target_os = "linux", target_os = "macos")
    ))]
    #[test]
    #[ignore = "requires an isolated, clean-Git evidence publication environment"]
    fn salej_complete_union_horizon_diagnostic_bundle() {
        let build_identity = union_horizon_validated_build_identity();
        assert!(
            union_horizon_identity_is_publishable(&build_identity),
            "Salej completion requires a clean Linux Git-verified build; diagnostic Gitless snapshots must fail before executing the evidence matrix",
        );
        salej_runtime_identity_rejects_revision_or_dirty_state_rewrites();
        salej_union_horizon_publication_never_replaces_an_existing_artifact();
        salej_union_horizon_publication_rejects_symlinks_and_survives_root_swap();
        salej_union_horizon_late_winner_matches_tantivy_across_fresh_segment_shapes();
        salej_union_horizon_cutoff_ties_are_exact_or_registered_across_fresh_segment_shapes();
        let (completion, manifest) = publish_union_horizon_completion_manifest();
        assert!(completion.path.ends_with(".json"));
        assert_lower_hex(
            &completion.raw_file_sha256,
            64,
            "UNION_HORIZON completion raw identity",
        );
        assert!(completion.byte_len > 0);
        manifest.verify();

        assert_union_horizon_completion_rejected(&manifest, |tampered| {
            tampered.build_identity.test_executable_sha256 = "0".repeat(64);
        });
        assert_union_horizon_completion_rejected(&manifest, |tampered| {
            tampered.artifacts[0].raw_file_sha256 = "0".repeat(64);
        });
        assert_union_horizon_completion_rejected(&manifest, |tampered| {
            tampered.artifacts[1].proof_kind = UnionHorizonProofKind::LateWinner;
        });

        let wrong_name_parent = tempfile::tempdir().expect("wrong-name bundle parent");
        let wrong_name_root = wrong_name_parent.path().join("bundle");
        let wrong_name_completion = write_union_horizon_bundle_fixture(&wrong_name_root, &manifest);
        let wrong_name_directory = crate::artifact::PinnedDirectory::ensure_path(&wrong_name_root)
            .expect("pin wrong-name bundle fixture");
        std::fs::rename(
            wrong_name_root.join(&manifest.artifacts[0].filename),
            wrong_name_root.join("same-count-substitution.json"),
        )
        .expect("rename one proof without changing bundle cardinality");
        assert_union_horizon_bundle_validation_panics(
            &wrong_name_directory,
            &manifest,
            &wrong_name_completion,
            "same-count filename substitution",
        );

        let mutated_parent = tempfile::tempdir().expect("mutated bundle parent");
        let mutated_root = mutated_parent.path().join("bundle");
        let mutated_completion = write_union_horizon_bundle_fixture(&mutated_root, &manifest);
        let mutated_directory = crate::artifact::PinnedDirectory::ensure_path(&mutated_root)
            .expect("pin mutated bundle fixture");
        std::fs::write(mutated_root.join(&manifest.artifacts[0].filename), b"{}")
            .expect("mutate proof bytes after pinning");
        assert_union_horizon_bundle_validation_panics(
            &mutated_directory,
            &manifest,
            &mutated_completion,
            "proof content mutation",
        );

        let missing_parent = tempfile::tempdir().expect("missing bundle parent");
        let missing_root = missing_parent.path().join("bundle");
        let missing_completion = write_union_horizon_bundle_fixture(&missing_root, &manifest);
        let missing_directory = crate::artifact::PinnedDirectory::ensure_path(&missing_root)
            .expect("pin missing-entry bundle fixture");
        std::fs::rename(
            missing_root.join(&manifest.artifacts[0].filename),
            missing_parent.path().join("moved-proof.json"),
        )
        .expect("move proof outside the completed bundle");
        assert_union_horizon_bundle_validation_panics(
            &missing_directory,
            &manifest,
            &missing_completion,
            "missing proof",
        );

        let extra_parent = tempfile::tempdir().expect("extra bundle parent");
        let extra_root = extra_parent.path().join("bundle");
        let extra_completion = write_union_horizon_bundle_fixture(&extra_root, &manifest);
        let extra_directory = crate::artifact::PinnedDirectory::ensure_path(&extra_root)
            .expect("pin extra-entry bundle fixture");
        std::fs::write(extra_root.join("extra.json"), b"diagnostic")
            .expect("add extra completed-bundle entry");
        assert_union_horizon_bundle_validation_panics(
            &extra_directory,
            &manifest,
            &extra_completion,
            "extra entry",
        );
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn scalar_g1a_boosted_should_permutations_and_duplicate_all_match_tantivy_bits() {
        let snapshot = RepositorySnapshot::from_entries(
            "boosted-should-order-regression",
            [
                RepositoryEntry {
                    relative_path: std::path::PathBuf::from("docs/01-short.txt"),
                    bytes: b"alpha beta gamma".to_vec(),
                },
                RepositoryEntry {
                    relative_path: std::path::PathBuf::from("docs/02-medium.txt"),
                    bytes: b"alpha beta gamma filler filler filler".to_vec(),
                },
                RepositoryEntry {
                    relative_path: std::path::PathBuf::from("docs/03-long.txt"),
                    bytes: b"alpha beta gamma filler filler filler filler filler filler".to_vec(),
                },
            ],
        )
        .expect("boost-order corpus snapshot");
        let corpus_hash = snapshot.manifest.manifest_hash().expect("corpus hash");
        // Every term has equal df/tf, so scorer costs tie. These two source
        // orders reach the parity-pinned union as `(alpha + gamma) + beta`
        // versus `(alpha + beta) + gamma`; boosts 2/5/120 differ by one score
        // bit on each of the three fieldnorm lanes above.
        let query_suite = GeneratedQuerySuite::from_cases(
            QueryGeneratorSpec {
                seed: 0x6202,
                default_limit: 10,
                include_shared_relevance_queries: false,
            },
            &corpus_hash,
            [
                (
                    "boosted-should-beta-before-gamma",
                    "alpha^2 beta^5 gamma^120",
                ),
                (
                    "boosted-should-gamma-before-beta",
                    "alpha^2 gamma^120 beta^5",
                ),
                ("outer-boosted-duplicate-all", "(* AND *)^2"),
            ]
            .into_iter()
            .map(|(id, query)| GeneratedQueryCase {
                id: id.to_owned(),
                syntax: QuerySyntax::Default,
                query_kind: GeneratedQueryKind::Boolean,
                query: query.to_owned(),
                limit: 10,
                offset: 0,
                count_requested: true,
                filters: crate::generator::GeneratedQueryFilters::default(),
                expected_divergence: None,
                source: "runner.rs score-order and duplicate-All regression".to_owned(),
            })
            .collect(),
        )
        .expect("boost-order query suite");
        let fixture = Fixture {
            documents: snapshot.documents,
            corpus_manifest: snapshot.manifest,
            corpus_hash,
            query_suite,
        };
        let root = tempfile::tempdir()
            .expect("boost-order regression tempdir")
            .keep();

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let report = run_scalar_g1a_deterministic_regression(&cx, &root, &fixture)
                .await
                .unwrap_or_else(|error| {
                    panic!(
                        "corpus_hash={} boosted-Should campaign_error={error}",
                        fixture.corpus_hash
                    )
                });
            assert_eq!(report.cases.len(), 3);

            let mut permutation_score_bits = Vec::new();
            for case in &report.cases {
                assert_eq!(case.disposition, CampaignDisposition::Exact, "{case:?}");
                assert_eq!(case.comparison_status, Some(ComparisonStatus::Exact));
                assert_eq!(case.rank_class, Some(RankClass::RankExact));

                let object_hash = case
                    .artifact_hash
                    .as_deref()
                    .expect("exact boosted-Should artifact hash");
                let object_path = root.join("objects").join(format!("{object_hash}.json"));
                let object: ArtifactObject = serde_json::from_slice(
                    &std::fs::read(&object_path).expect("boosted-Should artifact bytes"),
                )
                .expect("boosted-Should artifact object");
                let subject_hits = object
                    .comparison
                    .subject
                    .hits
                    .iter()
                    .map(|hit| (hit.doc_id.as_str(), hit.score_bits))
                    .collect::<Vec<_>>();
                let oracle_hits = object
                    .comparison
                    .oracle
                    .hits
                    .iter()
                    .map(|hit| (hit.doc_id.as_str(), hit.score_bits))
                    .collect::<Vec<_>>();
                assert!(!subject_hits.is_empty(), "{} was vacuous", case.case_id);
                assert_eq!(
                    subject_hits, oracle_hits,
                    "{} must preserve Tantivy documents and exact f32 score bits",
                    case.case_id
                );
                if case.case_id.starts_with("boosted-should-") {
                    permutation_score_bits.push(
                        object
                            .comparison
                            .subject
                            .hits
                            .iter()
                            .map(|hit| hit.score_bits)
                            .collect::<Vec<_>>(),
                    );
                }
            }
            assert_eq!(permutation_score_bits.len(), 2);
            assert_eq!(permutation_score_bits[0].len(), 3);
            assert_eq!(permutation_score_bits[1].len(), 3);
            assert!(
                permutation_score_bits[0]
                    .iter()
                    .zip(&permutation_score_bits[1])
                    .all(|(left, right)| left != right),
                "the clause permutations must change every order-sensitive f32 score",
            );
        });
    }

    #[test]
    fn campaign_run_id_is_single_use_before_engine_ingest() {
        let fixture = make_fixture();
        let temp = tempfile::tempdir().expect("tempdir");
        let campaign = runner(
            temp.path(),
            CampaignSelection::CaseIds {
                ids: vec!["term".to_owned()],
            },
            DivergenceRegistry::default(),
        );
        let mut first_subject = ScriptedEngine::new(subject_descriptor(), BTreeMap::new());
        let mut first_oracle = ScriptedEngine::new(oracle_descriptor(), BTreeMap::new());
        let error_behavior = BTreeMap::from([("term".to_owned(), ScriptedBehavior::Error)]);
        let mut retry_subject = ScriptedEngine::new(subject_descriptor(), error_behavior.clone());
        let mut retry_oracle = ScriptedEngine::new(oracle_descriptor(), error_behavior);

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            campaign
                .run(
                    &cx,
                    "single-use",
                    &mut first_subject,
                    &mut first_oracle,
                    &fixture.documents,
                    &fixture.corpus_manifest,
                    &fixture.query_suite,
                )
                .await
                .expect("first campaign");

            let error = campaign
                .run(
                    &cx,
                    "single-use",
                    &mut retry_subject,
                    &mut retry_oracle,
                    &fixture.documents,
                    &fixture.corpus_manifest,
                    &fixture.query_suite,
                )
                .await
                .expect_err("run ID reuse must fail before query execution");
            assert!(matches!(error, GauntletError::RunManifestConflict { .. }));
            assert_eq!(retry_subject.index_calls.load(Ordering::Relaxed), 0);
            assert_eq!(retry_oracle.index_calls.load(Ordering::Relaxed), 0);
            assert_eq!(retry_subject.observe_calls.load(Ordering::Relaxed), 0);
            assert_eq!(retry_oracle.observe_calls.load(Ordering::Relaxed), 0);
        });
    }

    #[test]
    fn campaign_run_id_rejects_a_changed_selection() {
        let fixture = make_fixture();
        let temp = tempfile::tempdir().expect("tempdir");
        let first = runner(
            temp.path(),
            CampaignSelection::CaseIds {
                ids: vec!["term".to_owned()],
            },
            DivergenceRegistry::default(),
        );
        let changed = runner(
            temp.path(),
            CampaignSelection::CaseIds {
                ids: vec!["multi-term".to_owned()],
            },
            DivergenceRegistry::default(),
        );
        let mut first_subject = ScriptedEngine::new(subject_descriptor(), BTreeMap::new());
        let mut first_oracle = ScriptedEngine::new(oracle_descriptor(), BTreeMap::new());
        let mut changed_subject = ScriptedEngine::new(subject_descriptor(), BTreeMap::new());
        let mut changed_oracle = ScriptedEngine::new(oracle_descriptor(), BTreeMap::new());

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            first
                .run(
                    &cx,
                    "selection-reuse",
                    &mut first_subject,
                    &mut first_oracle,
                    &fixture.documents,
                    &fixture.corpus_manifest,
                    &fixture.query_suite,
                )
                .await
                .expect("first selection");
            let error = changed
                .run(
                    &cx,
                    "selection-reuse",
                    &mut changed_subject,
                    &mut changed_oracle,
                    &fixture.documents,
                    &fixture.corpus_manifest,
                    &fixture.query_suite,
                )
                .await
                .expect_err("selection change cannot reuse run ID");
            assert!(matches!(error, GauntletError::RunManifestConflict { .. }));
            assert_eq!(changed_subject.index_calls.load(Ordering::Relaxed), 0);
            assert_eq!(changed_oracle.index_calls.load(Ordering::Relaxed), 0);
        });
    }

    #[test]
    fn register_match_is_required_and_cannot_mask_a_rank_failure() {
        let mut fixture = make_fixture();
        let term = fixture
            .query_suite
            .cases
            .iter_mut()
            .find(|case| case.id == "term")
            .expect("term case");
        term.expected_divergence = Some("DIV-004".to_owned());
        fixture.query_suite = GeneratedQuerySuite::from_cases(
            fixture.query_suite.manifest.spec.clone(),
            &fixture.corpus_hash,
            fixture.query_suite.cases,
        )
        .expect("rebuilt suite");
        let registry = DivergenceRegistry::new(vec![DivergenceRegisterEntry {
            id: "DIV-004".to_owned(),
            class: DivergenceClass::OversizedQueryToken,
            fixture_id: "term".to_owned(),
            mismatch_signatures: vec![oversized_query_signature()],
            decision: DivergenceRegisterDecision::Accept,
            root_cause: "query token exceeds the symmetric admission bound".to_owned(),
            consumer_impact: "programmatic ASTs can observe MatchNone lowering".to_owned(),
            reviewer: "fresh-eyes-agent".to_owned(),
            reviewed_at: "2026-07-18".to_owned(),
        }])
        .expect("registry");
        let mut oversized_review = registry.entries[0].clone();
        oversized_review.reviewer = "r".repeat(MAX_DIVERGENCE_REVIEWER_BYTES + 1);
        assert!(DivergenceRegistry::new(vec![oversized_review]).is_err());

        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().to_path_buf();
        let behavior = BTreeMap::from([("term".to_owned(), ScriptedBehavior::OversizedQueryToken)]);
        let mut subject = ScriptedEngine::new(subject_descriptor(), behavior.clone());
        let mut oracle = ScriptedEngine::new(oracle_descriptor(), behavior);
        let campaign = runner(
            &root,
            CampaignSelection::CaseIds {
                ids: vec!["term".to_owned()],
            },
            registry.clone(),
        );
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let report = campaign
                .run(
                    &cx,
                    "registered",
                    &mut subject,
                    &mut oracle,
                    &fixture.documents,
                    &fixture.corpus_manifest,
                    &fixture.query_suite,
                )
                .await
                .expect("registered report");
            assert!(report.passed);
            assert_eq!(
                report.cases[0].disposition,
                CampaignDisposition::RegisterClassified
            );
            assert_eq!(
                report.cases[0]
                    .registered_divergence
                    .as_ref()
                    .map(|entry| entry.reviewer.as_str()),
                Some("fresh-eyes-agent")
            );
            assert!(root.join("campaigns/registered/report.json").is_file());
            let mut missing_registry = report.clone();
            missing_registry.divergence_registry = DivergenceRegistry::default();
            assert!(missing_registry.canonical_bytes().is_err());
        });

        let mut fixture = make_fixture();
        let term = fixture
            .query_suite
            .cases
            .iter_mut()
            .find(|case| case.id == "term")
            .expect("term case");
        term.expected_divergence = Some("DIV-004".to_owned());
        fixture.query_suite = GeneratedQuerySuite::from_cases(
            fixture.query_suite.manifest.spec.clone(),
            &fixture.corpus_hash,
            fixture.query_suite.cases,
        )
        .expect("rebuilt duplicate suite");
        let temp = tempfile::tempdir().expect("tempdir");
        let behavior = BTreeMap::from([(
            "term".to_owned(),
            ScriptedBehavior::DuplicateOversizedQueryToken,
        )]);
        let mut subject = ScriptedEngine::new(subject_descriptor(), behavior.clone());
        let mut oracle = ScriptedEngine::new(oracle_descriptor(), behavior);
        let campaign = runner(
            temp.path(),
            CampaignSelection::CaseIds {
                ids: vec!["term".to_owned()],
            },
            registry.clone(),
        );
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let report = campaign
                .run(
                    &cx,
                    "duplicate-register-shape",
                    &mut subject,
                    &mut oracle,
                    &fixture.documents,
                    &fixture.corpus_manifest,
                    &fixture.query_suite,
                )
                .await
                .expect("duplicate shape report");
            assert_eq!(
                report.cases[0].disposition,
                CampaignDisposition::Unclassified
            );
        });

        let mut fixture = make_fixture();
        fixture.query_suite.cases[0].expected_divergence = Some("DIV-004".to_owned());
        let protected_id = fixture.query_suite.cases[0].id.clone();
        let query_spec = fixture.query_suite.manifest.spec.clone();
        fixture.query_suite = GeneratedQuerySuite::from_cases(
            query_spec,
            &fixture.corpus_hash,
            fixture.query_suite.cases,
        )
        .expect("rebuilt suite");
        let temp = tempfile::tempdir().expect("tempdir");
        let behavior = BTreeMap::from([(protected_id.clone(), ScriptedBehavior::RankMismatch)]);
        let mut subject = ScriptedEngine::new(subject_descriptor(), behavior.clone());
        let mut oracle = ScriptedEngine::new(oracle_descriptor(), behavior);
        let campaign = runner(
            temp.path(),
            CampaignSelection::CaseIds {
                ids: vec![protected_id],
            },
            registry,
        );
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let report = campaign
                .run(
                    &cx,
                    "masked-rank-failure",
                    &mut subject,
                    &mut oracle,
                    &fixture.documents,
                    &fixture.corpus_manifest,
                    &fixture.query_suite,
                )
                .await
                .expect("failed report");
            assert!(!report.passed);
            assert_eq!(
                report.cases[0].disposition,
                CampaignDisposition::Unclassified
            );
        });
    }

    #[test]
    fn repeated_mismatches_deduplicate_and_query_errors_do_not_abort_later_cases() {
        let fixture = make_fixture();
        let selected = vec!["term".to_owned(), "multi-term".to_owned()];
        let behavior = BTreeMap::from([
            (selected[0].clone(), ScriptedBehavior::RankMismatch),
            (selected[1].clone(), ScriptedBehavior::RankMismatch),
        ]);
        let mut subject = ScriptedEngine::new(subject_descriptor(), behavior.clone());
        let mut oracle = ScriptedEngine::new(oracle_descriptor(), behavior);
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().to_path_buf();
        let campaign = runner(
            &root,
            CampaignSelection::CaseIds { ids: selected },
            DivergenceRegistry::default(),
        );
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let report = campaign
                .run(
                    &cx,
                    "dedup",
                    &mut subject,
                    &mut oracle,
                    &fixture.documents,
                    &fixture.corpus_manifest,
                    &fixture.query_suite,
                )
                .await
                .expect("dedup report");
            assert!(!report.passed);
            assert_eq!(report.mismatches.len(), 1);
            assert_eq!(report.mismatches[0].occurrence_count, 2);
            assert_eq!(report.mismatches[0].case_ids.len(), 2);
            assert_eq!(report.mismatches[0].signature.len(), 64);
            assert!(root.join("campaigns/dedup/report.json").is_file());
            let mut wrong_mismatches = report.clone();
            wrong_mismatches.mismatches[0].occurrence_count += 1;
            assert!(
                ArtifactStore::new(&root)
                    .complete_campaign(&wrong_mismatches)
                    .is_err()
            );
        });

        let fixture = make_fixture();
        let selected = vec!["term".to_owned(), "multi-term".to_owned()];
        let mut subject = ScriptedEngine::new(
            subject_descriptor(),
            BTreeMap::from([("term".to_owned(), ScriptedBehavior::Error)]),
        );
        let mut oracle = ScriptedEngine::new(oracle_descriptor(), BTreeMap::new());
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().to_path_buf();
        let campaign = runner(
            &root,
            CampaignSelection::CaseIds { ids: selected },
            DivergenceRegistry::default(),
        );
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let report = campaign
                .run(
                    &cx,
                    "continue-after-error",
                    &mut subject,
                    &mut oracle,
                    &fixture.documents,
                    &fixture.corpus_manifest,
                    &fixture.query_suite,
                )
                .await
                .expect("continued report");
            assert!(!report.passed);
            assert_eq!(report.cases.len(), 2);
            assert_eq!(
                report.cases[0].disposition,
                CampaignDisposition::InfrastructureError
            );
            assert_eq!(report.cases[1].disposition, CampaignDisposition::Exact);
            assert_eq!(subject.observe_calls.load(Ordering::Relaxed), 2);
            assert_eq!(oracle.observe_calls.load(Ordering::Relaxed), 2);
            let report_path = root.join("campaigns/continue-after-error/report.json");
            assert!(report_path.is_file());
            let stored: CampaignReport = serde_json::from_slice(
                &std::fs::read(report_path).expect("stored infrastructure report"),
            )
            .expect("decode infrastructure report");
            assert!(!stored.passed);
            assert_eq!(stored.cases[0].reason, report.cases[0].reason);
        });
    }

    #[test]
    fn any_document_count_drift_fails_closed_without_persisting_untrusted_evidence() {
        let fixture = make_fixture();
        let selected = CampaignSelection::CaseIds {
            ids: vec!["term".to_owned()],
        };
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("asymmetric");
        let mut subject = ScriptedEngine::new(subject_descriptor(), BTreeMap::new());
        let mut oracle = ScriptedEngine::new(oracle_descriptor(), BTreeMap::new())
            .with_reported_doc_count(fixture.corpus_manifest.document_count + 1);
        let campaign = runner(&root, selected.clone(), DivergenceRegistry::default());
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let report = campaign
                .run(
                    &cx,
                    "asymmetric-doc-count",
                    &mut subject,
                    &mut oracle,
                    &fixture.documents,
                    &fixture.corpus_manifest,
                    &fixture.query_suite,
                )
                .await
                .expect("asymmetric drift report");
            assert!(!report.passed);
            assert_eq!(
                report.cases[0].disposition,
                CampaignDisposition::InfrastructureError
            );
            assert_eq!(
                report.cases[0].reason.as_deref(),
                Some("observation_document_count_drift")
            );
            assert!(report.cases[0].artifact_hash.is_none());
            let report_path = root.join("campaigns/asymmetric-doc-count/report.json");
            let stored: CampaignReport = serde_json::from_slice(
                &std::fs::read(report_path).expect("stored infrastructure report"),
            )
            .expect("decode infrastructure report");
            assert!(!stored.passed);
            assert_eq!(
                stored.cases[0].disposition,
                CampaignDisposition::InfrastructureError
            );
            assert_eq!(
                stored.cases[0].reason.as_deref(),
                Some("observation_document_count_drift")
            );
        });

        let fixture = make_fixture();
        let wrong_count = fixture.corpus_manifest.document_count + 1;
        let temp = tempfile::tempdir().expect("tempdir");
        let mut subject = ScriptedEngine::new(subject_descriptor(), BTreeMap::new())
            .with_reported_doc_count(wrong_count);
        let mut oracle = ScriptedEngine::new(oracle_descriptor(), BTreeMap::new())
            .with_reported_doc_count(wrong_count);
        let campaign = runner(temp.path(), selected.clone(), DivergenceRegistry::default());
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let report = campaign
                .run(
                    &cx,
                    "shared-doc-count-drift",
                    &mut subject,
                    &mut oracle,
                    &fixture.documents,
                    &fixture.corpus_manifest,
                    &fixture.query_suite,
                )
                .await
                .expect("shared drift report");
            assert!(!report.passed);
            assert_eq!(
                report.cases[0].disposition,
                CampaignDisposition::InfrastructureError
            );
            assert_eq!(
                report.cases[0].reason.as_deref(),
                Some("observation_document_count_drift")
            );
            assert!(report.cases[0].artifact_hash.is_none());
        });

        let fixture = make_fixture();
        let expected_count = fixture.corpus_manifest.document_count;
        let temp = tempfile::tempdir().expect("tempdir");
        let mut subject = ScriptedEngine::new(subject_descriptor(), BTreeMap::new())
            .with_reported_doc_count(expected_count + 1);
        let mut oracle = ScriptedEngine::new(oracle_descriptor(), BTreeMap::new())
            .with_reported_doc_count(expected_count + 2);
        let campaign = runner(temp.path(), selected, DivergenceRegistry::default());
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let report = campaign
                .run(
                    &cx,
                    "two-sided-doc-count-drift",
                    &mut subject,
                    &mut oracle,
                    &fixture.documents,
                    &fixture.corpus_manifest,
                    &fixture.query_suite,
                )
                .await
                .expect("two-sided drift report");
            assert_eq!(
                report.cases[0].disposition,
                CampaignDisposition::InfrastructureError
            );
            assert!(report.cases[0].artifact_hash.is_none());
        });
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn tantivy_campaign_adapter_enforces_the_one_shot_lifecycle() {
        let fixture = make_fixture();
        let lexical_revision = oracle_version_contract()
            .expect("oracle version contract")
            .lexical_git_revision;
        let contract = SemanticContract::shipping_default();
        let query = fixture
            .query_suite
            .cases
            .iter()
            .find(|case| case.id == "term")
            .expect("term query case")
            .clone();
        let mut evidence_case =
            DifferentialCase::new("tantivy-lifecycle-term", &query.query, query.limit);
        evidence_case.offset = query.offset;
        evidence_case.count_requested = query.count_requested;
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let mut before_begin = crate::TantivyOracle::in_memory(&lexical_revision, false)
                .expect("oracle before begin");
            assert!(matches!(
                before_begin.index_batch(&cx, &fixture.documents[..1]).await,
                Err(GauntletError::InvalidCampaign { .. })
            ));
            assert!(matches!(
                before_begin
                    .commit_corpus(&cx, &fixture.corpus_manifest, &contract)
                    .await,
                Err(GauntletError::InvalidCampaign { .. })
            ));
            assert!(matches!(
                before_begin
                    .observe_generated(&cx, &query, &evidence_case)
                    .await,
                Err(GauntletError::InvalidCampaign { .. })
            ));

            let mut before_commit = crate::TantivyOracle::in_memory(&lexical_revision, false)
                .expect("oracle before commit");
            before_commit
                .begin_corpus(&cx, &fixture.corpus_manifest, &contract)
                .await
                .expect("begin before-commit oracle");
            assert!(matches!(
                before_commit
                    .observe_generated(&cx, &query, &evidence_case)
                    .await,
                Err(GauntletError::InvalidCampaign { .. })
            ));
            before_commit.abort_corpus();
            assert!(
                before_commit
                    .begin_corpus(&cx, &fixture.corpus_manifest, &contract)
                    .await
                    .is_err(),
                "an aborted oracle must remain poisoned",
            );
            assert!(matches!(
                before_commit
                    .index_batch(&cx, &fixture.documents[..1])
                    .await,
                Err(GauntletError::InvalidCampaign { .. })
            ));
            assert!(matches!(
                before_commit
                    .observe_generated(&cx, &query, &evidence_case)
                    .await,
                Err(GauntletError::InvalidCampaign { .. })
            ));

            let mut oracle = crate::TantivyOracle::in_memory(&lexical_revision, false)
                .expect("committed in-memory oracle");
            oracle
                .begin_corpus(&cx, &fixture.corpus_manifest, &contract)
                .await
                .expect("fresh begin");
            for batch in fixture.documents.chunks(5) {
                oracle.index_batch(&cx, batch).await.expect("index batch");
            }
            let receipt = oracle
                .commit_corpus(&cx, &fixture.corpus_manifest, &contract)
                .await
                .expect("commit");
            assert_eq!(
                receipt.document_count,
                fixture.corpus_manifest.document_count
            );
            assert!(matches!(
                oracle.index_batch(&cx, &fixture.documents[..1]).await,
                Err(GauntletError::InvalidCampaign { .. })
            ));
            assert!(matches!(
                oracle
                    .commit_corpus(&cx, &fixture.corpus_manifest, &contract)
                    .await,
                Err(GauntletError::InvalidCampaign { .. })
            ));
            assert!(
                oracle
                    .begin_corpus(&cx, &fixture.corpus_manifest, &contract)
                    .await
                    .is_err()
            );
        });
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn tantivy_campaign_adapter_rejects_wrapped_delete_to_zero_history() {
        use frankensearch_core::LexicalSearch;

        let fixture = make_fixture();
        let lexical_revision = oracle_version_contract()
            .expect("oracle version contract")
            .lexical_git_revision;
        let contract = SemanticContract::shipping_default();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let mut oracle =
                crate::TantivyOracle::in_memory(&lexical_revision, false).expect("fresh oracle");
            oracle
                .index_documents(
                    &cx,
                    &[frankensearch_core::IndexableDocument::new(
                        "old-document",
                        "stale corpus statistics",
                    )],
                )
                .await
                .expect("index old document");
            oracle
                .index()
                .delete_document(&cx, "old-document")
                .await
                .expect("delete old document");
            oracle.index().commit(&cx).await.expect("commit deletion");
            assert_eq!(oracle.index().doc_count(), 0);
            assert!(
                oracle
                    .begin_corpus(&cx, &fixture.corpus_manifest, &contract)
                    .await
                    .is_err()
            );
        });
    }

    #[test]
    fn receipt_mismatch_is_a_campaign_error_and_beta_bound_is_pinned() {
        let fixture = make_fixture();
        let mut subject =
            ScriptedEngine::new(subject_descriptor(), BTreeMap::new()).with_tampered_receipt();
        let mut oracle = ScriptedEngine::new(oracle_descriptor(), BTreeMap::new());
        let temp = tempfile::tempdir().expect("tempdir");
        let campaign = runner(
            temp.path(),
            CampaignSelection::All,
            DivergenceRegistry::default(),
        );
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            assert!(
                campaign
                    .run(
                        &cx,
                        "bad-receipt",
                        &mut subject,
                        &mut oracle,
                        &fixture.documents,
                        &fixture.corpus_manifest,
                        &fixture.query_suite,
                    )
                    .await
                    .is_err()
            );
            assert_eq!(subject.abort_calls.load(Ordering::Relaxed), 1);
            assert_eq!(oracle.abort_calls.load(Ordering::Relaxed), 1);
        });

        let fixture = make_fixture();
        let mut subject = ScriptedEngine::new(subject_descriptor(), BTreeMap::new())
            .with_semantic_drift_on_commit();
        let mut oracle = ScriptedEngine::new(oracle_descriptor(), BTreeMap::new());
        let temp = tempfile::tempdir().expect("tempdir");
        let campaign = runner(
            temp.path(),
            CampaignSelection::All,
            DivergenceRegistry::default(),
        );
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            assert!(
                campaign
                    .run(
                        &cx,
                        "semantic-drift-on-commit",
                        &mut subject,
                        &mut oracle,
                        &fixture.documents,
                        &fixture.corpus_manifest,
                        &fixture.query_suite,
                    )
                    .await
                    .is_err()
            );
            assert_eq!(subject.abort_calls.load(Ordering::Relaxed), 1);
            assert_eq!(oracle.abort_calls.load(Ordering::Relaxed), 1);
        });

        let uniform = beta_posterior_lower_bound(0, 0, 0.95);
        assert!((uniform - 0.05).abs() < 1.0e-12);
        let one_success = beta_posterior_lower_bound(1, 1, 0.95);
        assert!((one_success - 0.05_f64.sqrt()).abs() < 1.0e-12);
        assert!(beta_posterior_lower_bound(9, 10, 0.95) > one_success);

        let contract = SemanticContract::shipping_default();
        assert_eq!(
            contract.analyzer_contract_hash,
            DEFAULT_ANALYZER_CONTRACT_HASH
        );
        assert_eq!(contract.schema_contract_hash, DEFAULT_SCHEMA_CONTRACT_HASH);

        let substitution = Divergence {
            class: DivergenceClass::RankMismatch,
            pointer: "/comparison/subject/hits/0".to_owned(),
            oracle: "oracle-doc@40400000".to_owned(),
            subject: "subject-doc@40400000".to_owned(),
        };
        let missing = Divergence {
            class: DivergenceClass::RankMismatch,
            pointer: "/comparison/subject/hits/0".to_owned(),
            oracle: "0".to_owned(),
            subject: "subject-doc@40400000".to_owned(),
        };
        assert_ne!(
            mismatch_signature(RankClass::RankMismatch, &substitution),
            mismatch_signature(RankClass::RankMismatch, &missing)
        );
    }

    // ==== Divergence shrinker (bd-quill-duel-shrinker-2j21) ====

    struct ScriptedShrinkEngine {
        descriptor: EngineDescriptor,
        skew_on: Option<String>,
        documents: Vec<GeneratedDocument>,
    }

    impl ScriptedShrinkEngine {
        fn honest(family: EngineFamily, label: &str) -> Self {
            Self {
                descriptor: EngineDescriptor {
                    family,
                    implementation: label.to_owned(),
                    crate_version: env!("CARGO_PKG_VERSION").to_owned(),
                    source_revision: "test".to_owned(),
                    source_dirty: false,
                    config_hash: "test-config".to_owned(),
                },
                skew_on: None,
                documents: Vec::new(),
            }
        }

        fn skewed(family: EngineFamily, label: &str, trigger_doc: &str) -> Self {
            let mut engine = Self::honest(family, label);
            engine.skew_on = Some(trigger_doc.to_owned());
            engine
        }
    }

    impl DifferentialCampaignEngine for ScriptedShrinkEngine {
        fn descriptor(&self) -> EngineDescriptor {
            self.descriptor.clone()
        }

        fn semantic_contract(&self) -> SemanticContract {
            SemanticContract::scalar_g1a()
        }

        fn begin_corpus<'a>(
            &'a mut self,
            _cx: &'a Cx,
            _manifest: &'a CorpusManifest,
            _semantic_contract: &'a SemanticContract,
        ) -> CampaignFuture<'a, ()> {
            Box::pin(async move {
                self.documents.clear();
                Ok(())
            })
        }

        fn index_batch<'a>(
            &'a mut self,
            _cx: &'a Cx,
            documents: &'a [GeneratedDocument],
        ) -> CampaignFuture<'a, ()> {
            Box::pin(async move {
                self.documents.extend_from_slice(documents);
                Ok(())
            })
        }

        fn commit_corpus<'a>(
            &'a mut self,
            _cx: &'a Cx,
            manifest: &'a CorpusManifest,
            semantic_contract: &'a SemanticContract,
        ) -> CampaignFuture<'a, EngineIndexReceipt> {
            Box::pin(async move {
                Ok(EngineIndexReceipt {
                    corpus_manifest_hash: manifest.manifest_hash()?,
                    document_count: u64::try_from(self.documents.len()).unwrap_or(u64::MAX),
                    total_content_bytes: manifest.total_content_bytes,
                    semantic_contract: semantic_contract.clone(),
                })
            })
        }

        fn observe_generated<'a>(
            &'a mut self,
            _cx: &'a Cx,
            query: &'a GeneratedQueryCase,
            evidence_case: &'a DifferentialCase,
        ) -> CampaignFuture<'a, EngineObservation> {
            Box::pin(async move {
                let mut hits: Vec<RankedHit> = self
                    .documents
                    .iter()
                    .enumerate()
                    .map(|(ordinal, document)| {
                        // Deterministic content-driven score, well separated.
                        let mut hasher = Sha256::new();
                        hasher.update(document.id.as_bytes());
                        hasher.update(document.content.as_bytes());
                        let digest = hasher.finalize();
                        let score = 0.5_f32 + (f32::from(digest[0]) / 255.0) * 10.0;
                        RankedHit {
                            doc_id: document.id.clone(),
                            score_bits: score.to_bits(),
                            native_tie_key: NativeTieKey::QuillDocId {
                                doc_id: u32::try_from(ordinal).unwrap_or(u32::MAX),
                            },
                        }
                    })
                    .collect();
                hits.sort_by(|left, right| {
                    right
                        .score_bits
                        .cmp(&left.score_bits)
                        .then_with(|| left.doc_id.cmp(&right.doc_id))
                });
                // The skew: when the trigger doc is in the corpus AND the
                // query names the trigger token, zero its score — a rank
                // flip beyond tie groups (RankMismatch) with order intact.
                if let Some(trigger) = &self.skew_on {
                    let query_names_trigger = query.query.contains("zzz");
                    if query_names_trigger {
                        for hit in &mut hits {
                            if &hit.doc_id == trigger {
                                hit.score_bits = 0.0_f32.to_bits();
                            }
                        }
                        hits.sort_by(|left, right| {
                            right
                                .score_bits
                                .cmp(&left.score_bits)
                                .then_with(|| left.doc_id.cmp(&right.doc_id))
                        });
                    }
                }
                hits.truncate(usize::try_from(evidence_case.limit).unwrap_or(usize::MAX));
                let count = u64::try_from(hits.len()).unwrap_or(u64::MAX);
                let doc_count = u64::try_from(self.documents.len()).unwrap_or(u64::MAX);
                Ok(EngineObservation {
                    hits,
                    cutoff_tie_group: Vec::new(),
                    cutoff_tie_complete: false,
                    offset_tie_group: Vec::new(),
                    offset_tie_complete: false,
                    snippets: BTreeMap::new(),
                    match_count: CountState::Value(count),
                    doc_count,
                    ast_differences: Vec::new(),
                })
            })
        }

        fn abort_corpus(&mut self) {
            self.documents.clear();
        }
    }

    fn shrink_fixture_documents(count: usize) -> Vec<GeneratedDocument> {
        (0..count)
            .map(|index| GeneratedDocument {
                id: format!("doc-{index:03}"),
                title: None,
                content: format!("alpha beta document number {index} searchable content"),
                created_at_ms: 1_700_000_000 + i64::try_from(index).unwrap_or(0),
                cass: None,
                metadata: BTreeMap::new(),
                pathology: None,
                unicode_lane: crate::generator::UnicodeLane::Ascii,
            })
            .collect()
    }

    fn shrink_query() -> GeneratedQueryCase {
        GeneratedQueryCase {
            id: "shrink-case".to_owned(),
            syntax: QuerySyntax::Default,
            query_kind: GeneratedQueryKind::Harvested {
                semantic_class: "test".to_owned(),
            },
            query: "zzz alpha beta gamma".to_owned(),
            limit: 64,
            offset: 0,
            count_requested: true,
            filters: crate::generator::GeneratedQueryFilters::default(),
            expected_divergence: None,
            source: "shrink-test".to_owned(),
        }
    }

    fn shrink_evidence_case(query_text: &str) -> DifferentialCase {
        DifferentialCase::new("shrink-case", query_text, 64)
    }

    fn make_honest() -> ShrinkEngineFactory {
        Box::new(|| {
            let engine: Box<dyn DifferentialCampaignEngine> =
                Box::new(ScriptedShrinkEngine::honest(EngineFamily::Quill, "honest"));
            Ok(engine)
        })
    }

    fn make_skewed(trigger: &str) -> ShrinkEngineFactory {
        let trigger = trigger.to_owned();
        Box::new(move || {
            let engine: Box<dyn DifferentialCampaignEngine> = Box::new(
                ScriptedShrinkEngine::skewed(EngineFamily::Tantivy, "skewed", &trigger),
            );
            Ok(engine)
        })
    }

    #[test]
    fn shrink_minimizes_corpus_and_query_and_preserves_original_context() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let documents = shrink_fixture_documents(12);
            let trigger_id = "doc-007".to_owned();
            let query = shrink_query();
            let request = ShrinkRequest {
                corpus_manifest_hash: "full-corpus-hash".to_owned(),
                documents,
                query,
                evidence_case: shrink_evidence_case("zzz alpha beta gamma"),
                divergence_class: DivergenceClass::RankMismatch,
            };
            let driver = ShrinkDriver::new(
                ComparatorConfig::default(),
                SemanticContract::scalar_g1a(),
                DEFAULT_SHRINK_FUEL,
            );
            let reproduction = driver
                .shrink(
                    &cx,
                    &request,
                    &mut make_honest(),
                    &mut make_skewed(&trigger_id),
                )
                .await
                .expect("shrink completes");

            // Corpus reduced to a minimal set that still contains the trigger.
            assert!(
                reproduction
                    .minimized_documents
                    .iter()
                    .any(|document| document.id == trigger_id),
                "trigger survives: {:?}",
                reproduction
                    .minimized_documents
                    .iter()
                    .map(|document| &document.id)
                    .collect::<Vec<_>>()
            );
            assert!(
                reproduction.minimized_documents.len() <= 4,
                "ddmin converges near the trigger: {}",
                reproduction.minimized_documents.len()
            );
            // Query minimized to the trigger token alone.
            assert_eq!(reproduction.minimized_query_text, "zzz");
            // Original context preserved (anti-over-minimization amendment).
            assert_eq!(reproduction.original_document_count, 12);
            assert_eq!(reproduction.original_query_text, "zzz alpha beta gamma");
            assert_eq!(
                reproduction.original_corpus_manifest_hash,
                "full-corpus-hash"
            );
            assert_eq!(reproduction.divergence_class, DivergenceClass::RankMismatch);
            assert!(reproduction.candidates_evaluated > 0);
            assert!(reproduction.reduction_steps > 0);
            // Auto-triage: identical sets with a rank flip => BM25 arithmetic.
            assert_eq!(reproduction.triage.class, DivergenceClass::RankMismatch);
        });
    }

    #[test]
    fn shrink_fuel_exhaustion_is_a_typed_error() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let request = ShrinkRequest {
                corpus_manifest_hash: "h".to_owned(),
                documents: shrink_fixture_documents(12),
                query: shrink_query(),
                evidence_case: shrink_evidence_case("zzz alpha beta gamma"),
                divergence_class: DivergenceClass::RankMismatch,
            };
            let driver = ShrinkDriver::new(
                ComparatorConfig::default(),
                SemanticContract::scalar_g1a(),
                1,
            );
            let error = driver
                .shrink(
                    &cx,
                    &request,
                    &mut make_honest(),
                    &mut make_skewed("doc-007"),
                )
                .await
                .expect_err("one evaluation cannot finish a shrink");
            assert!(matches!(error, ShrinkError::FuelExhausted { .. }));
        });
    }

    #[test]
    fn shrink_shadow_line_parses_and_preserves_stamped_generation() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let record = ShadowDivergenceRecord {
                schema_version: 1,
                stamped_generation: 42,
                corpus_manifest_hash: "shadow-corpus".to_owned(),
                documents: shrink_fixture_documents(8),
                query: shrink_query(),
                evidence_case: shrink_evidence_case("zzz alpha beta gamma"),
                divergence_class: DivergenceClass::RankMismatch,
            };
            let line = serde_json::to_string(&record).expect("serialize record");
            let driver = ShrinkDriver::new(
                ComparatorConfig::default(),
                SemanticContract::scalar_g1a(),
                DEFAULT_SHRINK_FUEL,
            );
            let reproduction = driver
                .shrink_shadow_line(&cx, &line, &mut make_honest(), &mut make_skewed("doc-004"))
                .await
                .expect("shadow shrink completes");
            assert!(
                reproduction
                    .original_corpus_manifest_hash
                    .ends_with("#gen-42"),
                "stamped generation rides into the reproduction: {}",
                reproduction.original_corpus_manifest_hash
            );
            assert!(
                reproduction
                    .minimized_documents
                    .iter()
                    .any(|document| document.id == "doc-004")
            );

            let bad_line = "{\"schema_version\":99}";
            let error = driver
                .shrink_shadow_line(
                    &cx,
                    bad_line,
                    &mut make_honest(),
                    &mut make_skewed("doc-004"),
                )
                .await
                .expect_err("unsupported schema fails closed");
            assert!(matches!(error, ShrinkError::InvalidShadowRecord { .. }));
        });
    }

    #[test]
    fn shrink_shadow_line_roundtrips_production_schema() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let documents = shrink_fixture_documents(8)
                .into_iter()
                .map(|document| frankensearch_core::ShadowDocument {
                    id: document.id,
                    content: document.content,
                    title: document.title,
                    metadata: document.metadata,
                })
                .collect::<Vec<_>>();
            let record = frankensearch_core::ShadowDivergenceRecord {
                schema_version: frankensearch_core::SHADOW_DIVERGENCE_SCHEMA_VERSION,
                manifest_generation: 73,
                corpus_hash: frankensearch_core::compute_shadow_corpus_hash(&documents),
                documents,
                query: frankensearch_core::ShadowQuery {
                    text: "zzz alpha beta gamma".to_owned(),
                    limit: 10,
                    fusion_candidates: true,
                },
                serving_top_k: vec![frankensearch_core::ShadowRankedHit {
                    rank: 1,
                    document_id: "doc-004".to_owned(),
                    score_bits: 1.0_f32.to_bits(),
                    lexical_score_bits: Some(1.0_f32.to_bits()),
                }],
                shadow_top_k: vec![frankensearch_core::ShadowRankedHit {
                    rank: 1,
                    document_id: "doc-003".to_owned(),
                    score_bits: 1.0_f32.to_bits(),
                    lexical_score_bits: Some(1.0_f32.to_bits()),
                }],
                classification: frankensearch_core::ShadowDivergenceClass::RankMismatch,
                serve_latency_micros: 17,
                shadow_latency_micros: 31,
            };
            record.validate().expect("production schema validates");
            let line = serde_json::to_string(&record).expect("serialize production record");
            let driver = ShrinkDriver::new(
                ComparatorConfig::default(),
                SemanticContract::scalar_g1a(),
                DEFAULT_SHRINK_FUEL,
            );
            let reproduction = driver
                .shrink_shadow_line(&cx, &line, &mut make_honest(), &mut make_skewed("doc-004"))
                .await
                .expect("production shadow record shrinks");
            assert_eq!(reproduction.original_document_count, 8);
            assert_eq!(reproduction.original_query_text, "zzz alpha beta gamma");
            assert!(
                reproduction
                    .original_corpus_manifest_hash
                    .ends_with("#gen-73")
            );
            assert!(
                reproduction
                    .minimized_documents
                    .iter()
                    .any(|document| document.id == "doc-004")
            );
        });
    }

    #[test]
    fn auto_triage_maps_comparator_evidence_to_layers() {
        let base_observation = |ids: &[&str]| EngineObservation {
            hits: ids
                .iter()
                .enumerate()
                .map(|(ordinal, id)| RankedHit {
                    doc_id: (*id).to_owned(),
                    score_bits: (10.0_f32 - ordinal as f32).to_bits(),
                    native_tie_key: NativeTieKey::QuillDocId {
                        doc_id: u32::try_from(ordinal).unwrap_or(u32::MAX),
                    },
                })
                .collect(),
            cutoff_tie_group: Vec::new(),
            cutoff_tie_complete: false,
            offset_tie_group: Vec::new(),
            offset_tie_complete: false,
            snippets: BTreeMap::new(),
            match_count: CountState::Value(ids.len() as u64),
            doc_count: ids.len() as u64,
            ast_differences: Vec::new(),
        };
        let report = ComparisonReport {
            status: ComparisonStatus::Failed,
            rank_class: RankClass::RankMismatch,
            score_epsilon_reason: None,
            divergences: Vec::new(),
            first_divergence: None,
            subject: base_observation(&["a", "b", "c"]),
            oracle: base_observation(&["a", "c", "b"]),
        };
        let verdict = auto_triage(DivergenceClass::RankMismatch, &report);
        assert_eq!(verdict.suspected_layer, SuspectedLayer::FieldNormArithmetic);
        assert_eq!(verdict.confidence, TriageConfidence::Medium);

        let mut differing = report.clone();
        differing.subject = base_observation(&["a", "b"]);
        let verdict = auto_triage(DivergenceClass::RankMismatch, &differing);
        assert_eq!(verdict.suspected_layer, SuspectedLayer::Indexing);

        let verdict = auto_triage(DivergenceClass::TieOrder, &report);
        assert_eq!(verdict.suspected_layer, SuspectedLayer::TieOrder);
        assert_eq!(verdict.confidence, TriageConfidence::High);

        let verdict = auto_triage(DivergenceClass::PostingRecordSemantics, &report);
        assert_eq!(verdict.suspected_layer, SuspectedLayer::FieldNormArithmetic);
        assert_eq!(verdict.confidence, TriageConfidence::High);
        assert!(
            verdict
                .evidence
                .iter()
                .any(|line| line.contains("record option")),
            "posting-record triage must retain the fix-only root-cause hint"
        );
    }

    #[test]
    fn persist_shrunk_reproduction_writes_a_content_addressed_fixture()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let reproduction = ShrunkReproduction {
            schema_version: 1,
            divergence_class: DivergenceClass::RankMismatch,
            original_corpus_manifest_hash: "original".to_owned(),
            original_document_count: 12,
            original_query_text: "zzz alpha".to_owned(),
            original_query_id: "case".to_owned(),
            minimized_documents: shrink_fixture_documents(2),
            minimized_query_text: "zzz".to_owned(),
            triage: TriageVerdict {
                class: DivergenceClass::RankMismatch,
                suspected_layer: SuspectedLayer::FieldNormArithmetic,
                confidence: TriageConfidence::Medium,
                evidence: vec!["rows".to_owned()],
            },
            reduction_steps: 9,
            candidates_evaluated: 41,
        };
        let path = persist_shrunk_reproduction(directory.path(), &reproduction)?;
        assert!(path.exists());
        assert!(path.starts_with(directory.path().join("shrunk")));
        let roundtrip: ShrunkReproduction = serde_json::from_slice(&std::fs::read(&path)?)?;
        assert_eq!(roundtrip, reproduction);
        Ok(())
    }

    // ==== Live oracle campaign activation (bd-quill-e6-gauntlet-scale-rm3q.9) ====

    #[cfg(feature = "tantivy-oracle")]
    fn live_campaign_engines(
        provenance: &CampaignProvenance,
    ) -> (crate::engine::QuillSubject, crate::engine::TantivyOracle) {
        let config = frankensearch_quill::QuillConfig {
            deterministic_ingest: true,
            ..frankensearch_quill::QuillConfig::default()
        };
        let subject = crate::engine::QuillSubject::in_memory(
            config,
            &provenance.subject_git_revision,
            provenance.subject_source_dirty,
        )
        .expect("fresh scalar Quill subject");
        let oracle = crate::engine::TantivyOracle::in_memory_scalar_g1a(
            &provenance.oracle_git_revision,
            provenance.oracle_source_dirty,
        )
        .expect("fresh scalar G1a Tantivy oracle");
        (subject, oracle)
    }

    #[cfg(feature = "tantivy-oracle")]
    fn live_cass_campaign_engines(
        provenance: &CampaignProvenance,
    ) -> (
        crate::engine::CassQuillSubject,
        crate::engine::CassTantivyOracle,
    ) {
        let config = frankensearch_quill::QuillConfig {
            deterministic_ingest: true,
            glob_expansion_limit: 4_096,
            ..frankensearch_quill::QuillConfig::default()
        };
        let subject = crate::engine::CassQuillSubject::in_memory(
            config,
            &provenance.subject_git_revision,
            provenance.subject_source_dirty,
        )
        .expect("fresh CASS Quill subject");
        let oracle = crate::engine::CassTantivyOracle::in_memory(
            &provenance.oracle_git_revision,
            provenance.oracle_source_dirty,
        )
        .expect("fresh CASS Tantivy oracle");
        (subject, oracle)
    }

    #[cfg(feature = "tantivy-oracle")]
    async fn run_live_default_profile_campaign(
        cx: &Cx,
        root: &std::path::Path,
        run_id: &str,
    ) -> Result<CampaignReport, GauntletError> {
        let fixture = make_scalar_g1a_regression_fixture();
        run_live_default_profile_fixture(cx, root, run_id, &fixture).await
    }

    #[cfg(feature = "tantivy-oracle")]
    async fn run_live_default_profile_fixture(
        cx: &Cx,
        root: &std::path::Path,
        run_id: &str,
        fixture: &Fixture,
    ) -> Result<CampaignReport, GauntletError> {
        let selection = CampaignSelection::DefaultSyntax;
        let semantic_contract = SemanticContract::scalar_g1a();
        let provenance = CampaignProvenance::collect(
            &fixture.corpus_manifest,
            &fixture.query_suite.manifest,
            &selection,
            &semantic_contract,
        )
        .expect("collect provenance");
        let (mut subject, mut oracle) = live_campaign_engines(&provenance);
        let campaign = DifferentialCampaignRunner::new(
            ArtifactStore::new(root),
            semantic_contract,
            CampaignConfig {
                selection,
                contract_mode: CampaignContractMode::CoreLexicalV3,
                require_provenance: true,
                index_batch_size: 5,
                snippet_max_chars: None,
                ..CampaignConfig::default()
            },
            DivergenceRegistry::default(),
        )
        .expect("live campaign runner")
        .with_provenance(provenance);
        campaign
            .run(
                cx,
                run_id,
                &mut subject,
                &mut oracle,
                &fixture.documents,
                &fixture.corpus_manifest,
                &fixture.query_suite,
            )
            .await
    }

    #[cfg(feature = "tantivy-oracle")]
    async fn run_live_cass_profile_fixture(
        cx: &Cx,
        root: &std::path::Path,
        run_id: &str,
        fixture: &Fixture,
    ) -> Result<CampaignReport, GauntletError> {
        let selection = CampaignSelection::CassSyntax;
        let semantic_contract = SemanticContract::cass();
        let provenance = CampaignProvenance::collect(
            &fixture.corpus_manifest,
            &fixture.query_suite.manifest,
            &selection,
            &semantic_contract,
        )
        .expect("collect CASS provenance");
        let (mut subject, mut oracle) = live_cass_campaign_engines(&provenance);
        let campaign = DifferentialCampaignRunner::new(
            ArtifactStore::new(root),
            semantic_contract,
            CampaignConfig {
                selection,
                require_provenance: true,
                index_batch_size: 5,
                snippet_max_chars: None,
                ..CampaignConfig::default()
            },
            DivergenceRegistry::default(),
        )
        .expect("live CASS campaign runner")
        .with_provenance(provenance);
        campaign
            .run(
                cx,
                run_id,
                &mut subject,
                &mut oracle,
                &fixture.documents,
                &fixture.corpus_manifest,
                &fixture.query_suite,
            )
            .await
    }

    #[cfg(feature = "tantivy-oracle")]
    fn load_campaign_case_object(
        root: &std::path::Path,
        result: &CampaignCaseResult,
    ) -> ArtifactObject {
        let object_hash = result
            .artifact_hash
            .as_deref()
            .expect("successful campaign case has an immutable object");
        serde_json::from_slice(
            &std::fs::read(root.join("objects").join(format!("{object_hash}.json")))
                .expect("read immutable campaign object"),
        )
        .expect("decode immutable campaign object")
    }

    #[cfg(feature = "tantivy-oracle")]
    fn assert_persisted_core_object_mutation_matrix_fails(
        source_root: &std::path::Path,
        report: &CampaignReport,
    ) {
        fn core_comparison(object: &mut ArtifactObject) -> &mut LexicalContractComparison {
            let ArtifactLexicalContractEvidence::CoreLexicalV3 { comparison } =
                &mut object.lexical_contract
            else {
                unreachable!("mutation target must retain Core Lexical V3 evidence")
            };
            comparison
        }

        fn subject_bundle(object: &mut ArtifactObject) -> &mut LexicalContractBundle {
            &mut core_comparison(object).subject
        }

        fn subject_full_context(object: &mut ArtifactObject) -> &mut LexicalObservationContext {
            &mut subject_bundle(object).full_search.context
        }

        fn success_fields(
            observation: &mut LexicalObservation,
        ) -> (
            &mut Vec<LexicalHitObservation>,
            &mut u64,
            &mut LexicalEmptyShape,
            &mut LexicalCountState,
        ) {
            let LexicalObservationOutcome::Success {
                hits,
                returned_count,
                empty_shape,
                total_count,
            } = &mut observation.outcome
            else {
                unreachable!("selected mutation target must be a successful observation")
            };
            (hits, returned_count, empty_shape, total_count)
        }

        fn first_subject_full_hit(object: &mut ArtifactObject) -> &mut LexicalHitObservation {
            success_fields(&mut subject_bundle(object).full_search)
                .0
                .first_mut()
                .expect("selected mutation target must retain a full-search hit")
        }

        fn attempted_hydration(
            execution: &mut LexicalHydrationExecution,
        ) -> (
            &mut LexicalObservation,
            &mut LexicalObservation,
            &mut LexicalHydrationResult,
        ) {
            let LexicalHydrationExecution::Attempted {
                input,
                post_state,
                result,
            } = execution
            else {
                unreachable!("selected mutation target must retain attempted hydration")
            };
            (input, post_state, result)
        }

        fn flip_sha256_nibble(value: &mut String) {
            assert_eq!(value.len(), 64, "mutation target must be a SHA-256");
            let replacement = if value.starts_with('0') { "1" } else { "0" };
            value.replace_range(0..1, replacement);
        }

        fn mutate_sensitive_digest(value: &mut SensitiveValueObservation) {
            let sha256 = match value {
                SensitiveValueObservation::PresentEmpty { sha256, .. }
                | SensitiveValueObservation::Present { sha256, .. } => sha256,
                SensitiveValueObservation::NotExposed | SensitiveValueObservation::Absent => {
                    unreachable!("mutation target must retain a digest-bearing sensitive value")
                }
            };
            flip_sha256_nibble(sha256);
        }

        fn mutate_sensitive_length(value: &mut SensitiveValueObservation) {
            let byte_len = match value {
                SensitiveValueObservation::PresentEmpty { byte_len, .. }
                | SensitiveValueObservation::Present { byte_len, .. } => byte_len,
                SensitiveValueObservation::NotExposed | SensitiveValueObservation::Absent => {
                    unreachable!("mutation target must retain a length-bearing sensitive value")
                }
            };
            *byte_len = byte_len.saturating_add(1);
        }

        fn digest_bearing_metadata_hit(
            observation: &mut LexicalObservation,
        ) -> &mut LexicalHitObservation {
            success_fields(observation)
                .0
                .iter_mut()
                .find(|hit| {
                    matches!(
                        &hit.metadata,
                        SensitiveValueObservation::PresentEmpty { .. }
                            | SensitiveValueObservation::Present { .. }
                    )
                })
                .expect("selected mutation target must retain metadata digest evidence")
        }

        fn digest_bearing_explanation_hit(
            observation: &mut LexicalObservation,
        ) -> &mut LexicalHitObservation {
            success_fields(observation)
                .0
                .iter_mut()
                .find(|hit| {
                    matches!(
                        &hit.explanation,
                        SensitiveValueObservation::PresentEmpty { .. }
                            | SensitiveValueObservation::Present { .. }
                    )
                })
                .expect("selected mutation target must retain explanation digest evidence")
        }

        fn query_error_observation() -> LexicalErrorObservation {
            let error = frankensearch_core::SearchError::QueryParseError {
                query: "redacted by persisted mutation test".to_owned(),
                detail: "typed failure replacement".to_owned(),
            };
            crate::comparator::observe_lexical_search_error(&error)
                .expect("observe typed mutation error")
        }

        fn replace_with_query_error(outcome: &mut LexicalObservationOutcome) {
            *outcome = LexicalObservationOutcome::Error(query_error_observation());
        }

        fn mutate_one_full_score_bit(object: &mut ArtifactObject) {
            let hits = success_fields(&mut subject_bundle(object).full_search).0;
            for index in 0..hits.len() {
                let candidate_bits = hits[index].normalized_score_bits ^ 1;
                let candidate = f32::from_bits(candidate_bits);
                let previous_allows = index == 0
                    || f32::from_bits(hits[index - 1].normalized_score_bits) >= candidate;
                let next_allows = index + 1 == hits.len()
                    || candidate >= f32::from_bits(hits[index + 1].normalized_score_bits);
                if candidate.is_finite() && previous_allows && next_allows {
                    hits[index].normalized_score_bits = candidate_bits;
                    hits[index].raw_lexical_score_bits = Some(candidate_bits);
                    return;
                }
            }
            unreachable!("selected mutation target must admit a finite one-bit score mutation");
        }

        fn swap_tied_full_hits(object: &mut ArtifactObject) {
            let hits = success_fields(&mut subject_bundle(object).full_search).0;
            let index = hits
                .windows(2)
                .position(|pair| pair[0].normalized_score_bits == pair[1].normalized_score_bits)
                .expect("selected mutation target must retain an adjacent score tie");
            hits.swap(index, index + 1);
            hits[index].rank = u64::try_from(index).expect("persisted hit ordinal must fit in u64");
            hits[index + 1].rank =
                u64::try_from(index + 1).expect("persisted hit ordinal must fit in u64");
        }

        let scratch = tempfile::tempdir().expect("isolated mutation store");
        let scratch_root = scratch.path();
        let source_campaign = source_root.join("campaigns").join(&report.run_id);
        let scratch_campaign = scratch_root.join("campaigns").join(&report.run_id);
        let scratch_cases = scratch_campaign.join("cases");
        let scratch_objects = scratch_root.join("objects");
        std::fs::create_dir_all(&scratch_cases).expect("create scratch campaign cases");
        std::fs::create_dir_all(&scratch_objects).expect("create scratch object store");
        for name in ["reservation.json", "report.json"] {
            std::fs::copy(source_campaign.join(name), scratch_campaign.join(name))
                .expect("copy campaign control file");
        }
        for (ordinal, result) in report.cases.iter().enumerate() {
            let Some(hash) = result.artifact_hash.as_deref() else {
                continue;
            };
            std::fs::copy(
                source_campaign
                    .join("cases")
                    .join(format!("q{ordinal:06}.json")),
                scratch_cases.join(format!("q{ordinal:06}.json")),
            )
            .expect("copy case run manifest");
            let name = format!("{hash}.json");
            std::fs::copy(
                source_root.join("objects").join(&name),
                scratch_objects.join(name),
            )
            .expect("copy immutable object");
        }
        let store = ArtifactStore::new(scratch_root);
        assert_eq!(
            store
                .load_verified_campaign(&report.run_id)
                .expect("copied campaign starts verified"),
            *report
        );

        let (target_ordinal, target_hash, target_object) = report
            .cases
            .iter()
            .enumerate()
            .find_map(|(ordinal, result)| {
                let hash = result.artifact_hash.as_ref()?;
                let object = load_campaign_case_object(source_root, result);
                let ArtifactLexicalContractEvidence::CoreLexicalV3 { comparison } =
                    &object.lexical_contract
                else {
                    return None;
                };
                let crate::comparator::LexicalObservationOutcome::Success { hits, .. } =
                    &comparison.subject.full_search.outcome
                else {
                    return None;
                };
                let has_adjacent_score_tie = hits
                    .windows(2)
                    .any(|pair| pair[0].normalized_score_bits == pair[1].normalized_score_bits);
                let has_metadata_digest = hits.iter().any(|hit| {
                    matches!(
                        &hit.metadata,
                        SensitiveValueObservation::PresentEmpty { .. }
                            | SensitiveValueObservation::Present { .. }
                    )
                });
                let all_hydrations_attempted = [
                    &comparison.subject.all_lexical_winners_hydration,
                    &comparison.subject.strict_hybrid_winners_hydration,
                    &comparison.subject.semantic_only_hydration,
                    &comparison.subject.mixed_winners_hydration,
                ]
                .into_iter()
                .all(|transition| {
                    matches!(
                        &transition.execution,
                        LexicalHydrationExecution::Attempted { .. }
                    )
                });
                let strict_has_explanation_digest = match &comparison
                    .subject
                    .strict_hybrid_winners_hydration
                    .execution
                {
                    LexicalHydrationExecution::Attempted { input, .. } => match &input.outcome {
                        LexicalObservationOutcome::Success { hits, .. } => hits.iter().any(|hit| {
                            matches!(
                                &hit.explanation,
                                SensitiveValueObservation::PresentEmpty { .. }
                                    | SensitiveValueObservation::Present { .. }
                            )
                        }),
                        LexicalObservationOutcome::Error(_) => false,
                    },
                    LexicalHydrationExecution::NotRun { .. } => false,
                };
                if !has_adjacent_score_tie
                    || !has_metadata_digest
                    || !all_hydrations_attempted
                    || !strict_has_explanation_digest
                {
                    return None;
                }
                Some((ordinal, hash.clone(), object))
            })
            .expect(
                "live Core V3 campaign has a tied, metadata-bearing, fully hydrated persisted case",
            );
        let target_path = scratch_objects.join(format!("{target_hash}.json"));
        let original_bytes = std::fs::read(&target_path).expect("read scratch target object");
        let target_case_path = scratch_cases.join(format!("q{target_ordinal:06}.json"));
        let original_case_bytes =
            std::fs::read(&target_case_path).expect("read scratch target run manifest");
        let reservation_path = scratch_campaign.join("reservation.json");
        let original_reservation_bytes =
            std::fs::read(&reservation_path).expect("read scratch campaign reservation");
        let report_path = scratch_campaign.join("report.json");
        let original_report_bytes =
            std::fs::read(&report_path).expect("read scratch campaign report");

        let assert_stale_address_rejected = |label: &str, bytes: &[u8]| {
            assert_ne!(bytes, original_bytes, "{label} must alter persisted bytes");
            std::fs::write(&target_path, bytes).expect("write scratch mutation");
            assert!(
                store.load_verified_campaign(&report.run_id).is_err(),
                "verified campaign reload accepted persisted mutation {label}"
            );
            std::fs::write(&target_path, &original_bytes).expect("restore scratch object");
            store
                .load_verified_campaign(&report.run_id)
                .unwrap_or_else(|error| panic!("restored campaign failed after {label}: {error}"));
        };

        let mut raw_content_tamper = original_bytes.clone();
        raw_content_tamper.push(b' ');
        let mut existing_mutation_count = 0;
        assert_stale_address_rejected("raw_content_byte", &raw_content_tamper);
        existing_mutation_count += 1;

        let original_value: serde_json::Value =
            serde_json::from_slice(&original_bytes).expect("object JSON");
        for lane in [
            "full_search",
            "fusion_candidates",
            "all_lexical_winners_hydration",
            "strict_hybrid_winners_hydration",
            "semantic_only_hydration",
            "mixed_winners_hydration",
        ] {
            let mut missing_lane = original_value.clone();
            assert!(
                missing_lane["lexical_contract"]["comparison"]["subject"]
                    .as_object_mut()
                    .expect("subject lexical bundle")
                    .remove(lane)
                    .is_some(),
                "persisted fixture must contain lane {lane}"
            );
            assert_stale_address_rejected(
                &format!("missing_lane_{lane}"),
                &serde_json::to_vec(&missing_lane).expect("encode missing-lane mutation"),
            );
            existing_mutation_count += 1;
        }

        type TypedMutation = fn(&mut ArtifactObject);
        let typed_mutations: &[(&str, TypedMutation)] = &[
            ("object_canonicalization_schema", |object| {
                object.canonicalization_version = object.canonicalization_version.saturating_add(1);
            }),
            ("comparison_schema", |object| {
                core_comparison(object).schema_version.push_str("-tampered");
            }),
            ("bundle_schema", |object| {
                subject_bundle(object).schema_version.push_str("-tampered");
            }),
            ("bundle_engine_role", |object| {
                subject_bundle(object).engine_role = LexicalEngineRole::Oracle;
            }),
            ("bundle_snapshot_sha256", |object| {
                flip_sha256_nibble(&mut subject_bundle(object).snapshot_sha256);
            }),
            ("bundle_capability", |object| {
                let bundle = subject_bundle(object);
                bundle.fusion_metadata_deferred = !bundle.fusion_metadata_deferred;
            }),
            ("context_schema", |object| {
                subject_full_context(object)
                    .schema_version
                    .push_str("-tampered");
            }),
            ("context_boundary", |object| {
                subject_full_context(object).boundary =
                    crate::comparator::LexicalBoundary::FusionCandidates;
            }),
            ("context_backend_engine", |object| {
                subject_full_context(object)
                    .backend
                    .engine
                    .push_str("-tampered");
            }),
            ("context_backend_revision", |object| {
                subject_full_context(object)
                    .backend
                    .revision
                    .push_str("-tampered");
            }),
            ("context_backend_index_identity", |object| {
                flip_sha256_nibble(&mut subject_full_context(object).backend.index_identity);
            }),
            ("context_corpus_sha256", |object| {
                flip_sha256_nibble(&mut subject_full_context(object).corpus_sha256);
            }),
            ("context_query_contract_sha256", |object| {
                flip_sha256_nibble(&mut subject_full_context(object).query_contract_sha256);
            }),
            ("context_query_sha256", |object| {
                flip_sha256_nibble(&mut subject_full_context(object).query_sha256);
            }),
            ("context_query_bytes", |object| {
                let context = subject_full_context(object);
                context.query_bytes = context.query_bytes.saturating_add(1);
            }),
            ("context_normalized_query", |object| {
                subject_full_context(object).normalized_query = LexicalNormalizedQuery::Value {
                    transform_id: "persisted-mutation-transform".to_owned(),
                    sha256: "0".repeat(64),
                    byte_len: 0,
                };
            }),
            ("context_query_class", |object| {
                let context = subject_full_context(object);
                context.query_class = match context.query_class {
                    LexicalQueryClass::Empty => LexicalQueryClass::Identifier,
                    LexicalQueryClass::Identifier => LexicalQueryClass::ShortKeyword,
                    LexicalQueryClass::ShortKeyword => LexicalQueryClass::NaturalLanguage,
                    LexicalQueryClass::NaturalLanguage => LexicalQueryClass::ShortKeyword,
                };
            }),
            ("context_seed", |object| {
                subject_full_context(object).seed ^= 1;
            }),
            ("context_limit", |object| {
                let context = subject_full_context(object);
                context.limit = context.limit.saturating_add(1);
            }),
            ("context_metadata_exposure", |object| {
                subject_full_context(object).exposure.metadata = LexicalFieldExposure::NotExposed;
            }),
            ("context_explanation_exposure", |object| {
                subject_full_context(object).exposure.explanation =
                    LexicalFieldExposure::NotExposed;
            }),
            ("context_count_exposure", |object| {
                subject_full_context(object).exposure.total_count =
                    LexicalCountExposure::NotRequested;
            }),
            ("context_snippet_exposure", |object| {
                subject_full_context(object).exposure.snippet = LexicalFieldExposure::Exposed;
            }),
            ("context_highlight_exposure", |object| {
                subject_full_context(object).exposure.highlight_spans =
                    LexicalFieldExposure::Exposed;
            }),
            ("hit_rank", |object| {
                let hit = first_subject_full_hit(object);
                hit.rank = hit.rank.saturating_add(1);
            }),
            ("hit_order", swap_tied_full_hits),
            ("hit_doc_id", |object| {
                first_subject_full_hit(object).doc_id.push_str("-tampered");
            }),
            ("hit_one_bit_score", mutate_one_full_score_bit),
            ("hit_raw_lexical_component", |object| {
                let bundle = subject_bundle(object);
                let (input, _, _) =
                    attempted_hydration(&mut bundle.all_lexical_winners_hydration.execution);
                let hit = success_fields(input)
                    .0
                    .first_mut()
                    .expect("all-winner hydration input must retain a hit");
                hit.raw_lexical_score_bits = hit.raw_lexical_score_bits.map(|bits| bits ^ 1);
            }),
            ("hit_source", |object| {
                first_subject_full_hit(object).source = LexicalScoreSource::Hybrid;
            }),
            ("hit_index_component", |object| {
                let bundle = subject_bundle(object);
                let (input, _, _) =
                    attempted_hydration(&mut bundle.strict_hybrid_winners_hydration.execution);
                let hit = success_fields(input)
                    .0
                    .first_mut()
                    .expect("strict hydration input must retain a hit");
                hit.index = hit.index.map(|index| index.saturating_add(1));
            }),
            ("hit_fast_component", |object| {
                let bundle = subject_bundle(object);
                let (input, _, _) =
                    attempted_hydration(&mut bundle.strict_hybrid_winners_hydration.execution);
                let hit = success_fields(input)
                    .0
                    .first_mut()
                    .expect("strict hydration input must retain a hit");
                hit.fast_score_bits = hit.fast_score_bits.map(|bits| bits ^ 1);
            }),
            ("hit_quality_component", |object| {
                first_subject_full_hit(object).quality_score_bits = Some(0.125_f32.to_bits());
            }),
            ("hit_rerank_component", |object| {
                first_subject_full_hit(object).rerank_score_bits = Some(0.25_f32.to_bits());
            }),
            ("hit_metadata_presence", |object| {
                digest_bearing_metadata_hit(&mut subject_bundle(object).full_search).metadata =
                    SensitiveValueObservation::Absent;
            }),
            ("hit_metadata_digest", |object| {
                mutate_sensitive_digest(
                    &mut digest_bearing_metadata_hit(&mut subject_bundle(object).full_search)
                        .metadata,
                );
            }),
            ("hit_metadata_length", |object| {
                mutate_sensitive_length(
                    &mut digest_bearing_metadata_hit(&mut subject_bundle(object).full_search)
                        .metadata,
                );
            }),
            ("hit_explanation_presence", |object| {
                let explanation = &mut first_subject_full_hit(object).explanation;
                *explanation = if matches!(explanation, SensitiveValueObservation::Absent) {
                    SensitiveValueObservation::Present {
                        sha256: "0".repeat(64),
                        byte_len: 1,
                    }
                } else {
                    SensitiveValueObservation::Absent
                };
            }),
            ("hit_explanation_digest", |object| {
                let bundle = subject_bundle(object);
                let (input, _, _) =
                    attempted_hydration(&mut bundle.strict_hybrid_winners_hydration.execution);
                mutate_sensitive_digest(&mut digest_bearing_explanation_hit(input).explanation);
            }),
            ("hit_snippet_state", |object| {
                first_subject_full_hit(object).snippet = SensitiveValueObservation::Absent;
            }),
            ("hit_highlight_state", |object| {
                first_subject_full_hit(object).highlight_spans = LexicalObserved::Absent;
            }),
            ("success_returned_count", |object| {
                let (_, returned_count, _, _) =
                    success_fields(&mut subject_bundle(object).full_search);
                *returned_count = returned_count.saturating_add(1);
            }),
            ("success_empty_shape", |object| {
                let (_, _, empty_shape, _) =
                    success_fields(&mut subject_bundle(object).full_search);
                *empty_shape = LexicalEmptyShape::Empty;
            }),
            ("success_count_state", |object| {
                let (_, _, _, total_count) =
                    success_fields(&mut subject_bundle(object).full_search);
                *total_count = LexicalCountState::NotRequested;
            }),
            ("error_outcome_kind", |object| {
                replace_with_query_error(&mut subject_bundle(object).full_search.outcome);
            }),
            ("error_class", |object| {
                let mut error = query_error_observation();
                error.class = LexicalErrorClass::Index;
                subject_bundle(object).full_search.outcome =
                    LexicalObservationOutcome::Error(error);
            }),
            ("error_code", |object| {
                let mut error = query_error_observation();
                error.code.push_str("-tampered");
                subject_bundle(object).full_search.outcome =
                    LexicalObservationOutcome::Error(error);
            }),
            ("error_contract_payload", |object| {
                let mut error = query_error_observation();
                mutate_sensitive_digest(&mut error.contract_payload);
                subject_bundle(object).full_search.outcome =
                    LexicalObservationOutcome::Error(error);
            }),
            ("hydration_all_selection", |object| {
                subject_bundle(object)
                    .all_lexical_winners_hydration
                    .selection = LexicalHydrationSelection::SemanticOnlyControl {
                    control_id: u32::MAX,
                };
            }),
            ("hydration_strict_candidate_rank", |object| {
                let selection = &mut subject_bundle(object)
                    .strict_hybrid_winners_hydration
                    .selection;
                let LexicalHydrationSelection::StrictHybridWinnerSubset { candidate_ranks } =
                    selection
                else {
                    unreachable!("strict hydration must retain strict-rank selection")
                };
                candidate_ranks[0] = u64::MAX;
            }),
            ("hydration_semantic_control_id", |object| {
                let selection = &mut subject_bundle(object).semantic_only_hydration.selection;
                let LexicalHydrationSelection::SemanticOnlyControl { control_id } = selection
                else {
                    unreachable!("semantic hydration must retain its control selection")
                };
                *control_id ^= 1;
            }),
            ("hydration_mixed_lexical_origin_rank", |object| {
                let selection = &mut subject_bundle(object).mixed_winners_hydration.selection;
                let LexicalHydrationSelection::MixedFinalWinners { origins } = selection else {
                    unreachable!("mixed hydration must retain mixed origins")
                };
                let rank = origins
                    .iter_mut()
                    .find_map(|origin| match origin {
                        LexicalWinnerOrigin::Lexical { candidate_rank, .. } => Some(candidate_rank),
                        LexicalWinnerOrigin::NonLexicalControl { .. } => None,
                    })
                    .expect("mixed hydration must retain a lexical origin");
                *rank = rank.saturating_add(1);
            }),
            ("hydration_mixed_lexical_projection", |object| {
                let selection = &mut subject_bundle(object).mixed_winners_hydration.selection;
                let LexicalHydrationSelection::MixedFinalWinners { origins } = selection else {
                    unreachable!("mixed hydration must retain mixed origins")
                };
                let projection = origins
                    .iter_mut()
                    .find_map(|origin| match origin {
                        LexicalWinnerOrigin::Lexical { projection, .. } => Some(projection),
                        LexicalWinnerOrigin::NonLexicalControl { .. } => None,
                    })
                    .expect("mixed hydration must retain a lexical origin");
                *projection = match projection {
                    LexicalWinnerProjection::LexicalOnly => LexicalWinnerProjection::HybridFast,
                    LexicalWinnerProjection::HybridFast => LexicalWinnerProjection::LexicalOnly,
                };
            }),
            ("hydration_mixed_nonlexical_control_id", |object| {
                let selection = &mut subject_bundle(object).mixed_winners_hydration.selection;
                let LexicalHydrationSelection::MixedFinalWinners { origins } = selection else {
                    unreachable!("mixed hydration must retain mixed origins")
                };
                let control_id = origins
                    .iter_mut()
                    .find_map(|origin| match origin {
                        LexicalWinnerOrigin::NonLexicalControl { control_id, .. } => {
                            Some(control_id)
                        }
                        LexicalWinnerOrigin::Lexical { .. } => None,
                    })
                    .expect("mixed hydration must retain a non-lexical origin");
                *control_id ^= 1;
            }),
            ("hydration_mixed_nonlexical_kind", |object| {
                let selection = &mut subject_bundle(object).mixed_winners_hydration.selection;
                let LexicalHydrationSelection::MixedFinalWinners { origins } = selection else {
                    unreachable!("mixed hydration must retain mixed origins")
                };
                let kind = origins
                    .iter_mut()
                    .find_map(|origin| match origin {
                        LexicalWinnerOrigin::NonLexicalControl { kind, .. } => Some(kind),
                        LexicalWinnerOrigin::Lexical { .. } => None,
                    })
                    .expect("mixed hydration must retain a non-lexical origin");
                *kind = match kind {
                    LexicalNonLexicalControlKind::SemanticFast => {
                        LexicalNonLexicalControlKind::GraphOnlyHybrid
                    }
                    LexicalNonLexicalControlKind::GraphOnlyHybrid => {
                        LexicalNonLexicalControlKind::SemanticFast
                    }
                };
            }),
            ("hydration_mixed_origin_order", |object| {
                let selection = &mut subject_bundle(object).mixed_winners_hydration.selection;
                let LexicalHydrationSelection::MixedFinalWinners { origins } = selection else {
                    unreachable!("mixed hydration must retain mixed origins")
                };
                origins.swap(0, 1);
            }),
            ("hydration_input", |object| {
                let bundle = subject_bundle(object);
                let (input, _, _) =
                    attempted_hydration(&mut bundle.all_lexical_winners_hydration.execution);
                success_fields(input).0[0].doc_id.push_str("-tampered");
            }),
            ("hydration_post_state", |object| {
                let bundle = subject_bundle(object);
                let (_, post_state, _) =
                    attempted_hydration(&mut bundle.all_lexical_winners_hydration.execution);
                success_fields(post_state).0[0].doc_id.push_str("-tampered");
            }),
            ("hydration_result", |object| {
                let bundle = subject_bundle(object);
                let (_, _, result) =
                    attempted_hydration(&mut bundle.all_lexical_winners_hydration.execution);
                *result = LexicalHydrationResult::Error(query_error_observation());
            }),
            ("derived", |object| {
                core_comparison(object).status = LexicalComparisonStatus::Mismatch;
            }),
        ];
        assert_eq!(
            typed_mutations.len(),
            60,
            "Core V3 must keep one typed mutation for every admitted comparison field family"
        );
        for &(label, mutate) in typed_mutations {
            let mut mutated = target_object.clone();
            mutate(&mut mutated);
            let object_bytes = mutated
                .canonical_bytes()
                .expect("encode canonical typed object mutation");
            let object_hash = mutated
                .object_hash()
                .expect("address canonical typed object mutation");
            let object_path = scratch_objects.join(format!("{object_hash}.json"));
            std::fs::write(&object_path, &object_bytes)
                .expect("write coherently addressed typed object mutation");

            let mut run_manifest: crate::artifact::RunManifest =
                serde_json::from_slice(&original_case_bytes)
                    .expect("decode scratch target run manifest");
            run_manifest.object_hash.clone_from(&object_hash);
            std::fs::write(
                &target_case_path,
                serde_json::to_vec(&run_manifest).expect("encode updated target run manifest"),
            )
            .expect("write updated target run manifest");

            let mut mutated_report = report.clone();
            mutated_report.cases[target_ordinal].artifact_hash = Some(object_hash);
            std::fs::write(
                &report_path,
                mutated_report
                    .canonical_bytes()
                    .expect("encode coherently referenced campaign report"),
            )
            .expect("write coherently referenced campaign report");

            let error = store
                .load_verified_campaign(&report.run_id)
                .expect_err("semantic replay must reject canonical typed mutation");
            std::fs::write(&target_case_path, &original_case_bytes)
                .expect("restore target run manifest");
            std::fs::write(&report_path, &original_report_bytes).expect("restore campaign report");
            store
                .load_verified_campaign(&report.run_id)
                .unwrap_or_else(|error| panic!("restored campaign failed after {label}: {error}"));
            assert!(
                matches!(
                    &error,
                    GauntletError::InvalidCampaign { .. }
                        | GauntletError::InvalidContract { .. }
                        | GauntletError::InvalidObservation { .. }
                        | GauntletError::InvalidPreparedArtifact { .. }
                ),
                "{label} must reach semantic contract or replay validation, got: {error}"
            );
            let error_text = error.to_string();
            assert!(
                !error_text.contains("content address")
                    && !error_text.contains("run manifest")
                    && !error_text.contains("noncanonical"),
                "{label} was rejected before semantic contract or replay validation: {error_text}"
            );
        }

        type ProvenanceMutation = fn(&mut CampaignProvenance);
        let provenance_mutations: &[(&str, ProvenanceMutation)] = &[
            ("subject_git_revision", |value| {
                value.subject_git_revision = "2".repeat(40);
            }),
            ("subject_source_dirty", |value| {
                value.subject_source_dirty = !value.subject_source_dirty;
            }),
            ("oracle_git_revision", |value| {
                value.oracle_git_revision = "3".repeat(40);
            }),
            ("oracle_source_dirty", |value| {
                value.oracle_source_dirty = !value.oracle_source_dirty;
            }),
            ("cargo_lock_sha256", |value| {
                value.cargo_lock_sha256 = "0".repeat(64);
            }),
            ("rustc_version_verbose", |value| {
                value.rustc_version_verbose.push_str("mismatch");
            }),
            ("rust_toolchain_channel", |value| {
                value.rust_toolchain_channel = "nightly-1970-01-01".to_owned();
            }),
            ("unicode_version", |value| {
                value.unicode_version = "0.0.0".to_owned();
            }),
            ("unicode_normalization_version", |value| {
                value.unicode_normalization_version = "0.0.0".to_owned();
            }),
            ("unicode_normalization_table_version", |value| {
                value.unicode_normalization_table_version = "0.0.0".to_owned();
            }),
            ("query_generator_id", |value| {
                value.query_generator_id = "wrong-generator".to_owned();
            }),
            ("query_generator_schema_version", |value| {
                value.query_generator_schema_version =
                    value.query_generator_schema_version.saturating_add(1);
            }),
            ("query_seed", |value| {
                value.query_seed ^= 1;
            }),
            ("query_source_identity_sha256", |value| {
                value.query_source_identity_sha256 = "0".repeat(64);
            }),
            ("query_profile_sha256", |value| {
                value.query_profile_sha256 = "0".repeat(64);
            }),
            ("analyzer_contract_hash", |value| {
                value.analyzer_contract_hash = "0".repeat(64);
            }),
            ("schema_contract_hash", |value| {
                value.schema_contract_hash = "0".repeat(64);
            }),
            ("corpus_manifest_hash", |value| {
                value.corpus_manifest_hash = "0".repeat(64);
            }),
            ("query_manifest_hash", |value| {
                value.query_manifest_hash = "0".repeat(64);
            }),
            ("corpus_seed", |value| {
                value.corpus_seed = Some(value.corpus_seed.map_or(0, |seed| seed ^ 1));
            }),
        ];
        assert_eq!(
            provenance_mutations.len(),
            20,
            "campaign provenance mutation coverage must remain field-complete"
        );
        for &(field, mutate) in provenance_mutations {
            let mut mutated_report = report.clone();
            mutate(
                mutated_report
                    .provenance
                    .as_mut()
                    .expect("live mutation campaign must retain provenance"),
            );
            std::fs::write(
                &reservation_path,
                mutated_report
                    .reservation_bytes_unchecked()
                    .expect("encode coherently mutated campaign reservation"),
            )
            .expect("write coherently mutated campaign reservation");
            std::fs::write(
                &report_path,
                mutated_report
                    .canonical_bytes_unchecked()
                    .expect("encode coherently mutated campaign report"),
            )
            .expect("write coherently mutated campaign report");

            let error = store
                .load_verified_campaign(&report.run_id)
                .expect_err("verified campaign reload must reject mutated provenance");
            std::fs::write(&reservation_path, &original_reservation_bytes)
                .expect("restore campaign reservation");
            std::fs::write(&report_path, &original_report_bytes)
                .expect("restore campaign report after provenance mutation");
            store
                .load_verified_campaign(&report.run_id)
                .unwrap_or_else(|error| {
                    panic!("restored campaign failed after provenance {field}: {error}")
                });
            assert!(
                matches!(&error, GauntletError::InvalidCampaign { .. }),
                "provenance {field} must reach campaign provenance validation, got: {error}"
            );
        }

        let foreign_hash = report
            .cases
            .iter()
            .enumerate()
            .find_map(|(ordinal, result)| {
                (ordinal != target_ordinal)
                    .then(|| result.artifact_hash.clone())
                    .flatten()
            })
            .expect("live Core V3 campaign has another valid immutable object");
        let mut run_manifest: crate::artifact::RunManifest =
            serde_json::from_slice(&original_case_bytes)
                .expect("decode scratch target run manifest for ordinal swap");
        run_manifest.object_hash.clone_from(&foreign_hash);
        std::fs::write(
            &target_case_path,
            serde_json::to_vec(&run_manifest).expect("encode ordinal-swapped run manifest"),
        )
        .expect("write ordinal-swapped run manifest");
        let mut swapped_report = report.clone();
        swapped_report.cases[target_ordinal].artifact_hash = Some(foreign_hash);
        std::fs::write(
            &report_path,
            swapped_report
                .canonical_bytes()
                .expect("encode ordinal-swapped campaign report"),
        )
        .expect("write ordinal-swapped campaign report");
        let error = store.load_verified_campaign(&report.run_id).expect_err(
            "campaign evidence validator must reject a valid object at the wrong ordinal",
        );
        std::fs::write(&target_case_path, &original_case_bytes)
            .expect("restore target run manifest after ordinal swap");
        std::fs::write(&report_path, &original_report_bytes)
            .expect("restore campaign report after ordinal swap");
        store
            .load_verified_campaign(&report.run_id)
            .unwrap_or_else(|error| panic!("restored campaign failed after ordinal swap: {error}"));
        assert!(
            matches!(&error, GauntletError::InvalidCampaign { .. })
                && error
                    .to_string()
                    .contains("campaign case result does not match its immutable artifact"),
            "valid but ordinal-mismatched object must reach campaign evidence validation: {error}"
        );
        existing_mutation_count += 1;
        assert_eq!(
            existing_mutation_count, 8,
            "the pre-V3 mutation corpus must execute one raw-byte case, six missing-lane cases, \
             and one wrong-ordinal case"
        );
        assert_eq!(
            existing_mutation_count + typed_mutations.len() + provenance_mutations.len(),
            88,
            "the persisted Core V3 replay corpus must execute every pre-V3, typed, and provenance \
             mutation"
        );
    }

    #[cfg(feature = "tantivy-oracle")]
    fn live_pr_artifact_root(fallback: &std::path::Path, profile: &str) -> std::path::PathBuf {
        let root = std::env::var_os("GAUNTLET_ARTIFACT_ROOT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| fallback.to_path_buf())
            .join(profile);
        std::fs::create_dir_all(&root).expect("create PR campaign artifact root");
        root
    }

    #[cfg(feature = "tantivy-oracle")]
    fn assert_cass_campaign_is_nonvacuous(root: &std::path::Path, report: &CampaignReport) {
        let selected = report.selected_queries().expect("selected CASS cases");
        assert_eq!(selected.len(), report.cases.len());
        for (query, result) in selected.into_iter().zip(&report.cases) {
            let object = load_campaign_case_object(root, result);
            assert!(
                object.comparison.subject.snippets.is_empty()
                    && object.comparison.oracle.snippets.is_empty(),
                "{}: CASS activation must remain snippet-free",
                query.id
            );
            if matches!(
                &query.query_kind,
                GeneratedQueryKind::Boolean
                    | GeneratedQueryKind::Glob { .. }
                    | GeneratedQueryKind::Range { .. }
                    | GeneratedQueryKind::StructuredFilter { .. }
            ) {
                for (engine, count) in [
                    ("subject", &object.comparison.subject.match_count),
                    ("oracle", &object.comparison.oracle.match_count),
                ] {
                    assert!(
                        matches!(count, crate::comparator::CountState::Value(value) if *value > 0),
                        "{}: {engine} coverage probe must match at least one document, got {count:?}",
                        query.id
                    );
                }
            }
        }
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn harvested_14_score_bits_preserve_ranked_and_counted_orders() {
        let fixture = make_scalar_g1a_regression_fixture();
        let lexical_revision = oracle_version_contract()
            .expect("oracle version contract")
            .lexical_git_revision;
        let config = frankensearch_quill::QuillConfig {
            deterministic_ingest: true,
            ..frankensearch_quill::QuillConfig::default()
        };
        let mut subject =
            crate::engine::QuillSubject::in_memory(config, "harvested-14-score-regression", false)
                .expect("fresh scalar Quill subject");
        let mut oracle =
            crate::engine::TantivyOracle::in_memory_scalar_g1a(&lexical_revision, false)
                .expect("fresh scalar G1a Tantivy oracle");
        let semantic_contract = SemanticContract::scalar_g1a();

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            DifferentialCampaignEngine::begin_corpus(
                &mut subject,
                &cx,
                &fixture.corpus_manifest,
                &semantic_contract,
            )
            .await
            .expect("begin Quill corpus");
            DifferentialCampaignEngine::begin_corpus(
                &mut oracle,
                &cx,
                &fixture.corpus_manifest,
                &semantic_contract,
            )
            .await
            .expect("begin Tantivy corpus");
            for batch in fixture.documents.chunks(5) {
                DifferentialCampaignEngine::index_batch(&mut subject, &cx, batch)
                    .await
                    .expect("index Quill batch");
                DifferentialCampaignEngine::index_batch(&mut oracle, &cx, batch)
                    .await
                    .expect("index Tantivy batch");
            }
            DifferentialCampaignEngine::commit_corpus(
                &mut subject,
                &cx,
                &fixture.corpus_manifest,
                &semantic_contract,
            )
            .await
            .expect("commit Quill corpus");
            DifferentialCampaignEngine::commit_corpus(
                &mut oracle,
                &cx,
                &fixture.corpus_manifest,
                &semantic_contract,
            )
            .await
            .expect("commit Tantivy corpus");

            const TARGET: &str = "test-cooking-015";
            const FULL_QUERY: &str = "how to sear a steak properly";
            let mut term_scores = Vec::new();
            for query in ["how", "to", "sear", "a", "steak", "properly"] {
                let quill = subject
                    .index()
                    .expect("committed Quill index")
                    .search_paginated(&cx, query, 100, 0, false)
                    .expect("search ranked Quill");
                let tantivy = oracle
                    .index()
                    .search_doc_ids(&cx, query, 100)
                    .expect("search ranked Tantivy");
                let quill_score = quill
                    .hits
                    .iter()
                    .find(|hit| hit.document_id == TARGET)
                    .map(|hit| hit.score);
                let tantivy_score = tantivy
                    .iter()
                    .find(|hit| hit.doc_id.as_str() == TARGET)
                    .map(|hit| hit.bm25_score);
                eprintln!(
                    "harvested-14 query={query:?} quill={:?} tantivy={:?}",
                    quill_score.map(f32::to_bits),
                    tantivy_score.map(f32::to_bits)
                );
                assert_eq!(
                    quill_score.map(f32::to_bits),
                    tantivy_score.map(f32::to_bits),
                    "single-term score drift for {query:?}"
                );
                if let Some(score) = quill_score {
                    term_scores.push((query, score));
                }
            }
            let parse_order_sum = term_scores
                .iter()
                .fold(0.0_f32, |sum, (_, score)| sum + score);
            eprintln!(
                "harvested-14 term_scores={:?} parse_order_sum={:#010x}",
                term_scores
                    .iter()
                    .map(|(term, score)| (*term, score.to_bits()))
                    .collect::<Vec<_>>(),
                parse_order_sum.to_bits()
            );

            let quill_ranked = subject
                .index()
                .expect("committed Quill index")
                .search_paginated(&cx, FULL_QUERY, 100, 0, false)
                .expect("search ranked Quill aggregate");
            let tantivy_ranked = oracle
                .index()
                .search_doc_ids(&cx, FULL_QUERY, 100)
                .expect("search ranked Tantivy aggregate");
            let quill_ranked_score = quill_ranked
                .hits
                .iter()
                .find(|hit| hit.document_id == TARGET)
                .expect("ranked Quill aggregate contains target")
                .score;
            let tantivy_ranked_score = tantivy_ranked
                .iter()
                .find(|hit| hit.doc_id.as_str() == TARGET)
                .expect("ranked Tantivy aggregate contains target")
                .bm25_score;
            assert_eq!(
                quill_ranked_score.to_bits(),
                tantivy_ranked_score.to_bits(),
                "ranked aggregate must preserve Tantivy TopDocs f32 accumulation order"
            );
            assert_eq!(
                quill_ranked_score.to_bits(),
                0x4005_5fc7,
                "fixture must keep exercising the ranked TopDocs order"
            );
            assert_eq!(
                parse_order_sum.to_bits(),
                quill_ranked_score.to_bits(),
                "ranked root must retain analyzed term order as children exhaust"
            );

            let quill_counted = subject
                .index()
                .expect("committed Quill index")
                .search_paginated(&cx, FULL_QUERY, 100, 0, true)
                .expect("search counted Quill aggregate");
            let tantivy_counted = oracle
                .index()
                .search_doc_ids_counted(&cx, FULL_QUERY, 100)
                .expect("search counted Tantivy aggregate");
            let quill_counted_score = quill_counted
                .hits
                .iter()
                .find(|hit| hit.document_id == TARGET)
                .expect("counted Quill aggregate contains target")
                .score;
            let tantivy_counted_score = tantivy_counted
                .iter()
                .find(|hit| hit.doc_id.as_str() == TARGET)
                .expect("counted Tantivy aggregate contains target")
                .bm25_score;
            assert_eq!(
                quill_counted_score.to_bits(),
                tantivy_counted_score.to_bits(),
                "counted aggregate must preserve Tantivy exhaustive f32 accumulation order"
            );
            assert_eq!(
                quill_counted_score.to_bits(),
                0x4005_5fc8,
                "fixture must keep exercising the exhaustive swap-remove order"
            );
            assert_ne!(
                quill_ranked_score.to_bits(),
                quill_counted_score.to_bits(),
                "fixture must distinguish Tantivy's ranked and counted collector contracts"
            );
        });
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn live_default_profile_campaign_stamps_provenance_and_reloads_verified() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = live_pr_artifact_root(temp.path(), "default");
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let report = run_live_default_profile_campaign(&cx, &root, "e6.9-default-pr-lane")
                .await
                .expect("live default-profile campaign must complete and pass");
            assert!(
                report.passed,
                "default profile must be admissible: rank_mismatches={:?} lexical_mismatches={:?} coverage={:?} cases={:?}",
                report.mismatches, report.lexical_mismatches, report.lexical_coverage, report.cases,
            );
            assert!(report.lexical_mismatches.is_empty());
            assert!(report.cases.iter().all(|case| {
                matches!(
                    &case.lexical_contract,
                    CampaignLexicalCaseSummary::CoreLexicalV3 {
                        status: LexicalComparisonStatus::Equivalent,
                        mismatch_count: 0,
                        ..
                    }
                )
            }));
            let CampaignLexicalCoverageSummary::CoreLexicalV3 {
                subject,
                oracle,
                admissible: true,
            } = &report.lexical_coverage
            else {
                panic!("default live campaign must have admissible core lexical v3 coverage");
            };
            for (engine, coverage) in [("subject", subject), ("oracle", oracle)] {
                assert!(
                    coverage.full_search.empty > 0,
                    "{engine} must exercise an empty ordinary search"
                );
                assert!(
                    coverage.fusion_candidates.empty > 0,
                    "{engine} must exercise an empty candidate search"
                );
                if coverage.metadata_deferred_cases > 0 {
                    for (shape, probe) in [
                        (
                            "all lexical winners",
                            &coverage.all_lexical_winners_hydration,
                        ),
                        (
                            "strict hybrid winners",
                            &coverage.strict_hybrid_winners_hydration,
                        ),
                        ("mixed winners", &coverage.mixed_winners_hydration),
                    ] {
                        assert!(
                            probe.restoration > 0,
                            "{engine} must exercise deferred metadata restoration for {shape}"
                        );
                    }
                }
            }

            const KNOWN_MISS_QUERY: &str = "flurbnozzlezyphraxicqvktmps";
            let selected_queries = report.selected_queries().expect("selected default queries");
            let known_miss = selected_queries
                .into_iter()
                .find(|query| query.query == KNOWN_MISS_QUERY)
                .expect("the frozen default suite must contain its known-miss query");
            let known_miss_result = report
                .cases
                .iter()
                .find(|result| result.case_id == known_miss.id)
                .expect("known-miss result follows the selected query manifest");
            let known_miss_object = load_campaign_case_object(&root, known_miss_result);
            assert!(
                known_miss_object.comparison.subject.hits.is_empty()
                    && known_miss_object.comparison.oracle.hits.is_empty(),
                "known-miss rich rank envelopes must both be empty"
            );
            for (engine, observation) in [
                ("subject", &known_miss_object.comparison.subject),
                ("oracle", &known_miss_object.comparison.oracle),
            ] {
                assert_eq!(
                    observation.match_count,
                    crate::comparator::CountState::Value(0),
                    "known-miss {engine} rich rank envelope must report an exact zero match count"
                );
            }
            let ArtifactLexicalContractEvidence::CoreLexicalV3 {
                comparison: known_miss_lexical,
            } = &known_miss_object.lexical_contract
            else {
                panic!("known-miss artifact must carry core lexical v3 evidence");
            };
            for bundle in [&known_miss_lexical.subject, &known_miss_lexical.oracle] {
                for outcome in [
                    &bundle.full_search().outcome,
                    &bundle.fusion_candidates().outcome,
                ] {
                    assert!(
                        matches!(
                            outcome,
                            crate::comparator::LexicalObservationOutcome::Success {
                                hits,
                                returned_count: 0,
                                empty_shape: crate::comparator::LexicalEmptyShape::Empty,
                                ..
                            } if hits.is_empty()
                        ),
                        "known-miss ordinary search and candidate lanes must be explicit empty successes"
                    );
                }
            }

            let mut exercised_persisted_metadata_tamper = false;
            for case in &report.cases {
                let hash = case
                    .artifact_hash
                    .as_deref()
                    .expect("passing core lexical case has an artifact");
                let bytes = std::fs::read(root.join("objects").join(format!("{hash}.json")))
                    .expect("read immutable core lexical artifact");
                let object: ArtifactObject =
                    serde_json::from_slice(&bytes).expect("decode core lexical artifact");
                object.validate().expect("untampered artifact validates");
                let mut tampered = object;
                let ArtifactLexicalContractEvidence::CoreLexicalV3 { comparison } =
                    &mut tampered.lexical_contract
                else {
                    panic!("default live artifact must carry core lexical v3 evidence");
                };
                let crate::comparator::LexicalObservationOutcome::Success { hits, .. } =
                    &mut comparison.subject.full_search.outcome
                else {
                    continue;
                };
                let Some(hit) = hits.first_mut() else {
                    continue;
                };
                hit.metadata = if matches!(
                    &hit.metadata,
                    crate::comparator::SensitiveValueObservation::Absent
                ) {
                    crate::comparator::SensitiveValueObservation::PresentEmpty {
                        sha256: sha256_text("{}"),
                        byte_len: 2,
                    }
                } else {
                    crate::comparator::SensitiveValueObservation::Absent
                };
                assert!(
                    tampered.validate().is_err(),
                    "persisted metadata presence tamper must fail replay"
                );
                exercised_persisted_metadata_tamper = true;
                break;
            }
            assert!(
                exercised_persisted_metadata_tamper,
                "live campaign must persist at least one metadata-bearing lexical hit"
            );

            let provenance = report
                .provenance
                .as_ref()
                .expect("production campaign stamps provenance");
            assert!(!provenance.subject_git_revision.is_empty());
            assert!(!provenance.cargo_lock_sha256.is_empty());
            assert!(provenance.rustc_version_verbose.contains("release:"));
            assert!(provenance.rust_toolchain_channel.starts_with("nightly-"));
            assert_eq!(
                provenance.unicode_version,
                format!(
                    "{}.{}.{}",
                    char::UNICODE_VERSION.0,
                    char::UNICODE_VERSION.1,
                    char::UNICODE_VERSION.2
                )
            );
            assert!(!provenance.unicode_normalization_version.is_empty());
            assert_eq!(
                provenance.unicode_normalization_table_version,
                unicode_normalization_table_version()
            );
            assert_eq!(provenance.query_generator_id, GENERATOR_ID);
            assert_eq!(
                provenance.query_generator_schema_version,
                QUERY_MANIFEST_SCHEMA_VERSION
            );
            assert_eq!(provenance.query_seed, 0x6201);
            assert!(is_lower_sha256(&provenance.query_source_identity_sha256));
            assert!(!provenance.query_profile_sha256.is_empty());
            assert_eq!(
                provenance.analyzer_contract_hash,
                report.semantic_contract.analyzer_contract_hash
            );
            assert_eq!(
                provenance.schema_contract_hash,
                report.semantic_contract.schema_contract_hash
            );
            assert_eq!(provenance.corpus_manifest_hash, report.corpus_manifest_hash);
            assert_eq!(provenance.query_manifest_hash, report.query_manifest_hash);
            assert_eq!(provenance.corpus_seed, None);

            // CI-grade acceptance: ONLY a verified reload counts as evidence.
            let reloaded = ArtifactStore::new(&root)
                .load_verified_campaign("e6.9-default-pr-lane")
                .expect("verified reload accepts the completed campaign");
            assert_eq!(reloaded, report);
            assert_eq!(reloaded.provenance, report.provenance);
            assert_persisted_core_object_mutation_matrix_fails(&root, &report);
        });
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn live_cass_profile_campaign_covers_contract_and_reloads_verified() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = live_pr_artifact_root(temp.path(), "cass");
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fixture = make_cass_activation_fixture();
            let report = run_live_cass_profile_fixture(&cx, &root, "e6.9-cass-pr-lane", &fixture)
                .await
                .expect("live CASS campaign must complete");
            assert!(
                report.passed,
                "CASS profile is green: {:?}",
                report.mismatches
            );
            assert_eq!(report.semantic_contract, SemanticContract::cass());
            assert_eq!(
                report.engines.subject.implementation,
                "frankensearch-quill/cass-index"
            );
            assert_eq!(
                report.engines.oracle.config_hash,
                crate::engine::CASS_TANTIVY_ORACLE_CONFIG_HASH
            );
            assert_cass_campaign_is_nonvacuous(&root, &report);
            let reloaded = ArtifactStore::new(&root)
                .load_verified_campaign("e6.9-cass-pr-lane")
                .expect("verified reload accepts the CASS campaign");
            assert_eq!(reloaded, report);
        });
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn live_campaign_provenance_mismatch_fails_closed() {
        let temp = tempfile::tempdir().expect("tempdir");
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            run_live_default_profile_campaign(&cx, temp.path(), "e6.9-provenance-tamper")
                .await
                .expect("campaign completes");
            // Tamper with the reservation's provenance block: the verified reload
            // must reject the campaign instead of trusting it.
            let reservation_path = temp
                .path()
                .join("campaigns")
                .join("e6.9-provenance-tamper")
                .join("reservation.json");
            let bytes = std::fs::read(&reservation_path).expect("read reservation");
            let mut reservation: serde_json::Value =
                serde_json::from_slice(&bytes).expect("parse reservation");
            reservation["provenance"]["cargo_lock_sha256"] =
                serde_json::Value::String("00".repeat(32));
            std::fs::write(
                &reservation_path,
                serde_json::to_vec(&reservation).expect("serialize tampered"),
            )
            .expect("write tampered");
            let rejected =
                ArtifactStore::new(temp.path()).load_verified_campaign("e6.9-provenance-tamper");
            assert!(rejected.is_err(), "tampered provenance fails closed");
        });
    }

    #[cfg(feature = "tantivy-oracle")]
    const NIGHTLY_REPOSITORY_PATHS: &[&str] = &[
        "README.md",
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        ".github/workflows/ci.yml",
        "tests/fixtures/corpus.json",
        "tests/fixtures/edge_cases.json",
        "tests/fixtures/queries.json",
        "tests/fixtures/quill_language_contract.json",
        "crates/frankensearch-lexical/Cargo.toml",
        "crates/frankensearch-lexical/src/cass_compat.rs",
        "crates/frankensearch-lexical/src/lib.rs",
        "crates/frankensearch-lexical/src/quill_contract.rs",
        "crates/frankensearch-quill/Cargo.toml",
        "crates/frankensearch-quill/src/argus.rs",
        "crates/frankensearch-quill/src/config.rs",
        "crates/frankensearch-quill/src/contract.rs",
        "crates/frankensearch-quill/src/delta.rs",
        "crates/frankensearch-quill/src/error.rs",
        "crates/frankensearch-quill/src/grimoire.rs",
        "crates/frankensearch-quill/src/index.rs",
        "crates/frankensearch-quill/src/keeper.rs",
        "crates/frankensearch-quill/src/lib.rs",
        "crates/frankensearch-quill/src/query.rs",
        "crates/frankensearch-quill/src/quiver.rs",
        "crates/frankensearch-quill/src/schema.rs",
        "crates/frankensearch-quill/src/scribe.rs",
        "crates/frankensearch-quill/src/segment.rs",
        "crates/frankensearch-quill/src/snippet.rs",
        "crates/frankensearch-quill/src/stats.rs",
        "crates/frankensearch-quill/src/tracing_conventions.rs",
        "crates/frankensearch-quill-gauntlet/Cargo.toml",
        "crates/frankensearch-quill-gauntlet/fixtures/generator-v2.json",
        "crates/frankensearch-quill-gauntlet/src/artifact.rs",
        "crates/frankensearch-quill-gauntlet/src/comparator.rs",
        "crates/frankensearch-quill-gauntlet/src/engine.rs",
        "crates/frankensearch-quill-gauntlet/src/generator.rs",
        "crates/frankensearch-quill-gauntlet/src/lib.rs",
        "crates/frankensearch-quill-gauntlet/src/runner.rs",
        "crates/frankensearch-quill-gauntlet/src/version_contract.rs",
    ];

    #[cfg(feature = "tantivy-oracle")]
    fn nightly_query_suite(corpus_manifest: &CorpusManifest, seed: u64) -> GeneratedQuerySuite {
        let corpus_hash = corpus_manifest.manifest_hash().expect("corpus hash");
        GeneratedQuerySuite::generate(
            QueryGeneratorSpec {
                seed,
                default_limit: 20,
                include_shared_relevance_queries: true,
            },
            &corpus_hash,
            &SharedFixtureSuite::load().expect("shared fixtures"),
        )
        .expect("nightly query suite")
    }

    #[cfg(feature = "tantivy-oracle")]
    fn nightly_generated_fixture() -> Fixture {
        let corpus = SyntheticCorpus::new(SyntheticCorpusSpec {
            seed: 0xE609,
            document_count: 2_000,
            vocabulary_size: 2_048,
            zipf_exponent: ZipfExponent::S11,
            max_document_bytes: 2_048,
        })
        .expect("synthetic spec");
        let documents = corpus.iter().collect::<Vec<_>>();
        let corpus_manifest = corpus.manifest().expect("synthetic corpus manifest");
        let corpus_hash = corpus_manifest.manifest_hash().expect("corpus hash");
        let query_suite = nightly_query_suite(&corpus_manifest, 0x9602);
        Fixture {
            documents,
            corpus_manifest,
            corpus_hash,
            query_suite,
        }
    }

    #[cfg(feature = "tantivy-oracle")]
    fn nightly_repository_fixture() -> Fixture {
        let snapshot = RepositorySnapshot::from_tracked_paths(
            &workspace_root(),
            "frankensearch-e6-nightly",
            NIGHTLY_REPOSITORY_PATHS
                .iter()
                .copied()
                .map(std::path::PathBuf::from),
        )
        .expect("content-addressed repository snapshot");
        assert!(
            snapshot.manifest.skipped_repository_entries.is_empty(),
            "pinned nightly repository paths must all be readable UTF-8 files: {:?}",
            snapshot.manifest.skipped_repository_entries
        );
        assert_eq!(
            snapshot.documents.len(),
            NIGHTLY_REPOSITORY_PATHS.len(),
            "nightly repository snapshot must include every pinned path"
        );
        let corpus_hash = snapshot.manifest.manifest_hash().expect("corpus hash");
        let query_suite = nightly_query_suite(&snapshot.manifest, 0x9603);
        Fixture {
            documents: snapshot.documents,
            corpus_manifest: snapshot.manifest,
            corpus_hash,
            query_suite,
        }
    }

    #[cfg(feature = "tantivy-oracle")]
    fn required_nightly_artifact_root() -> std::path::PathBuf {
        let root = std::env::var_os("GAUNTLET_ARTIFACT_ROOT")
            .map(std::path::PathBuf::from)
            .expect("nightly lane requires GAUNTLET_ARTIFACT_ROOT");
        assert!(
            root.is_relative()
                && root.starts_with("target/coverage")
                && !root
                    .components()
                    .any(|component| component == std::path::Component::ParentDir),
            "nightly artifacts must use a relative target/coverage path so RCH returns them"
        );
        std::fs::create_dir_all(&root).expect("create nightly campaign artifact root");
        root
    }

    #[cfg(feature = "tantivy-oracle")]
    async fn run_and_reload_default_nightly_campaign(
        cx: &Cx,
        root: &std::path::Path,
        run_id: &str,
        fixture: &Fixture,
    ) -> CampaignReport {
        std::fs::create_dir_all(root).expect("create default nightly campaign root");
        let report = run_live_default_profile_fixture(cx, root, run_id, fixture)
            .await
            .expect("nightly full lane completes");
        assert!(report.passed, "nightly lane green: {:?}", report.mismatches);
        let reloaded = ArtifactStore::new(root)
            .load_verified_campaign(run_id)
            .expect("verified reload accepts the nightly campaign");
        assert_eq!(reloaded, report);
        reloaded
    }

    #[cfg(feature = "tantivy-oracle")]
    async fn run_and_reload_cass_nightly_campaign(
        cx: &Cx,
        root: &std::path::Path,
        run_id: &str,
        fixture: &Fixture,
    ) -> CampaignReport {
        std::fs::create_dir_all(root).expect("create CASS nightly campaign root");
        let report = run_live_cass_profile_fixture(cx, root, run_id, fixture)
            .await
            .expect("CASS nightly full lane completes");
        assert!(
            report.passed,
            "CASS nightly lane green: {:?}",
            report.mismatches
        );
        let reloaded = ArtifactStore::new(root)
            .load_verified_campaign(run_id)
            .expect("verified reload accepts the CASS nightly campaign");
        assert_eq!(reloaded, report);
        reloaded
    }

    #[cfg(feature = "tantivy-oracle")]
    fn assert_same_seed_campaign_replay(first: &CampaignReport, replay: &CampaignReport) {
        assert_eq!(
            first.corpus_manifest_hash, replay.corpus_manifest_hash,
            "same-seed corpus manifest"
        );
        assert_eq!(
            first.query_manifest_hash, replay.query_manifest_hash,
            "same-seed query manifest"
        );
        assert_eq!(
            first
                .mismatches
                .iter()
                .map(|mismatch| mismatch.signature.as_str())
                .collect::<Vec<_>>(),
            replay
                .mismatches
                .iter()
                .map(|mismatch| mismatch.signature.as_str())
                .collect::<Vec<_>>(),
            "same-seed mismatch signatures"
        );
        assert_eq!(
            first.report_hash().expect("first report hash"),
            replay.report_hash().expect("replay report hash"),
            "same-seed report hash"
        );
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    #[ignore = "nightly full lane: both profiles over pinned generated + repository corpora"]
    fn live_both_profiles_campaign_nightly_full_lane() {
        let artifact_root = required_nightly_artifact_root();
        let generated = nightly_generated_fixture();
        let repository = nightly_repository_fixture();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (label, fixture) in [("generated", &generated), ("repository", &repository)] {
                for profile in ["default", "cass"] {
                    let run_id = format!("e6.9-{profile}-nightly-{label}");
                    let first_root = artifact_root.join(profile).join("first");
                    let replay_root = artifact_root.join(profile).join("replay");
                    let (first, replay) = if profile == "default" {
                        (
                            run_and_reload_default_nightly_campaign(
                                &cx,
                                &first_root,
                                &run_id,
                                fixture,
                            )
                            .await,
                            run_and_reload_default_nightly_campaign(
                                &cx,
                                &replay_root,
                                &run_id,
                                fixture,
                            )
                            .await,
                        )
                    } else {
                        (
                            run_and_reload_cass_nightly_campaign(
                                &cx,
                                &first_root,
                                &run_id,
                                fixture,
                            )
                            .await,
                            run_and_reload_cass_nightly_campaign(
                                &cx,
                                &replay_root,
                                &run_id,
                                fixture,
                            )
                            .await,
                        )
                    };
                    assert_same_seed_campaign_replay(&first, &replay);
                }
            }
        });
    }
}
