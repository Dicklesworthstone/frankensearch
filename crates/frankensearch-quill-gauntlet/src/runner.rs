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

use crate::GauntletError;
use crate::artifact::{ArtifactObject, ArtifactStore, CampaignArtifactContext};
use crate::comparator::{
    ComparatorConfig, ComparisonReport, ComparisonStatus, Divergence, DivergenceClass,
    EngineObservation, RankClass, compare_observations,
};
#[cfg(feature = "tantivy-oracle")]
use crate::engine::GauntletEngine;
use crate::engine::{
    ComparisonMode, DifferentialCase, DifferentialCaseMetadata, EngineDescriptor,
    EnginePairIdentity, HarnessRun, MAX_SNIPPET_CHARS,
};
use crate::generator::{
    CorpusManifest, GENERATOR_ID, GeneratedDocument, GeneratedQueryCase, GeneratedQueryKind,
    GeneratedQuerySuite, GlobPatternClass, MAX_DOCUMENT_BYTES, MAX_QUERY_CASES, MAX_QUERY_ID_BYTES,
    QuerySuiteSource, QuerySyntax, RangeClass, SyntheticCorpus, is_canonical_query_id,
};

/// Schema version for deterministic campaign reports.
pub const CAMPAIGN_REPORT_SCHEMA_VERSION: u32 = 1;
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
/// Default schema/query/ranking protocol implemented by the shipping Tantivy adapter.
pub const DEFAULT_SCHEMA_CONTRACT_HASH: &str =
    "9fed22a53e5060243e9528fbbf40605a0df8ea120b3d74ac41ecbb097c2df571";
const MISMATCH_SIGNATURE_DOMAIN: &[u8] = b"frankensearch/quill/mismatch-signature/v1\0";
const CAMPAIGN_REPORT_HASH_DOMAIN: &[u8] = b"frankensearch/quill/campaign-report/v1\0";
const DIVERGENCE_REGISTRY_HASH_DOMAIN: &[u8] = b"frankensearch/quill/divergence-registry/v1\0";
const MAX_DIVERGENCE_REGISTRY_ENTRIES: usize = 1_024;
const MAX_DIVERGENCE_REGISTER_PROSE_BYTES: usize = 64 * 1024;
const MAX_DIVERGENCE_REVIEWER_BYTES: usize = 1_024;
const MAX_DIVERGENCE_REGISTRY_TEXT_BYTES: usize = 16 * 1024 * 1024;
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
pub trait DifferentialCampaignEngine: Send {
    fn descriptor(&self) -> EngineDescriptor;
    /// Adapter-owned semantic identity; never copied from the runner request.
    fn semantic_contract(&self) -> SemanticContract;

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
                        && case.filters.created_from_ms.is_none()
                        && case.filters.created_to_ms.is_none()
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
        let invalid_class = !matches!(self.class, DivergenceClass::OversizedQueryToken);
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

/// Deterministic runner policy included in the report hash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CampaignConfig {
    pub selection: CampaignSelection,
    pub comparator_config: ComparatorConfig,
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

/// Stable evidence row for one campaign query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CampaignCaseResult {
    pub case_id: String,
    pub query_class: String,
    pub disposition: CampaignDisposition,
    pub comparison_status: Option<ComparisonStatus>,
    pub rank_class: Option<RankClass>,
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
    /// `std::char::UNICODE_VERSION` of the executing toolchain.
    pub unicode_version: String,
    /// Locked `unicode-normalization` crate version from `Cargo.lock`.
    pub unicode_normalization_version: String,
    /// Canonical hash of the campaign query profile (selection + profile).
    pub query_profile_sha256: String,
    /// Synthetic corpus seed when the corpus is generator-produced.
    pub corpus_seed: Option<u64>,
}

impl CampaignProvenance {
    /// Collect the full immutable provenance for one campaign execution.
    ///
    /// Git state is read from the invoking checkout unless overridden by the
    /// `GAUNTLET_SUBJECT_REVISION` / `GAUNTLET_SUBJECT_DIRTY` and matching
    /// oracle environment variables (the CI path supplies them explicitly so
    /// runners never guess). Toolchain facts come from the executing `rustc`
    /// and the workspace `Cargo.lock`.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError::InvalidCampaign`] when any required fact
    /// cannot be collected; a campaign must fail closed, never guess.
    pub fn collect(
        corpus_seed: Option<u64>,
        selection: &CampaignSelection,
    ) -> Result<Self, GauntletError> {
        let (subject_git_revision, subject_source_dirty) =
            collect_git_state("GAUNTLET_SUBJECT_REVISION", "GAUNTLET_SUBJECT_DIRTY")?;
        let (oracle_git_revision, oracle_source_dirty) =
            collect_git_state("GAUNTLET_ORACLE_REVISION", "GAUNTLET_ORACLE_DIRTY")?;
        let cargo_lock_sha256 = hash_workspace_lockfile()?;
        let rustc_version_verbose = collect_rustc_verbose()?;
        let unicode_normalization_version = locked_crate_version("unicode-normalization")?;
        let profile_bytes =
            serde_json::to_vec(selection).map_err(|error| GauntletError::InvalidCampaign {
                reason: format!("query profile serialization failed: {error}"),
            })?;
        let query_profile_sha256 = sha256_hex(&profile_bytes);
        Ok(Self {
            subject_git_revision,
            subject_source_dirty,
            oracle_git_revision,
            oracle_source_dirty,
            cargo_lock_sha256,
            rustc_version_verbose,
            unicode_version: format!(
                "{}.{}.{}",
                char::UNICODE_VERSION.0,
                char::UNICODE_VERSION.1,
                char::UNICODE_VERSION.2
            ),
            unicode_normalization_version,
            query_profile_sha256,
            corpus_seed,
        })
    }
}

fn collect_git_state(revision_env: &str, dirty_env: &str) -> Result<(String, bool), GauntletError> {
    if let (Ok(revision), Ok(dirty)) = (std::env::var(revision_env), std::env::var(dirty_env)) {
        return Ok((revision, dirty == "1" || dirty.eq_ignore_ascii_case("true")));
    }
    let revision = run_capture("git", &["rev-parse", "HEAD"], revision_env)?;
    let porcelain = run_capture("git", &["status", "--porcelain"], dirty_env)?;
    Ok((revision.trim().to_owned(), !porcelain.trim().is_empty()))
}

fn run_capture(program: &str, args: &[&str], label: &str) -> Result<String, GauntletError> {
    let output = std::process::Command::new(program)
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
    run_capture("rustc", &["-Vv"], "rustc -Vv")
}

fn workspace_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
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
            validate_campaign_case_result(query, result, &self.divergence_registry)?;
        }
        let confidence = f64::from_bits(self.config.posterior_confidence_bits);
        if summarize_query_classes(&self.cases, confidence) != self.query_classes {
            return Err(campaign_error(
                "campaign report query-class summaries are inconsistent",
            ));
        }
        validate_mismatch_groups(&self.mismatches, &self.cases)?;
        if self.passed != self.cases.iter().all(|result| result.disposition.passes()) {
            return Err(campaign_error(
                "campaign report pass bit does not match case dispositions",
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
        let (disposition, reason, registered_divergence) =
            classify_case(query, &object.comparison, &self.report.divergence_registry);
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
            || result.first_divergence != object.comparison.first_divergence
            || result.artifact_hash.as_deref() != Some(object_hash)
        {
            return Err(campaign_error(
                "campaign case result does not match its immutable artifact",
            ));
        }
        self.mismatches.record(&object.comparison, &query.id)?;
        self.advance()
    }

    pub(super) fn finish(self) -> Result<(), GauntletError> {
        if self.next_ordinal != self.selected.len() || self.next_ordinal != self.report.cases.len()
        {
            return Err(campaign_error(
                "campaign evidence ended before every selected query was validated",
            ));
        }
        if self.mismatches.finish() != self.report.mismatches {
            return Err(campaign_error(
                "campaign mismatch groups do not match immutable case artifacts",
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
        && result.artifact_hash.as_deref().is_some_and(is_lower_xxh3);
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
    if !valid_shape {
        return Err(campaign_error(
            "campaign case disposition and evidence fields are inconsistent",
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
            validate_engine_state(
                &*index_session.subject,
                &*index_session.oracle,
                &engines,
                &self.semantic_contract,
            )?;
            let result = match (subject_result, oracle_result) {
                (Ok(subject_observation), Ok(oracle_observation)) => self.finish_case(
                    run_id,
                    ordinal,
                    query,
                    query_class,
                    &query_manifest_hash,
                    query_suite.manifest.source,
                    &query_suite.manifest.source_identity_sha256,
                    &engines,
                    corpus_manifest.document_count,
                    evidence_case,
                    subject_observation,
                    oracle_observation,
                    &mut mismatches,
                ),
                (subject_result, oracle_result) => {
                    let (reason, diagnostic) = engine_error_details(subject_result, oracle_result);
                    CampaignCaseResult {
                        case_id: query.id.clone(),
                        query_class,
                        disposition: CampaignDisposition::InfrastructureError,
                        comparison_status: None,
                        rank_class: None,
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
        let passed = cases.iter().all(|result| result.disposition.passes());
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
        engines: &EnginePairIdentity,
        expected_doc_count: u64,
        evidence_case: DifferentialCase,
        subject: EngineObservation,
        oracle: EngineObservation,
        mismatches: &mut MismatchCollection,
    ) -> CampaignCaseResult {
        if subject.doc_count != expected_doc_count && oracle.doc_count != expected_doc_count {
            return infrastructure_case(
                query,
                query_class,
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
                    "mismatch_budget_exceeded",
                    error.to_string(),
                );
            }
        };
        let (disposition, reason, registered_divergence) =
            classify_case(query, &comparison, &self.registry);
        let first_divergence = comparison.first_divergence.clone();
        let comparison_status = Some(comparison.status);
        let rank_class = Some(comparison.rank_class);
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
            query: query.clone(),
            registered_divergence: registered_divergence.clone(),
        };
        let object = match ArtifactObject::from_campaign_run(run, context) {
            Ok(object) => object,
            Err(error) => {
                return infrastructure_case(
                    query,
                    query_class,
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
                        "artifact_prepare_failed",
                        error.to_string(),
                    );
                }
            };
        if let Err(error) = self.store.persist(&prepared) {
            return infrastructure_case(
                query,
                query_class,
                "artifact_persist_failed",
                error.to_string(),
            );
        }
        mismatches.apply(&object.comparison, &query.id, mismatch_text_bytes);
        CampaignCaseResult {
            case_id: query.id.clone(),
            query_class,
            disposition,
            comparison_status,
            rank_class,
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
    reason: &'static str,
    diagnostic: String,
) -> CampaignCaseResult {
    CampaignCaseResult {
        case_id: query.id.clone(),
        query_class,
        disposition: CampaignDisposition::InfrastructureError,
        comparison_status: None,
        rank_class: None,
        artifact_hash: None,
        registered_divergence: None,
        first_divergence: None,
        reason: Some(reason.to_owned()),
        diagnostic: Some(diagnostic),
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

fn is_auto_class(class: DivergenceClass) -> bool {
    match class {
        DivergenceClass::TieOrder | DivergenceClass::ScoreEpsilon => true,
        DivergenceClass::RankMismatch
        | DivergenceClass::SnippetMismatch
        | DivergenceClass::CountMismatch
        | DivergenceClass::DocumentCountMismatch
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
        | DivergenceClass::RankMismatch => format!(
            "rank:{}:{}",
            rank_value(&divergence.oracle),
            rank_value(&divergence.subject)
        ),
        DivergenceClass::SnippetMismatch => format!(
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
        DivergenceClass::OversizedQueryToken => format!(
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

fn engine_error_details(
    subject: Result<EngineObservation, GauntletError>,
    oracle: Result<EngineObservation, GauntletError>,
) -> (String, String) {
    match (subject, oracle) {
        (Err(subject), Err(oracle)) => (
            "both_engine_executions_failed".to_owned(),
            format!("subject: {subject}; oracle: {oracle}"),
        ),
        (Err(subject), Ok(_)) => ("subject_execution_failed".to_owned(), subject.to_string()),
        (Ok(_), Err(oracle)) => ("oracle_execution_failed".to_owned(), oracle.to_string()),
        (Ok(_), Ok(_)) => (
            "invalid_engine_error_state".to_owned(),
            "both engine results unexpectedly succeeded".to_owned(),
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
            DivergenceClass::CountMismatch | DivergenceClass::DocumentCountMismatch => {
                evidence.push("count evidence disagrees with the oracle".to_owned());
                (SuspectedLayer::Indexing, TriageConfidence::Medium)
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
    use std::io::{self, Write};
    #[cfg(feature = "tantivy-oracle")]
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::comparator::{AstDifference, AstLoweringKind, CountState, NativeTieKey, RankedHit};
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
        assert!(
            logs.lines().any(|line| {
                is_stage_close(line, ARGUS_QUERY)
                    && line.contains("offset=17")
                    && line.contains("exact_count=true")
            }),
            "live G1a trace did not execute Quill's paginated collector: {logs}",
        );
        assert!(
            logs.lines().any(|line| {
                is_stage_close(line, ARGUS_QUERY) && line.contains("exact_count=false")
            }),
            "live G1a trace did not execute Quill's count-free collector: {logs}",
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
            let mut changed_query = report.clone();
            changed_query.query_suite.cases[0]
                .query
                .push_str(" tampered");
            assert!(changed_query.canonical_bytes().is_err());
            let mut wrong_artifact = report.clone();
            wrong_artifact.cases[0].artifact_hash = Some("0".repeat(16));
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
                    25,
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
                assert_eq!(
                    actual_golden.schema_version, expected_golden.schema_version,
                    "corpus_hash={} query_seed={} rank golden schema drifted",
                    first.corpus_manifest_hash, actual_golden.query_seed,
                );
                assert_eq!(
                    actual_golden.corpus_manifest_hash, expected_golden.corpus_manifest_hash,
                    "corpus_hash={} query_seed={} rank golden corpus binding drifted",
                    first.corpus_manifest_hash, actual_golden.query_seed,
                );
                assert_eq!(
                    actual_golden.query_manifest_hash,
                    expected_golden.query_manifest_hash,
                    "corpus_hash={} query_seed={} rank golden query-manifest binding drifted; actual={}",
                    first.corpus_manifest_hash,
                    actual_golden.query_seed,
                    serde_json::to_string_pretty(&actual_golden)
                        .expect("serialize actual E4.10 rank-list golden"),
                );
                assert_eq!(
                    actual_golden.query_seed, expected_golden.query_seed,
                    "corpus_hash={} query_seed={} rank golden replay seed drifted",
                    first.corpus_manifest_hash, actual_golden.query_seed,
                );
                assert_eq!(
                    actual_golden.cases.len(),
                    expected_golden.cases.len(),
                    "corpus_hash={} query_seed={} rank golden case count drifted",
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
    fn asymmetric_document_count_drift_is_persisted_but_shared_drift_fails_closed() {
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
                CampaignDisposition::Unclassified
            );
            assert_eq!(
                report.cases[0].first_divergence.as_deref(),
                Some("/comparison/subject/doc_count")
            );
            let object_hash = report.cases[0]
                .artifact_hash
                .as_deref()
                .expect("persisted mismatch object");
            let object: ArtifactObject = serde_json::from_slice(
                &std::fs::read(root.join("objects").join(format!("{object_hash}.json")))
                    .expect("read mismatch object"),
            )
            .expect("decode mismatch object");
            assert!(
                object.comparison.divergences.iter().any(|divergence| {
                    divergence.class == DivergenceClass::DocumentCountMismatch
                })
            );
            let report_path = root.join("campaigns/asymmetric-doc-count/report.json");
            let stored: CampaignReport = serde_json::from_slice(
                &std::fs::read(report_path).expect("stored unclassified report"),
            )
            .expect("decode unclassified report");
            assert!(!stored.passed);
            assert_eq!(
                stored.cases[0].disposition,
                CampaignDisposition::Unclassified
            );
            let mut forged_semantics = object.clone();
            forged_semantics
                .campaign
                .as_mut()
                .expect("campaign context")
                .semantic_contract
                .schema_contract_hash = "f".repeat(64);
            assert!(
                ArtifactStore::new(root.join("forged"))
                    .prepare(
                        "forged-semantic-contract",
                        &forged_semantics,
                        BTreeMap::new()
                    )
                    .is_err()
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
        let version = oracle_version_contract().expect("oracle version");
        let revision = version.lexical_git_revision;
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
            let mut before_begin =
                crate::TantivyOracle::in_memory(&revision, false).expect("oracle before begin");
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

            let mut before_commit =
                crate::TantivyOracle::in_memory(&revision, false).expect("oracle before commit");
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

            let mut oracle = crate::TantivyOracle::in_memory(&revision, false)
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
        let version = oracle_version_contract().expect("oracle version");
        let contract = SemanticContract::shipping_default();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let mut oracle = crate::TantivyOracle::in_memory(&version.lexical_git_revision, false)
                .expect("fresh oracle");
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
    fn live_campaign_engines() -> (crate::engine::QuillSubject, crate::engine::TantivyOracle) {
        let lexical_revision = oracle_version_contract()
            .expect("oracle version contract")
            .lexical_git_revision;
        let config = frankensearch_quill::QuillConfig {
            deterministic_ingest: true,
            ..frankensearch_quill::QuillConfig::default()
        };
        let subject =
            crate::engine::QuillSubject::in_memory(config, "e6.9-live-activation-subject", false)
                .expect("fresh scalar Quill subject");
        let oracle = crate::engine::TantivyOracle::in_memory_scalar_g1a(&lexical_revision, false)
            .expect("fresh scalar G1a Tantivy oracle");
        (subject, oracle)
    }

    #[cfg(feature = "tantivy-oracle")]
    async fn run_live_default_profile_campaign(
        cx: &Cx,
        root: &std::path::Path,
        run_id: &str,
    ) -> Result<CampaignReport, GauntletError> {
        let fixture = make_scalar_g1a_regression_fixture();
        let (mut subject, mut oracle) = live_campaign_engines();
        let selection = CampaignSelection::DefaultSyntax;
        let provenance =
            CampaignProvenance::collect(Some(0x6201), &selection).expect("collect provenance");
        let campaign = DifferentialCampaignRunner::new(
            ArtifactStore::new(root),
            SemanticContract::scalar_g1a(),
            CampaignConfig {
                selection,
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
    #[test]
    fn live_default_profile_campaign_stamps_provenance_and_reloads_verified() {
        let temp = tempfile::tempdir().expect("tempdir");
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let report =
                run_live_default_profile_campaign(&cx, temp.path(), "e6.9-default-pr-lane")
                    .await
                    .expect("live default-profile campaign must complete and pass");
            assert!(
                report.passed,
                "default profile is green: {:?}",
                report.mismatches
            );
            let provenance = report
                .provenance
                .as_ref()
                .expect("production campaign stamps provenance");
            assert!(!provenance.subject_git_revision.is_empty());
            assert!(!provenance.cargo_lock_sha256.is_empty());
            assert!(provenance.rustc_version_verbose.contains("release:"));
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
            assert!(!provenance.query_profile_sha256.is_empty());

            // CI-grade acceptance: ONLY a verified reload counts as evidence.
            let reloaded = ArtifactStore::new(temp.path())
                .load_verified_campaign("e6.9-default-pr-lane")
                .expect("verified reload accepts the completed campaign");
            assert_eq!(reloaded, report);
            assert_eq!(reloaded.provenance, report.provenance);
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
    fn documents_content_hash(documents: &[GeneratedDocument]) -> String {
        let mut hasher = Sha256::new();
        for document in documents {
            let bytes = serde_json::to_vec(document).expect("document json");
            hasher.update((bytes.len() as u64).to_be_bytes());
            hasher.update(&bytes);
        }
        let digest = hasher.finalize();
        let mut output = String::with_capacity(64);
        for byte in digest {
            use std::fmt::Write as _;
            let _ = write!(output, "{byte:02x}");
        }
        output
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    #[ignore = "nightly full lane: pinned generated + repository-scale corpora"]
    fn live_default_profile_campaign_nightly_full_lane() {
        let temp = tempfile::tempdir().expect("tempdir");
        // Generated lane: pinned synthetic corpus, deterministic replay.
        let spec = SyntheticCorpusSpec {
            seed: 0xE609,
            document_count: 2_000,
            vocabulary_size: 2_048,
            zipf_exponent: ZipfExponent::S11,
            max_document_bytes: 2_048,
        };
        let documents: Vec<_> = SyntheticCorpus::new(spec.clone())
            .expect("synthetic spec")
            .iter()
            .collect();
        let corpus_manifest = CorpusManifest {
            schema_version: 1,
            generator_id: GENERATOR_ID.to_owned(),
            source: crate::generator::CorpusSourceManifest::Synthetic { spec: spec.clone() },
            document_count: u64::try_from(documents.len()).expect("doc count"),
            total_content_bytes: documents
                .iter()
                .map(|document| u64::try_from(document.content.len()).unwrap_or(u64::MAX))
                .sum(),
            content_sha256: documents_content_hash(&documents),
            skipped_repository_entries: Vec::new(),
        };
        let corpus_hash = corpus_manifest.manifest_hash().expect("corpus hash");
        let query_suite = GeneratedQuerySuite::generate(
            QueryGeneratorSpec {
                seed: 0x9602,
                default_limit: 20,
                include_shared_relevance_queries: true,
            },
            &corpus_hash,
            &SharedFixtureSuite::load().expect("shared fixtures"),
        )
        .expect("query suite");
        let (mut subject, mut oracle) = live_campaign_engines();
        let selection = CampaignSelection::DefaultSyntax;
        let provenance =
            CampaignProvenance::collect(Some(spec.seed), &selection).expect("collect provenance");
        let campaign = DifferentialCampaignRunner::new(
            ArtifactStore::new(temp.path()),
            SemanticContract::scalar_g1a(),
            CampaignConfig {
                selection,
                index_batch_size: 64,
                snippet_max_chars: None,
                ..CampaignConfig::default()
            },
            DivergenceRegistry::default(),
        )
        .expect("nightly campaign runner")
        .with_provenance(provenance);
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let report = campaign
                .run(
                    &cx,
                    "e6.9-default-nightly-full",
                    &mut subject,
                    &mut oracle,
                    &documents,
                    &corpus_manifest,
                    &query_suite,
                )
                .await
                .expect("nightly full lane completes");
            assert!(report.passed, "nightly lane green: {:?}", report.mismatches);
            ArtifactStore::new(temp.path())
                .load_verified_campaign("e6.9-default-nightly-full")
                .expect("verified reload accepts the nightly campaign");
        });
    }
}
