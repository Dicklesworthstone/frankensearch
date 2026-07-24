use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;

use asupersync::Cx;
use frankensearch_quill::{QuillConfig, QuillIndex, QuillSearchResult};
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::xxh3_64;

use crate::GauntletError;
use crate::comparator::{
    ComparatorConfig, ComparisonReport, CountState, EngineObservation, NativeTieKey, RankedHit,
    compare_observations,
};
use crate::generator::MAX_DOCUMENT_ID_BYTES;
#[cfg(feature = "tantivy-oracle")]
use crate::runner::SemanticContract;
use crate::version_contract::oracle_version_contract;

const MAX_ORACLE_LIMIT: u64 = 100_000;
const MAX_TIE_EXPANSION: u64 = 100_000;
const MAX_ORACLE_FETCH: u64 = 200_000;
const MAX_CASE_ID_BYTES: usize = 1_024;
const MAX_CASE_QUERY_BYTES: usize = 1024 * 1024;
const MAX_CASE_METADATA_BYTES: usize = 1_024;
const MAX_AST_DIFFERENCES: usize = 1_024;
const MAX_OBSERVATION_TEXT_BYTES: usize = 1024 * 1024;
const MAX_OBSERVATION_AGGREGATE_TEXT_BYTES: usize = 64 * 1024 * 1024;
/// Maximum snippet budget accepted at every harness and campaign boundary.
pub const MAX_SNIPPET_CHARS: u64 = 1_000_000;
pub const TANTIVY_ORACLE_CONFIG_HASH: &str = "shipping-schema-and-parser-v1";

/// Closed engine family used by the cross-engine false-green guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineFamily {
    Quill,
    Tantivy,
}

/// Whether the harness compares separate engines or two variants of one engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonMode {
    CrossEngine,
    InternalDifferential,
}

/// Build identity stamped into every immutable gauntlet object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineDescriptor {
    pub family: EngineFamily,
    pub implementation: String,
    pub crate_version: String,
    pub source_revision: String,
    pub source_dirty: bool,
    pub config_hash: String,
}

impl EngineDescriptor {
    fn validate(&self) -> Result<(), GauntletError> {
        for (label, value) in [
            ("implementation", self.implementation.as_str()),
            ("crate_version", self.crate_version.as_str()),
            ("source_revision", self.source_revision.as_str()),
            ("config_hash", self.config_hash.as_str()),
        ] {
            if value.is_empty()
                || value.len() > 256
                || value.trim() != value
                || value.chars().any(char::is_control)
            {
                return Err(GauntletError::InvalidContract {
                    reason: format!(
                        "engine descriptor {label} must be nonempty, canonical text of at most 256 bytes"
                    ),
                });
            }
        }
        Ok(())
    }

    fn implementation_fingerprint(&self) -> (&str, &str, &str, bool, &str) {
        (
            &self.implementation,
            &self.crate_version,
            &self.source_revision,
            self.source_dirty,
            &self.config_hash,
        )
    }
}

/// Subject/oracle pair with mode-specific distinctness validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnginePairIdentity {
    pub comparison_mode: ComparisonMode,
    pub subject: EngineDescriptor,
    pub oracle: EngineDescriptor,
    /// Shared campaign semantic contract declared by both adapters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_contract: Option<crate::runner::SemanticContract>,
}

impl EnginePairIdentity {
    /// Validate and construct an identity pair before either engine executes.
    ///
    /// # Errors
    ///
    /// Cross-engine comparisons reject the same closed engine family even when
    /// instances or configs differ. Internal differentials allow one family but
    /// require distinct implementation build fingerprints.
    pub fn new(
        comparison_mode: ComparisonMode,
        subject: EngineDescriptor,
        oracle: EngineDescriptor,
    ) -> Result<Self, GauntletError> {
        subject.validate()?;
        oracle.validate()?;
        let invalid = match comparison_mode {
            ComparisonMode::CrossEngine => {
                subject.family != EngineFamily::Quill || oracle.family != EngineFamily::Tantivy
            }
            ComparisonMode::InternalDifferential => {
                subject.family != oracle.family
                    || subject.implementation_fingerprint() == oracle.implementation_fingerprint()
            }
        };
        if invalid {
            return Err(GauntletError::EngineIdentityCollision {
                comparison_mode,
                subject: subject.implementation.clone(),
                oracle: oracle.implementation.clone(),
            });
        }
        Ok(Self {
            comparison_mode,
            subject,
            oracle,
            semantic_contract: None,
        })
    }

    pub(crate) fn bind_semantic_contract(
        &mut self,
        semantic_contract: crate::runner::SemanticContract,
    ) -> Result<(), GauntletError> {
        semantic_contract.validate()?;
        self.semantic_contract = Some(semantic_contract);
        Ok(())
    }

    pub(crate) fn validate_gauntlet_contract(&self) -> Result<(), GauntletError> {
        let mut rebuilt = Self::new(
            self.comparison_mode,
            self.subject.clone(),
            self.oracle.clone(),
        )?;
        if let Some(semantic_contract) = &self.semantic_contract {
            rebuilt.bind_semantic_contract(semantic_contract.clone())?;
        }
        if &rebuilt != self {
            return Err(GauntletError::InvalidContract {
                reason: "engine identity is not self-consistent".to_owned(),
            });
        }
        if self.comparison_mode == ComparisonMode::CrossEngine {
            let oracle_version = oracle_version_contract()?;
            if self.oracle.implementation != "frankensearch-lexical/tantivy-index"
                || self.oracle.crate_version != oracle_version.lexical_package_version
                || self.oracle.source_revision != oracle_version.lexical_git_revision
                || self.oracle.config_hash != TANTIVY_ORACLE_CONFIG_HASH
                || self.oracle.source_dirty
            {
                return Err(GauntletError::InvalidContract {
                    reason: "oracle descriptor does not match the lexical version contract"
                        .to_owned(),
                });
            }
        }
        Ok(())
    }
}

/// Engine-neutral query case consumed by both adapters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DifferentialCase {
    pub fixture_id: String,
    pub query: String,
    pub limit: u64,
    /// Number of ranked matches skipped before the returned page.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub offset: u64,
    pub tie_expansion_limit: u64,
    pub count_requested: bool,
    pub snippet_max_chars: Option<u64>,
    pub metadata: DifferentialCaseMetadata,
}

/// Deterministic fixture-generation inputs allowed in the object hash basis.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DifferentialCaseMetadata {
    pub generator_id: Option<String>,
    pub generator_seed: Option<u64>,
    pub corpus_hash: Option<String>,
}

impl DifferentialCase {
    #[must_use]
    pub fn new(fixture_id: impl Into<String>, query: impl Into<String>, limit: u64) -> Self {
        Self {
            fixture_id: fixture_id.into(),
            query: query.into(),
            limit,
            offset: 0,
            tie_expansion_limit: 256,
            count_requested: true,
            snippet_max_chars: Some(200),
            metadata: DifferentialCaseMetadata::default(),
        }
    }

    pub(crate) fn validate_observations(
        &self,
        engines: &EnginePairIdentity,
        subject: &EngineObservation,
        oracle: &EngineObservation,
    ) -> Result<(), GauntletError> {
        self.validate_shape()?;
        let counts_match_request = if self.count_requested {
            matches!(subject.match_count, CountState::Value(_))
                && matches!(oracle.match_count, CountState::Value(_))
        } else {
            subject.match_count == CountState::NotRequested
                && oracle.match_count == CountState::NotRequested
        };
        if !counts_match_request {
            return Err(GauntletError::InvalidObservation {
                reason: "count evidence does not match the differential case request".to_owned(),
            });
        }
        self.validate_observation_shape("subject", subject)?;
        self.validate_observation_shape("oracle", oracle)?;
        validate_observation_family("subject", engines.subject.family, subject)?;
        validate_observation_family("oracle", engines.oracle.family, oracle)?;
        Ok(())
    }

    fn validate_observation_shape(
        &self,
        label: &str,
        observation: &EngineObservation,
    ) -> Result<(), GauntletError> {
        let observation_text_is_bounded = observation
            .hits
            .iter()
            .chain(&observation.cutoff_tie_group)
            .chain(&observation.offset_tie_group)
            .all(|hit| hit.doc_id.len() <= MAX_DOCUMENT_ID_BYTES)
            && observation.ast_differences.len() <= MAX_AST_DIFFERENCES
            && observation.ast_differences.iter().all(|difference| {
                difference.oracle.len() <= MAX_OBSERVATION_TEXT_BYTES
                    && difference.subject.len() <= MAX_OBSERVATION_TEXT_BYTES
            });
        let aggregate_text_bytes = observation
            .hits
            .iter()
            .chain(&observation.cutoff_tie_group)
            .chain(&observation.offset_tie_group)
            .map(|hit| hit.doc_id.len())
            .chain(
                observation
                    .snippets
                    .iter()
                    .map(|(doc_id, snippet)| doc_id.len().saturating_add(snippet.len())),
            )
            .chain(observation.ast_differences.iter().map(|difference| {
                difference
                    .oracle
                    .len()
                    .saturating_add(difference.subject.len())
            }))
            .try_fold(0_usize, usize::checked_add);
        let snippets_are_bounded = observation.snippets.iter().all(|(doc_id, snippet)| {
            doc_id.len() <= MAX_DOCUMENT_ID_BYTES && snippet.len() <= MAX_OBSERVATION_TEXT_BYTES
        });
        if !observation_text_is_bounded
            || !snippets_are_bounded
            || aggregate_text_bytes.is_none_or(|bytes| bytes > MAX_OBSERVATION_AGGREGATE_TEXT_BYTES)
        {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label} contains oversized result evidence"),
            });
        }
        let hit_count = u64::try_from(observation.hits.len()).map_err(|_| {
            GauntletError::InvalidObservation {
                reason: format!("{label} top-k length does not fit u64"),
            }
        })?;
        let cutoff_count = u64::try_from(observation.cutoff_tie_group.len()).map_err(|_| {
            GauntletError::InvalidObservation {
                reason: format!("{label} cutoff tie-group length does not fit u64"),
            }
        })?;
        let offset_count = u64::try_from(observation.offset_tie_group.len()).map_err(|_| {
            GauntletError::InvalidObservation {
                reason: format!("{label} offset tie-group length does not fit u64"),
            }
        })?;
        let evidence_budget = self
            .offset
            .checked_add(self.limit)
            .and_then(|value| value.checked_add(self.tie_expansion_limit))
            .ok_or_else(|| GauntletError::InvalidObservation {
                reason: format!("{label} evidence budget overflowed"),
            })?;
        if hit_count > self.limit
            || hit_count > observation.doc_count
            || cutoff_count > observation.doc_count
            || offset_count > observation.doc_count
            || cutoff_count > evidence_budget
            || offset_count > evidence_budget
        {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label} result lengths exceed the case or document count"),
            });
        }
        if let CountState::Value(match_count) = observation.match_count
            && (match_count > observation.doc_count
                || hit_count != self.limit.min(match_count.saturating_sub(self.offset))
                || cutoff_count > match_count
                || offset_count > match_count)
        {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label} top-k evidence is inconsistent with its exact count"),
            });
        }
        let observed_ids = observation
            .hits
            .iter()
            .chain(&observation.cutoff_tie_group)
            .chain(&observation.offset_tie_group)
            .map(|hit| hit.doc_id.as_str())
            .collect::<BTreeSet<_>>();
        let observed_id_count =
            u64::try_from(observed_ids.len()).map_err(|_| GauntletError::InvalidObservation {
                reason: format!("{label} observed ID count does not fit u64"),
            })?;
        let exceeds_exact_count = matches!(
            observation.match_count,
            CountState::Value(match_count) if observed_id_count > match_count
        );
        if observed_id_count > observation.doc_count || exceeds_exact_count {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label} observed IDs exceed the exact count evidence"),
            });
        }

        let Some(cutoff) = observation.hits.last() else {
            if observation.cutoff_tie_group.is_empty() && observation.offset_tie_group.is_empty() {
                return Ok(());
            }
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label} has cutoff evidence without any top-k hit"),
            });
        };
        if !observation.cutoff_tie_group.is_empty() {
            let cutoff_score = f32::from_bits(cutoff.score_bits);
            let cutoff_keys = observation
                .cutoff_tie_group
                .iter()
                .map(|hit| (hit.doc_id.as_str(), hit.score_bits))
                .collect::<BTreeSet<_>>();
            let group_is_exact = observation.cutoff_tie_group.iter().all(|hit| {
                f32::from_bits(hit.score_bits)
                    .total_cmp(&cutoff_score)
                    .is_eq()
            }) && observation.hits.iter().all(|hit| {
                !f32::from_bits(hit.score_bits)
                    .total_cmp(&cutoff_score)
                    .is_eq()
                    || cutoff_keys.contains(&(hit.doc_id.as_str(), hit.score_bits))
            });
            if !group_is_exact {
                return Err(GauntletError::InvalidObservation {
                    reason: format!("{label} cutoff tie-group does not describe the top-k cutoff"),
                });
            }
        }
        if !observation.offset_tie_group.is_empty() {
            if self.offset == 0 {
                return Err(GauntletError::InvalidObservation {
                    reason: format!("{label} has offset tie-group evidence for a zero-offset case"),
                });
            }
            let leading = &observation.hits[0];
            let leading_score = f32::from_bits(leading.score_bits);
            let leading_keys = observation
                .offset_tie_group
                .iter()
                .map(|hit| (hit.doc_id.as_str(), hit.score_bits))
                .collect::<BTreeSet<_>>();
            let leading_group_is_exact = observation.offset_tie_group.iter().all(|hit| {
                f32::from_bits(hit.score_bits)
                    .total_cmp(&leading_score)
                    .is_eq()
            }) && observation.hits.iter().all(|hit| {
                !f32::from_bits(hit.score_bits)
                    .total_cmp(&leading_score)
                    .is_eq()
                    || leading_keys.contains(&(hit.doc_id.as_str(), hit.score_bits))
            });
            let page_ids = observation
                .hits
                .iter()
                .map(|hit| hit.doc_id.as_str())
                .collect::<BTreeSet<_>>();
            let proves_skipped_member = observation
                .offset_tie_group
                .iter()
                .any(|hit| !page_ids.contains(hit.doc_id.as_str()));
            if !leading_group_is_exact || !proves_skipped_member {
                return Err(GauntletError::InvalidObservation {
                    reason: format!(
                        "{label} offset tie-group does not describe the page's leading boundary"
                    ),
                });
            }
        }
        Ok(())
    }

    pub(crate) fn validate_shape(&self) -> Result<(), GauntletError> {
        let metadata_is_bounded = [
            self.metadata.generator_id.as_deref(),
            self.metadata.corpus_hash.as_deref(),
        ]
        .into_iter()
        .flatten()
        .all(|value| value.len() <= MAX_CASE_METADATA_BYTES);
        if self.fixture_id.is_empty()
            || self.fixture_id.len() > MAX_CASE_ID_BYTES
            || self.query.len() > MAX_CASE_QUERY_BYTES
            || !metadata_is_bounded
        {
            return Err(GauntletError::InvalidCase {
                reason: "fixture ID, query, or metadata exceed the bounded case contract"
                    .to_owned(),
            });
        }
        let fetch = self
            .offset
            .checked_add(self.limit)
            .and_then(|value| value.checked_add(self.tie_expansion_limit));
        if self.limit > MAX_ORACLE_LIMIT
            || self.offset > MAX_ORACLE_LIMIT
            || self.tie_expansion_limit > MAX_TIE_EXPANSION
            || self
                .snippet_max_chars
                .is_some_and(|value| value > MAX_SNIPPET_CHARS)
            || fetch.is_none_or(|value| value > MAX_ORACLE_FETCH)
        {
            return Err(GauntletError::InvalidCase {
                reason: "top-k, tie expansion, or snippets exceed the bounded oracle budget"
                    .to_owned(),
            });
        }
        Ok(())
    }
}

#[allow(clippy::trivially_copy_pass_by_ref)] // serde skip_serializing_if protocol
const fn is_zero(value: &u64) -> bool {
    *value == 0
}

fn validate_observation_family(
    label: &str,
    family: EngineFamily,
    observation: &EngineObservation,
) -> Result<(), GauntletError> {
    let valid = observation
        .hits
        .iter()
        .chain(&observation.cutoff_tie_group)
        .chain(&observation.offset_tie_group)
        .all(|hit| {
            matches!(
                (family, &hit.native_tie_key),
                (EngineFamily::Quill, NativeTieKey::QuillDocId { .. })
                    | (
                        EngineFamily::Tantivy,
                        NativeTieKey::TantivyDocAddress { .. }
                    )
            )
        });
    if valid {
        Ok(())
    } else {
        Err(GauntletError::InvalidObservation {
            reason: format!("{label} native tie keys do not match its engine family"),
        })
    }
}

/// Future returned by object-safe engine adapters.
pub type GauntletFuture<'a> =
    Pin<Box<dyn Future<Output = Result<EngineObservation, GauntletError>> + Send + 'a>>;

/// Minimal adapter boundary. A real `QuillIndex` can replace the stub unchanged.
pub trait GauntletEngine: Send + Sync {
    fn descriptor(&self) -> EngineDescriptor;

    fn observe<'a>(&'a self, cx: &'a Cx, case: &'a DifferentialCase) -> GauntletFuture<'a>;
}

/// Result of one harness execution before it is encoded as an artifact object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessRun {
    pub engines: EnginePairIdentity,
    pub case: DifferentialCase,
    pub comparator_config: ComparatorConfig,
    pub comparison: ComparisonReport,
}

/// Pure orchestration shell around engine adapters and the comparator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DifferentialHarness {
    comparison_mode: ComparisonMode,
    comparator_config: ComparatorConfig,
}

impl DifferentialHarness {
    #[must_use]
    pub const fn new(comparison_mode: ComparisonMode, comparator_config: ComparatorConfig) -> Self {
        Self {
            comparison_mode,
            comparator_config,
        }
    }

    /// Execute one subject/oracle case.
    ///
    /// Identity validation occurs before either adapter's `observe` method.
    ///
    /// # Errors
    ///
    /// Returns identity, engine, or comparator failures without producing a
    /// false-green report.
    pub async fn run(
        &self,
        cx: &Cx,
        subject: &dyn GauntletEngine,
        oracle: &dyn GauntletEngine,
        case: &DifferentialCase,
    ) -> Result<HarnessRun, GauntletError> {
        let engines = EnginePairIdentity::new(
            self.comparison_mode,
            subject.descriptor(),
            oracle.descriptor(),
        )?;
        self.comparator_config.validate_contract()?;
        case.validate_shape()?;
        let subject_observation = subject.observe(cx, case).await?;
        let oracle_observation = oracle.observe(cx, case).await?;
        case.validate_observations(&engines, &subject_observation, &oracle_observation)?;
        let comparison = compare_observations(
            subject_observation,
            oracle_observation,
            self.comparator_config,
        )?;
        Ok(HarnessRun {
            engines,
            case: case.clone(),
            comparator_config: self.comparator_config,
            comparison,
        })
    }
}

impl Default for DifferentialHarness {
    fn default() -> Self {
        Self::new(ComparisonMode::CrossEngine, ComparatorConfig::default())
    }
}

/// Live scalar Quill subject used by the G1a campaign.
pub struct QuillSubject {
    config: QuillConfig,
    descriptor: EngineDescriptor,
    index: Option<QuillIndex>,
    state: QuillCampaignState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QuillCampaignState {
    Fresh,
    Ingesting,
    Committed,
    Aborted,
}

impl QuillSubject {
    /// Construct a fresh owned-buffer scalar subject.
    ///
    /// # Errors
    ///
    /// Returns a typed Quill configuration/schema failure or invalid engine
    /// descriptor input.
    pub fn in_memory(
        config: QuillConfig,
        source_revision: impl Into<String>,
        source_dirty: bool,
    ) -> Result<Self, GauntletError> {
        let config_hash = quill_config_hash(&config);
        let descriptor = EngineDescriptor {
            family: EngineFamily::Quill,
            implementation: "frankensearch-quill/scalar-index".to_owned(),
            crate_version: env!("CARGO_PKG_VERSION").to_owned(),
            source_revision: source_revision.into(),
            source_dirty,
            config_hash,
        };
        descriptor.validate()?;
        Ok(Self {
            index: Some(QuillIndex::in_memory(config.clone())?),
            config,
            descriptor,
            state: QuillCampaignState::Fresh,
        })
    }

    #[must_use]
    pub const fn config(&self) -> &QuillConfig {
        &self.config
    }

    pub(crate) fn index(&self) -> Result<&QuillIndex, GauntletError> {
        self.index
            .as_ref()
            .ok_or_else(|| GauntletError::SubjectUnavailable {
                reason: "Quill campaign subject was aborted".to_owned(),
            })
    }

    pub(crate) fn index_mut(&mut self) -> Result<&mut QuillIndex, GauntletError> {
        self.index
            .as_mut()
            .ok_or_else(|| GauntletError::SubjectUnavailable {
                reason: "Quill campaign subject was aborted".to_owned(),
            })
    }

    pub(crate) fn claim_fresh_campaign(&mut self) -> Result<(), GauntletError> {
        if self.state != QuillCampaignState::Fresh {
            return Err(GauntletError::InvalidCampaign {
                reason: "Quill subject may execute only one campaign".to_owned(),
            });
        }
        self.state = QuillCampaignState::Ingesting;
        Ok(())
    }

    pub(crate) fn require_ingesting(&self) -> Result<(), GauntletError> {
        if self.state != QuillCampaignState::Ingesting {
            return Err(GauntletError::InvalidCampaign {
                reason: "Quill indexing and commit require an active ingest session".to_owned(),
            });
        }
        Ok(())
    }

    pub(crate) fn mark_committed(&mut self) -> Result<(), GauntletError> {
        self.require_ingesting()?;
        self.state = QuillCampaignState::Committed;
        Ok(())
    }

    pub(crate) fn require_committed(&self) -> Result<(), GauntletError> {
        if self.state != QuillCampaignState::Committed {
            return Err(GauntletError::InvalidCampaign {
                reason: "Quill observation requires a committed campaign snapshot".to_owned(),
            });
        }
        Ok(())
    }

    pub(crate) fn abort(&mut self) {
        self.state = QuillCampaignState::Aborted;
        self.index = None;
    }
}

impl GauntletEngine for QuillSubject {
    fn descriptor(&self) -> EngineDescriptor {
        self.descriptor.clone()
    }

    fn observe<'a>(&'a self, cx: &'a Cx, case: &'a DifferentialCase) -> GauntletFuture<'a> {
        Box::pin(async move {
            self.require_committed()?;
            quill_observe(self.index()?, cx, case)
        })
    }
}

fn quill_observe(
    index: &QuillIndex,
    cx: &Cx,
    case: &DifferentialCase,
) -> Result<EngineObservation, GauntletError> {
    case.validate_shape()?;
    if case.snippet_max_chars.is_some() {
        return Err(GauntletError::InvalidCase {
            reason: "the scalar G1a Quill adapter requires snippets to be disabled".to_owned(),
        });
    }
    let limit = usize::try_from(case.limit).map_err(|_| GauntletError::InvalidCase {
        reason: "limit does not fit usize".to_owned(),
    })?;
    let offset = usize::try_from(case.offset).map_err(|_| GauntletError::InvalidCase {
        reason: "offset does not fit usize".to_owned(),
    })?;
    let tie_expansion =
        usize::try_from(case.tie_expansion_limit).map_err(|_| GauntletError::InvalidCase {
            reason: "tie expansion limit does not fit usize".to_owned(),
        })?;
    let page_end = offset
        .checked_add(limit)
        .ok_or_else(|| GauntletError::InvalidCase {
            reason: "offset plus limit does not fit usize".to_owned(),
        })?;
    let fetch_limit =
        page_end
            .checked_add(tie_expansion)
            .ok_or_else(|| GauntletError::InvalidCase {
                reason: "expanded Quill observation window does not fit usize".to_owned(),
            })?;
    // The first call is the collector mode under test. Keep the expanded
    // exact-count call separate: it exists only to furnish comparator tie
    // evidence and must never stand in for pagination or count-free execution.
    let observed = index.search_paginated(cx, &case.query, limit, offset, case.count_requested)?;
    let evidence = index.search_paginated(cx, &case.query, fetch_limit, 0, true)?;
    quill_observation_from_results(&observed, &evidence, limit, offset, case.count_requested)
}

fn quill_observation_from_results(
    observed: &QuillSearchResult,
    evidence: &QuillSearchResult,
    limit: usize,
    offset: usize,
    count_requested: bool,
) -> Result<EngineObservation, GauntletError> {
    let page_end = offset
        .checked_add(limit)
        .ok_or_else(|| GauntletError::InvalidObservation {
            reason: "Quill observation page boundary does not fit usize".to_owned(),
        })?;
    if observed.doc_count != evidence.doc_count {
        return Err(GauntletError::InvalidObservation {
            reason: "Quill collector modes disagreed on the committed document count".to_owned(),
        });
    }
    if observed.diagnostics != evidence.diagnostics {
        return Err(GauntletError::InvalidObservation {
            reason: "Quill collector modes disagreed on parser diagnostics".to_owned(),
        });
    }
    let expected_start = offset.min(evidence.hits.len());
    let expected_end = page_end.min(evidence.hits.len());
    let expected_page = &evidence.hits[expected_start..expected_end];
    let rank_safe = observed.hits.len() == expected_page.len()
        && observed
            .hits
            .iter()
            .zip(expected_page)
            .all(|(actual, expected)| {
                actual.global_docid == expected.global_docid
                    && actual.document_id == expected.document_id
                    && actual.score.to_bits() == expected.score.to_bits()
            });
    if !rank_safe {
        return Err(GauntletError::InvalidObservation {
            reason: "Quill observed and exhaustive collector pages differ".to_owned(),
        });
    }
    let total_count = evidence
        .total_count
        .ok_or_else(|| GauntletError::InvalidObservation {
            reason: "Quill tie-evidence observation omitted its exact count".to_owned(),
        })?;
    let match_count = match (count_requested, observed.total_count) {
        (true, Some(observed_count)) if observed_count == total_count => {
            CountState::Value(observed_count)
        }
        (true, Some(_)) => {
            return Err(GauntletError::InvalidObservation {
                reason: "Quill counted page disagreed with its expanded tie evidence".to_owned(),
            });
        }
        (true, None) => {
            return Err(GauntletError::InvalidObservation {
                reason: "Quill counted page omitted its exact count".to_owned(),
            });
        }
        (false, None) => CountState::NotRequested,
        (false, Some(_)) => {
            return Err(GauntletError::InvalidObservation {
                reason: "Quill count-free page unexpectedly executed exact-count work".to_owned(),
            });
        }
    };
    let ranked = evidence
        .hits
        .iter()
        .map(|hit| RankedHit {
            doc_id: hit.document_id.clone(),
            score_bits: hit.score.to_bits(),
            native_tie_key: NativeTieKey::QuillDocId {
                doc_id: hit.global_docid,
            },
        })
        .collect::<Vec<_>>();
    let top_len = page_end.min(ranked.len());
    let page_window = &ranked[..top_len];
    let (cutoff_tie_group, cutoff_tie_complete) =
        cutoff_tie_group(&ranked, top_len, total_count, limit > 0 && top_len > offset);
    let (offset_tie_group, offset_tie_complete) = if limit == 0 {
        (Vec::new(), false)
    } else {
        offset_tie_group(
            page_window,
            offset,
            total_count,
            &cutoff_tie_group,
            cutoff_tie_complete,
        )
    };
    let hits = observed
        .hits
        .iter()
        .map(|hit| RankedHit {
            doc_id: hit.document_id.clone(),
            score_bits: hit.score.to_bits(),
            native_tie_key: NativeTieKey::QuillDocId {
                doc_id: hit.global_docid,
            },
        })
        .collect();
    Ok(EngineObservation {
        hits,
        cutoff_tie_group,
        cutoff_tie_complete,
        offset_tie_group,
        offset_tie_complete,
        snippets: BTreeMap::new(),
        match_count,
        doc_count: observed.doc_count,
        ast_differences: Vec::new(),
    })
}

fn cutoff_tie_group(
    hits: &[RankedHit],
    boundary: usize,
    total_count: u64,
    relevant: bool,
) -> (Vec<RankedHit>, bool) {
    if !relevant || boundary == 0 || boundary > hits.len() {
        return (Vec::new(), false);
    }
    let score_bits = hits[boundary - 1].score_bits;
    let group = hits
        .iter()
        .filter(|hit| hit.score_bits == score_bits)
        .cloned()
        .collect::<Vec<_>>();
    let complete = u64::try_from(hits.len()).is_ok_and(|fetched| fetched >= total_count)
        || hits
            .last()
            .is_some_and(|last| last.score_bits != score_bits);
    (group, complete)
}

fn offset_tie_group(
    hits: &[RankedHit],
    offset: usize,
    total_count: u64,
    cutoff_group: &[RankedHit],
    cutoff_complete: bool,
) -> (Vec<RankedHit>, bool) {
    if offset == 0 || offset >= hits.len() {
        return (Vec::new(), false);
    }
    let previous = &hits[offset - 1];
    let leading = &hits[offset];
    if previous.score_bits != leading.score_bits {
        return (Vec::new(), false);
    }
    if cutoff_group
        .first()
        .is_some_and(|hit| hit.score_bits == leading.score_bits)
    {
        return (cutoff_group.to_vec(), cutoff_complete);
    }
    let group = hits
        .iter()
        .filter(|hit| hit.score_bits == leading.score_bits)
        .cloned()
        .collect::<Vec<_>>();
    let complete = hits
        .iter()
        .skip(offset + 1)
        .any(|hit| hit.score_bits != leading.score_bits)
        || u64::try_from(hits.len()).is_ok_and(|fetched| fetched >= total_count);
    (group, complete)
}

fn quill_config_hash(config: &QuillConfig) -> String {
    let canonical = format!(
        "scribe={};delta={};fanout={};tier_small={};tier_medium={};bulk={};bulk_cadence={};compact={:016x};holes={:016x};glob={};fuel={};shards={};deterministic={};visibility_ms={};quarantine={}",
        config.scribe_shard_budget_bytes,
        config.delta_budget_bytes,
        config.tier_fanout,
        config.tier_small_max_docid_width,
        config.tier_medium_max_docid_width,
        config.bulk_load_mode,
        config.bulk_publish_segment_cadence,
        config.compaction_tombstone_density.to_bits(),
        config.merge_max_hole_ratio.to_bits(),
        config.glob_expansion_limit,
        config.query_fuel_budget,
        config.max_ingest_shards,
        config.deterministic_ingest,
        config.max_visibility_lag_ms,
        config.quarantine_on_unrepairable
    );
    format!("{:016x}", xxh3_64(canonical.as_bytes()))
}

/// Tantivy oracle adapter over the shipping lexical implementation.
#[cfg(feature = "tantivy-oracle")]
pub struct TantivyOracle {
    index: frankensearch_lexical::TantivyIndex,
    descriptor: EngineDescriptor,
    semantic_contract: SemanticContract,
    campaign_freshness_verified: bool,
    campaign_state: TantivyCampaignState,
}

#[cfg(feature = "tantivy-oracle")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TantivyCampaignState {
    Fresh,
    Ingesting,
    Committed,
    Aborted,
}

#[cfg(feature = "tantivy-oracle")]
impl TantivyOracle {
    /// Create an in-memory oracle using the shipping schema/parser.
    ///
    /// # Errors
    ///
    /// Returns an error when the embedded version contract or Tantivy index
    /// cannot be initialized.
    pub fn in_memory(
        observed_lexical_revision: &str,
        source_dirty: bool,
    ) -> Result<Self, GauntletError> {
        Self::from_index_with_campaign_freshness(
            frankensearch_lexical::TantivyIndex::in_memory()?,
            observed_lexical_revision,
            source_dirty,
            true,
            SemanticContract::shipping_default(),
        )
    }

    /// Create a fresh in-memory oracle for the snippet-free scalar G1a profile.
    ///
    /// # Errors
    ///
    /// Returns the same provenance or index-construction errors as [`Self::in_memory`].
    pub fn in_memory_scalar_g1a(
        observed_lexical_revision: &str,
        source_dirty: bool,
    ) -> Result<Self, GauntletError> {
        Self::from_index_with_campaign_freshness(
            frankensearch_lexical::TantivyIndex::in_memory_single_threaded_oracle()?,
            observed_lexical_revision,
            source_dirty,
            true,
            SemanticContract::scalar_g1a(),
        )
    }

    /// Wrap an existing shipping Tantivy index.
    ///
    /// # Errors
    ///
    /// Returns an error when the committed oracle version contract is invalid.
    pub fn from_index(
        index: frankensearch_lexical::TantivyIndex,
        observed_lexical_revision: &str,
        source_dirty: bool,
    ) -> Result<Self, GauntletError> {
        Self::from_index_with_campaign_freshness(
            index,
            observed_lexical_revision,
            source_dirty,
            false,
            SemanticContract::shipping_default(),
        )
    }

    fn from_index_with_campaign_freshness(
        index: frankensearch_lexical::TantivyIndex,
        observed_lexical_revision: &str,
        source_dirty: bool,
        campaign_freshness_verified: bool,
        semantic_contract: SemanticContract,
    ) -> Result<Self, GauntletError> {
        let contract = oracle_version_contract()?;
        contract.validate_source_state(observed_lexical_revision, source_dirty)?;
        Ok(Self {
            index,
            descriptor: EngineDescriptor {
                family: EngineFamily::Tantivy,
                implementation: "frankensearch-lexical/tantivy-index".to_owned(),
                crate_version: contract.lexical_package_version,
                source_revision: contract.lexical_git_revision,
                source_dirty,
                config_hash: TANTIVY_ORACLE_CONFIG_HASH.to_owned(),
            },
            semantic_contract,
            campaign_freshness_verified,
            campaign_state: TantivyCampaignState::Fresh,
        })
    }

    pub(crate) const fn campaign_semantic_contract(&self) -> &SemanticContract {
        &self.semantic_contract
    }

    pub(crate) fn claim_fresh_campaign(&mut self) -> Result<(), GauntletError> {
        if !self.campaign_freshness_verified {
            return Err(GauntletError::InvalidContract {
                reason: "Tantivy campaigns require a newly constructed one-shot oracle".to_owned(),
            });
        }
        if self.campaign_state != TantivyCampaignState::Fresh {
            return Err(GauntletError::InvalidCampaign {
                reason: "Tantivy oracle may execute only one campaign".to_owned(),
            });
        }
        self.campaign_state = TantivyCampaignState::Ingesting;
        Ok(())
    }

    pub(crate) fn require_ingesting(&self) -> Result<(), GauntletError> {
        if self.campaign_state != TantivyCampaignState::Ingesting {
            return Err(GauntletError::InvalidCampaign {
                reason: "Tantivy indexing and commit require an active ingest session".to_owned(),
            });
        }
        Ok(())
    }

    pub(crate) fn mark_committed(&mut self) -> Result<(), GauntletError> {
        self.require_ingesting()?;
        self.campaign_state = TantivyCampaignState::Committed;
        Ok(())
    }

    pub(crate) fn require_committed(&self) -> Result<(), GauntletError> {
        if self.campaign_state != TantivyCampaignState::Committed {
            return Err(GauntletError::InvalidCampaign {
                reason: "Tantivy observation requires a committed campaign snapshot".to_owned(),
            });
        }
        Ok(())
    }

    pub(crate) fn abort_campaign(&mut self) {
        self.campaign_state = TantivyCampaignState::Aborted;
        self.campaign_freshness_verified = false;
    }

    /// Index and commit a corpus through the shipping lexical trait.
    ///
    /// # Errors
    ///
    /// Propagates lexical indexing or commit failures.
    pub async fn index_documents(
        &mut self,
        cx: &Cx,
        documents: &[frankensearch_core::IndexableDocument],
    ) -> Result<(), GauntletError> {
        use frankensearch_core::LexicalSearch;

        self.campaign_freshness_verified = false;
        self.index.index_documents(cx, documents).await?;
        self.index.commit(cx).await?;
        Ok(())
    }

    #[must_use]
    pub(crate) const fn index(&self) -> &frankensearch_lexical::TantivyIndex {
        &self.index
    }
}

#[cfg(feature = "tantivy-oracle")]
impl GauntletEngine for TantivyOracle {
    fn descriptor(&self) -> EngineDescriptor {
        self.descriptor.clone()
    }

    fn observe<'a>(&'a self, cx: &'a Cx, case: &'a DifferentialCase) -> GauntletFuture<'a> {
        Box::pin(async move {
            case.validate_shape()?;
            let limit = usize::try_from(case.limit).map_err(|_| GauntletError::InvalidCase {
                reason: "limit does not fit usize".to_owned(),
            })?;
            let offset = usize::try_from(case.offset).map_err(|_| GauntletError::InvalidCase {
                reason: "offset does not fit usize".to_owned(),
            })?;
            let fetch_limit =
                offset
                    .checked_add(limit)
                    .ok_or_else(|| GauntletError::InvalidCase {
                        reason: "offset plus limit does not fit usize".to_owned(),
                    })?;
            let tie_expansion_limit = usize::try_from(case.tie_expansion_limit).map_err(|_| {
                GauntletError::InvalidCase {
                    reason: "tie expansion limit does not fit usize".to_owned(),
                }
            })?;
            let mut snippet_config = frankensearch_lexical::SnippetConfig::default();
            if let Some(max_chars) = case.snippet_max_chars {
                snippet_config.max_chars =
                    usize::try_from(max_chars).map_err(|_| GauntletError::InvalidCase {
                        reason: "snippet character limit does not fit usize".to_owned(),
                    })?;
            }
            let observation = self.index.oracle_observe_query(
                cx,
                &case.query,
                fetch_limit,
                tie_expansion_limit,
                &snippet_config,
            )?;
            let (offset_tie_group, offset_tie_complete) = if offset > 0
                && offset < observation.hits.len()
                && observation
                    .hits
                    .get(offset - 1)
                    .zip(observation.hits.get(offset))
                    .is_some_and(|(previous, first)| {
                        f32::from_bits(previous.score_bits)
                            .total_cmp(&f32::from_bits(first.score_bits))
                            .is_eq()
                    }) {
                let leading_bits = observation.hits[offset].score_bits;
                let same_leading_score = |score_bits| {
                    f32::from_bits(score_bits)
                        .total_cmp(&f32::from_bits(leading_bits))
                        .is_eq()
                };
                let cutoff_is_leading = observation
                    .cutoff_tie_group
                    .first()
                    .is_some_and(|hit| same_leading_score(hit.score_bits));
                if cutoff_is_leading {
                    (
                        observation.cutoff_tie_group.clone(),
                        observation.cutoff_tie_complete,
                    )
                } else {
                    let group = observation
                        .hits
                        .iter()
                        .filter(|hit| same_leading_score(hit.score_bits))
                        .cloned()
                        .collect::<Vec<_>>();
                    let complete = observation
                        .hits
                        .iter()
                        .skip(offset + 1)
                        .any(|hit| !same_leading_score(hit.score_bits))
                        || observation.total_count <= observation.hits.len();
                    (group, complete)
                }
            } else {
                (Vec::new(), false)
            };
            let mut snippets = BTreeMap::new();
            let hits: Vec<RankedHit> = observation
                .hits
                .into_iter()
                .skip(offset)
                .take(limit)
                .map(|hit| {
                    if case.snippet_max_chars.is_some()
                        && let Some(snippet) = hit.snippet
                    {
                        snippets.insert(hit.doc_id.clone(), snippet);
                    }
                    RankedHit {
                        doc_id: hit.doc_id,
                        score_bits: hit.score_bits,
                        native_tie_key: NativeTieKey::TantivyDocAddress {
                            segment_ord: hit.segment_ord,
                            doc_id: hit.segment_doc_id,
                        },
                    }
                })
                .collect();
            let cutoff_tie_group = if hits.is_empty() {
                Vec::new()
            } else {
                observation
                    .cutoff_tie_group
                    .into_iter()
                    .map(|hit| RankedHit {
                        doc_id: hit.doc_id,
                        score_bits: hit.score_bits,
                        native_tie_key: NativeTieKey::TantivyDocAddress {
                            segment_ord: hit.segment_ord,
                            doc_id: hit.segment_doc_id,
                        },
                    })
                    .collect()
            };
            let offset_tie_group = offset_tie_group
                .into_iter()
                .map(|hit| RankedHit {
                    doc_id: hit.doc_id,
                    score_bits: hit.score_bits,
                    native_tie_key: NativeTieKey::TantivyDocAddress {
                        segment_ord: hit.segment_ord,
                        doc_id: hit.segment_doc_id,
                    },
                })
                .collect();
            Ok(EngineObservation {
                hits,
                cutoff_tie_group,
                cutoff_tie_complete: observation.cutoff_tie_complete,
                offset_tie_group,
                offset_tie_complete,
                snippets,
                match_count: if case.count_requested {
                    CountState::Value(u64::try_from(observation.total_count).unwrap_or(u64::MAX))
                } else {
                    CountState::NotRequested
                },
                doc_count: u64::try_from(observation.doc_count).unwrap_or(u64::MAX),
                ast_differences: Vec::new(),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Bound;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use frankensearch_core::DocId;
    use frankensearch_quill::contract::fieldnorm_to_id;
    use frankensearch_quill::delta::{
        DeltaFieldNorm, DeltaNumericValue, DeltaSegment, DeltaSnapshot, DeltaStoredValue,
        DeltaTermPosting,
    };
    use frankensearch_quill::scribe::{DOC_ORDS_PER_LEASE, DeltaFlushInput};
    use frankensearch_quill::{
        Analyzer, CURRENT_ENGINE_VERSION, FieldDescriptor, FieldKind, Query, QueryField,
        QueryValue, SchemaDescriptor,
    };

    use super::*;
    use crate::comparator::{ComparisonStatus, RankClass};

    const E55_ID_FIELD: u16 = 0;
    const E55_CONTENT_FIELD: u16 = 1;
    const E55_TITLE_FIELD: u16 = 2;
    const E55_METADATA_FIELD: u16 = 3;
    const E55_ORD_FIELD: u16 = 4;
    const E55_I64_FIELD: u16 = 5;
    const E55_U64_FIELD: u16 = 6;
    const E55_TAG_FIELD: u16 = 7;
    const E55_HISTORICAL_ID: &str = "sealed-replacement";
    const E55_HISTORICAL_SEGMENT_ID: u64 = 0xe550_0000_0000_0001;
    const E55_FIRST_SEGMENT_ID: u64 = 0xe550_0000_0000_0002;
    const E55_SECOND_SEGMENT_ID: u64 = 0xe550_0000_0000_0003;
    const E55_NIGHTLY_SEED: u64 = 0xe55c_0f0f_5eed_2026;

    const E55_FIELDS: [FieldDescriptor; 8] = [
        FieldDescriptor {
            id: E55_ID_FIELD,
            name: "id",
            kind: FieldKind::Keyword,
            stored: true,
        },
        FieldDescriptor {
            id: E55_CONTENT_FIELD,
            name: "content",
            kind: FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault,
                positions: true,
            },
            stored: true,
        },
        FieldDescriptor {
            id: E55_TITLE_FIELD,
            name: "title",
            kind: FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault,
                positions: true,
            },
            stored: true,
        },
        FieldDescriptor {
            id: E55_METADATA_FIELD,
            name: "metadata_json",
            kind: FieldKind::StoredOnly,
            stored: true,
        },
        FieldDescriptor {
            id: E55_ORD_FIELD,
            name: "ord",
            kind: FieldKind::U64 {
                indexed: false,
                fast: true,
            },
            stored: true,
        },
        FieldDescriptor {
            id: E55_I64_FIELD,
            name: "signed_rank",
            kind: FieldKind::I64 {
                indexed: true,
                fast: true,
            },
            stored: true,
        },
        FieldDescriptor {
            id: E55_U64_FIELD,
            name: "unsigned_rank",
            kind: FieldKind::U64 {
                indexed: true,
                fast: true,
            },
            stored: true,
        },
        FieldDescriptor {
            id: E55_TAG_FIELD,
            name: "tag",
            kind: FieldKind::Keyword,
            stored: true,
        },
    ];

    const E55_SCHEMA: SchemaDescriptor = SchemaDescriptor {
        name: "quill-e55-mixed-residency-v1",
        fields: &E55_FIELDS,
    };

    struct CountingEngine {
        descriptor: EngineDescriptor,
        observe_calls: Arc<AtomicUsize>,
    }

    impl GauntletEngine for CountingEngine {
        fn descriptor(&self) -> EngineDescriptor {
            self.descriptor.clone()
        }

        fn observe<'a>(&'a self, _cx: &'a Cx, _case: &'a DifferentialCase) -> GauntletFuture<'a> {
            Box::pin(async move {
                self.observe_calls.fetch_add(1, Ordering::Relaxed);
                Err(GauntletError::SubjectUnavailable {
                    reason: "counting test engine executed".to_owned(),
                })
            })
        }
    }

    #[derive(Clone, Debug)]
    struct E55Document {
        id: String,
        content: String,
        title: String,
        tag: String,
        signed_rank: i64,
        unsigned_rank: u64,
    }

    impl E55Document {
        fn new(
            id: impl Into<String>,
            content: impl Into<String>,
            title: impl Into<String>,
            tag: impl Into<String>,
            signed_rank: i64,
            unsigned_rank: u64,
        ) -> Self {
            Self {
                id: id.into(),
                content: content.into(),
                title: title.into(),
                tag: tag.into(),
                signed_rank,
                unsigned_rank,
            }
        }
    }

    struct E55OwnedPosting {
        field_ord: u16,
        term: Vec<u8>,
        frequency: u32,
        positions: Option<Vec<u32>>,
    }

    fn e55_text_postings(field_ord: u16, text: &str) -> (u32, Vec<E55OwnedPosting>) {
        let mut positions = BTreeMap::<String, Vec<u32>>::new();
        for (position, term) in text.split_ascii_whitespace().enumerate() {
            positions.entry(term.to_owned()).or_default().push(
                u32::try_from(position).expect("E5.5 fixture token position fits the wire type"),
            );
        }
        let token_count = positions.values().map(Vec::len).sum::<usize>();
        let token_count =
            u32::try_from(token_count).expect("E5.5 fixture token count fits the wire type");
        let postings = positions
            .into_iter()
            .map(|(term, positions)| E55OwnedPosting {
                field_ord,
                frequency: u32::try_from(positions.len())
                    .expect("E5.5 fixture frequency fits the wire type"),
                term: term.into_bytes(),
                positions: Some(positions),
            })
            .collect();
        (token_count, postings)
    }

    fn e55_content_hash(document: &E55Document) -> u64 {
        let mut canonical = Vec::new();
        for value in [
            document.id.as_bytes(),
            document.content.as_bytes(),
            document.title.as_bytes(),
            document.tag.as_bytes(),
        ] {
            canonical.extend_from_slice(&value.len().to_be_bytes());
            canonical.extend_from_slice(value);
        }
        canonical.extend_from_slice(&document.signed_rank.to_be_bytes());
        canonical.extend_from_slice(&document.unsigned_rank.to_be_bytes());
        xxh3_64(&canonical)
    }

    fn e55_apply_document(
        delta: &mut DeltaSegment,
        global_docid: u32,
        document: &E55Document,
    ) -> Option<u32> {
        let (content_length, mut postings) =
            e55_text_postings(E55_CONTENT_FIELD, &document.content);
        let (title_length, title_postings) = e55_text_postings(E55_TITLE_FIELD, &document.title);
        postings.insert(
            0,
            E55OwnedPosting {
                field_ord: E55_ID_FIELD,
                term: document.id.as_bytes().to_vec(),
                frequency: 1,
                positions: None,
            },
        );
        postings.extend(title_postings);
        postings.push(E55OwnedPosting {
            field_ord: E55_TAG_FIELD,
            term: document.tag.as_bytes().to_vec(),
            frequency: 1,
            positions: None,
        });
        postings.sort_by(|left, right| {
            (left.field_ord, left.term.as_slice()).cmp(&(right.field_ord, right.term.as_slice()))
        });
        let borrowed_postings = postings
            .iter()
            .map(|posting| DeltaTermPosting {
                field_ord: posting.field_ord,
                term: &posting.term,
                frequency: posting.frequency,
                positions: posting.positions.as_deref(),
            })
            .collect::<Vec<_>>();
        let fieldnorms = [
            DeltaFieldNorm {
                field_ord: E55_ID_FIELD,
                raw_length: 1,
                fieldnorm_id: fieldnorm_to_id(1),
            },
            DeltaFieldNorm {
                field_ord: E55_CONTENT_FIELD,
                raw_length: content_length,
                fieldnorm_id: fieldnorm_to_id(content_length),
            },
            DeltaFieldNorm {
                field_ord: E55_TITLE_FIELD,
                raw_length: title_length,
                fieldnorm_id: fieldnorm_to_id(title_length),
            },
            DeltaFieldNorm {
                field_ord: E55_TAG_FIELD,
                raw_length: 1,
                fieldnorm_id: fieldnorm_to_id(1),
            },
        ];
        let numeric = [
            DeltaNumericValue::i64(E55_I64_FIELD, document.signed_rank),
            DeltaNumericValue::u64(E55_U64_FIELD, document.unsigned_rank),
        ];
        let ordinal = u64::from(global_docid).to_le_bytes();
        let stored = [
            DeltaStoredValue::new(E55_ID_FIELD, document.id.as_bytes()),
            DeltaStoredValue::new(E55_CONTENT_FIELD, document.content.as_bytes()),
            DeltaStoredValue::new(E55_TITLE_FIELD, document.title.as_bytes()),
            DeltaStoredValue::new(E55_METADATA_FIELD, b"{}"),
            DeltaStoredValue::new(E55_ORD_FIELD, &ordinal),
            DeltaStoredValue::new(E55_TAG_FIELD, document.tag.as_bytes()),
        ];
        delta
            .apply_document_with_values(
                global_docid,
                DocId::from(document.id.as_str()),
                e55_content_hash(document),
                &fieldnorms,
                &borrowed_postings,
                &numeric,
                &stored,
            )
            .expect("apply complete E5.5 Delta document")
            .replaced_delta_docid
    }

    struct E55DeltaBuilder {
        delta: DeltaSegment,
        first_docid: u32,
        next_docid: u32,
        live: BTreeMap<String, (u32, E55Document)>,
    }

    impl E55DeltaBuilder {
        fn new(lease_base: u64) -> Self {
            let first_docid =
                u32::try_from(lease_base).expect("E5.5 Q1 lease base fits global docids");
            Self {
                delta: DeltaSegment::new(E55_SCHEMA, lease_base, usize::MAX / 2)
                    .expect("construct E5.5 Delta"),
                first_docid,
                next_docid: first_docid,
                live: BTreeMap::new(),
            }
        }

        fn add(&mut self, document: E55Document) -> (u32, Option<u32>) {
            let global_docid = self.next_docid;
            self.next_docid = self
                .next_docid
                .checked_add(1)
                .expect("E5.5 fixture stays inside the Q1 domain");
            let expected_replacement = self.live.get(&document.id).map(|(docid, _)| *docid);
            let replaced = e55_apply_document(&mut self.delta, global_docid, &document);
            assert_eq!(replaced, expected_replacement, "Delta upsert witness");
            self.live
                .insert(document.id.clone(), (global_docid, document));
            (global_docid, replaced)
        }

        fn delete(&mut self, document_id: &str) -> u32 {
            let expected = self
                .live
                .remove(document_id)
                .map(|(docid, _)| docid)
                .expect("E5.5 delete names one live Delta row");
            assert_eq!(
                self.delta.delete_delta_id(document_id),
                Some(expected),
                "Delta delete witness"
            );
            expected
        }

        fn freeze(self, keeper_generation: u64) -> E55BuiltDelta {
            let exclusive_end = self.next_docid;
            assert!(exclusive_end > self.first_docid, "E5.5 Delta is nonempty");
            let snapshot = Arc::new(self.delta.freeze(keeper_generation));
            assert!(
                snapshot.is_live_document(self.first_docid),
                "first physical row is a permanent live Q1 anchor"
            );
            assert!(
                snapshot.is_live_document(exclusive_end - 1),
                "last physical row is a permanent live Q1 anchor"
            );
            E55BuiltDelta {
                snapshot,
                q1_range: (u64::from(self.first_docid), u64::from(exclusive_end)),
                live: self.live,
            }
        }
    }

    struct E55BuiltDelta {
        snapshot: Arc<DeltaSnapshot>,
        q1_range: (u64, u64),
        live: BTreeMap<String, (u32, E55Document)>,
    }

    #[derive(Clone)]
    enum E55QueryInput {
        Source(&'static str),
        Preparsed(Query),
    }

    #[derive(Clone)]
    struct E55QueryCase {
        id: &'static str,
        input: E55QueryInput,
    }

    #[derive(Clone, Copy, Debug)]
    enum E55CollectorMode {
        Full,
        Paginated,
        ExactCount,
        ZeroLimit,
        BeyondTotal,
        DocSet,
    }

    impl E55CollectorMode {
        const ALL: [Self; 6] = [
            Self::Full,
            Self::Paginated,
            Self::ExactCount,
            Self::ZeroLimit,
            Self::BeyondTotal,
            Self::DocSet,
        ];

        const fn id(self) -> &'static str {
            match self {
                Self::Full => "full",
                Self::Paginated => "paginated",
                Self::ExactCount => "exact-count",
                Self::ZeroLimit => "zero-limit-exact-count",
                Self::BeyondTotal => "offset-beyond-total",
                Self::DocSet => "docset",
            }
        }
    }

    fn e55_query_cases() -> Vec<E55QueryCase> {
        vec![
            E55QueryCase {
                id: "empty",
                input: E55QueryInput::Source(""),
            },
            E55QueryCase {
                id: "all",
                input: E55QueryInput::Source("*"),
            },
            E55QueryCase {
                id: "term",
                input: E55QueryInput::Source("alpha"),
            },
            E55QueryCase {
                id: "phrase",
                input: E55QueryInput::Source("\"alpha beta\""),
            },
            E55QueryCase {
                id: "boolean",
                input: E55QueryInput::Source("alpha AND beta"),
            },
            E55QueryCase {
                id: "boost-range-i64",
                input: E55QueryInput::Preparsed(Query::Boost {
                    query: Box::new(Query::Range {
                        field_id: E55_I64_FIELD,
                        lower: Bound::Included(QueryValue::I64(-7)),
                        upper: Bound::Excluded(QueryValue::I64(8)),
                    }),
                    factor: 2.5,
                }),
            },
            E55QueryCase {
                id: "range-str",
                input: E55QueryInput::Preparsed(Query::Range {
                    field_id: E55_TAG_FIELD,
                    lower: Bound::Included(QueryValue::Str("blue".to_owned())),
                    upper: Bound::Included(QueryValue::Str("green".to_owned())),
                }),
            },
            E55QueryCase {
                id: "range-i64",
                input: E55QueryInput::Preparsed(Query::Range {
                    field_id: E55_I64_FIELD,
                    lower: Bound::Included(QueryValue::I64(-7)),
                    upper: Bound::Excluded(QueryValue::I64(8)),
                }),
            },
            E55QueryCase {
                id: "range-u64",
                input: E55QueryInput::Preparsed(Query::Range {
                    field_id: E55_U64_FIELD,
                    lower: Bound::Included(QueryValue::U64(2)),
                    upper: Bound::Included(QueryValue::U64(8)),
                }),
            },
            E55QueryCase {
                id: "set-str",
                input: E55QueryInput::Preparsed(Query::Set {
                    field_id: E55_TAG_FIELD,
                    values: vec![
                        QueryValue::Str("blue".to_owned()),
                        QueryValue::Str("red".to_owned()),
                    ],
                }),
            },
            E55QueryCase {
                id: "set-i64",
                input: E55QueryInput::Preparsed(Query::Set {
                    field_id: E55_I64_FIELD,
                    values: vec![QueryValue::I64(-7), QueryValue::I64(9)],
                }),
            },
            E55QueryCase {
                id: "set-u64",
                input: E55QueryInput::Preparsed(Query::Set {
                    field_id: E55_U64_FIELD,
                    values: vec![QueryValue::U64(2), QueryValue::U64(13)],
                }),
            },
            E55QueryCase {
                id: "glob",
                input: E55QueryInput::Preparsed(Query::Glob {
                    field_ids: vec![E55_CONTENT_FIELD, E55_TITLE_FIELD],
                    pattern: "*lpha*".to_owned(),
                }),
            },
        ]
    }

    #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
    struct E55FieldStatsWitness {
        field_ord: u16,
        total_tokens: u64,
        doc_count: u64,
    }

    #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
    struct E55TermDfWitness {
        field_ord: u16,
        term: String,
        doc_freq: u64,
    }

    #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
    struct E55StatsWitness {
        bm25_doc_count: u64,
        live_doc_count: u64,
        fields: Vec<E55FieldStatsWitness>,
        term_doc_freqs: Vec<E55TermDfWitness>,
    }

    #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
    struct E55ResidencyShape {
        baseline_dead_keeper_segments: usize,
        new_keeper_segments: usize,
        delta_leaves: usize,
        live_leaf_ranges: Vec<(u64, u64)>,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct E410EdgeStateShape {
        keeper_segments: usize,
        keeper_at_seal_documents: u64,
        keeper_tombstones: u64,
        delta_leaves: usize,
        delta_physical_documents: usize,
        delta_live_documents: usize,
        live_documents: u64,
        tombstoned_docid: Option<u32>,
    }

    #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
    struct E55CaseEvidence {
        diagnostics: Vec<String>,
        observation: EngineObservation,
    }

    #[derive(Clone, Debug, Serialize)]
    struct E55ResidencyEvidence {
        seed: String,
        corpus_hash: String,
        extras_per_delta: usize,
        state: &'static str,
        shape: E55ResidencyShape,
        stats: E55StatsWitness,
        cases: BTreeMap<String, E55CaseEvidence>,
    }

    fn e55_differential_case(
        fixture_id: String,
        query_text: String,
        mode: E55CollectorMode,
        live_doc_count: u64,
        seed: u64,
        corpus_hash: u64,
    ) -> DifferentialCase {
        let (limit, offset, count_requested) = match mode {
            E55CollectorMode::Full => (live_doc_count, 0, false),
            E55CollectorMode::Paginated => (2, 1, false),
            E55CollectorMode::ExactCount => (2, 0, true),
            E55CollectorMode::ZeroLimit => (0, 0, true),
            E55CollectorMode::BeyondTotal => (2, live_doc_count.saturating_add(5), false),
            E55CollectorMode::DocSet => unreachable!("docset has no ranked case"),
        };
        DifferentialCase {
            fixture_id,
            query: query_text,
            limit,
            offset,
            tie_expansion_limit: live_doc_count.saturating_add(8),
            count_requested,
            snippet_max_chars: None,
            metadata: DifferentialCaseMetadata {
                generator_id: Some("quill-e55-mixed-residency-v1".to_owned()),
                generator_seed: Some(seed),
                corpus_hash: Some(format!("{corpus_hash:016x}")),
            },
        }
    }

    fn e55_ranked_evidence(
        index: &QuillIndex,
        cx: &Cx,
        query: &E55QueryCase,
        mode: E55CollectorMode,
        seed: u64,
        corpus_hash: u64,
    ) -> E55CaseEvidence {
        let snapshot = index.search_snapshot();
        let query_text = match &query.input {
            E55QueryInput::Source(source) => (*source).to_owned(),
            E55QueryInput::Preparsed(_) => format!("<preparsed:{}>", query.id),
        };
        let case = e55_differential_case(
            format!("e55-{}-{}", query.id, mode.id()),
            query_text,
            mode,
            snapshot.live_doc_count(),
            seed,
            corpus_hash,
        );
        case.validate_shape().expect("valid bounded E5.5 case");
        let limit = usize::try_from(case.limit).expect("E5.5 limit fits usize");
        let offset = usize::try_from(case.offset).expect("E5.5 offset fits usize");
        let tie_expansion =
            usize::try_from(case.tie_expansion_limit).expect("E5.5 tie expansion fits usize");
        let fetch_limit = offset
            .checked_add(limit)
            .and_then(|value| value.checked_add(tie_expansion))
            .expect("E5.5 evidence window fits usize");
        let (observed, evidence) = match &query.input {
            E55QueryInput::Source(source) => (
                index
                    .search_paginated(cx, source, limit, offset, case.count_requested)
                    .expect("execute E5.5 source collector"),
                index
                    .search_paginated(cx, source, fetch_limit, 0, true)
                    .expect("execute E5.5 source evidence collector"),
            ),
            E55QueryInput::Preparsed(parsed) => (
                index
                    .search_preparsed_paginated(cx, parsed, limit, offset, case.count_requested)
                    .expect("execute E5.5 preparsed collector"),
                index
                    .search_preparsed_paginated(cx, parsed, fetch_limit, 0, true)
                    .expect("execute E5.5 preparsed evidence collector"),
            ),
        };
        let diagnostics = observed
            .diagnostics
            .iter()
            .map(|diagnostic| format!("{diagnostic:?}"))
            .collect();
        let observation = quill_observation_from_results(
            &observed,
            &evidence,
            limit,
            offset,
            case.count_requested,
        )
        .expect("assemble E5.5 ranked observation");
        match mode {
            E55CollectorMode::ZeroLimit => {
                assert!(observation.hits.is_empty(), "limit=0 returns no hits");
                assert!(
                    matches!(observation.match_count, CountState::Value(_)),
                    "limit=0 retains exact-count evidence"
                );
            }
            E55CollectorMode::BeyondTotal => {
                assert!(
                    observation.hits.is_empty(),
                    "offset beyond total returns an empty page"
                );
                assert_eq!(
                    observation.match_count,
                    CountState::NotRequested,
                    "count-free beyond-total page does not expose count work"
                );
            }
            E55CollectorMode::Full
            | E55CollectorMode::Paginated
            | E55CollectorMode::ExactCount
            | E55CollectorMode::DocSet => {}
        }
        E55CaseEvidence {
            diagnostics,
            observation,
        }
    }

    fn e55_docset_evidence(index: &QuillIndex, cx: &Cx, query: &E55QueryCase) -> E55CaseEvidence {
        let (docids, diagnostics) = match &query.input {
            E55QueryInput::Source(source) => {
                let docids = index
                    .collect_docids(cx, source)
                    .expect("execute E5.5 source docset collector");
                let diagnostics = index
                    .search_paginated(cx, source, 0, 0, true)
                    .expect("collect E5.5 source diagnostic witness")
                    .diagnostics
                    .into_iter()
                    .map(|diagnostic| format!("{diagnostic:?}"))
                    .collect();
                (docids, diagnostics)
            }
            E55QueryInput::Preparsed(parsed) => (
                index
                    .collect_preparsed_docids(cx, parsed)
                    .expect("execute E5.5 preparsed docset collector"),
                Vec::new(),
            ),
        };
        assert!(
            docids.windows(2).all(|window| window[0] < window[1]),
            "E5.5 docset is sorted and unique"
        );
        let snapshot = index.search_snapshot();
        let hits = docids
            .into_iter()
            .map(|global_docid| RankedHit {
                doc_id: snapshot
                    .materialize_document_id(global_docid)
                    .expect("E5.5 docset winner materializes")
                    .to_string(),
                score_bits: 0.0_f32.to_bits(),
                native_tie_key: NativeTieKey::QuillDocId {
                    doc_id: global_docid,
                },
            })
            .collect::<Vec<_>>();
        let match_count =
            u64::try_from(hits.len()).expect("E5.5 docset count fits the observation wire type");
        E55CaseEvidence {
            diagnostics,
            observation: EngineObservation {
                cutoff_tie_group: hits.clone(),
                cutoff_tie_complete: true,
                hits,
                offset_tie_group: Vec::new(),
                offset_tie_complete: false,
                snippets: BTreeMap::new(),
                match_count: CountState::Value(match_count),
                doc_count: snapshot.live_doc_count(),
                ast_differences: Vec::new(),
            },
        }
    }

    fn e55_config() -> QuillConfig {
        QuillConfig {
            deterministic_ingest: true,
            glob_expansion_limit: 4_096,
            ..QuillConfig::default()
        }
    }

    fn e55_flush_input(segment_id: u64) -> DeltaFlushInput {
        DeltaFlushInput {
            segment_id,
            created_unix_s: 0,
            engine_version: CURRENT_ENGINE_VERSION,
        }
    }

    async fn e55_index_with_live_history(cx: &Cx) -> (QuillIndex, usize, u32) {
        let config = e55_config();
        let index = QuillIndex::in_memory_with_schema(E55_SCHEMA, config)
            .expect("construct historical E5.5 index");
        let generation = index.search_snapshot().keeper_generation();
        let mut historical = E55DeltaBuilder::new(0);
        let (historical_docid, replaced) = historical.add(E55Document::new(
            E55_HISTORICAL_ID,
            "historicalonly alpha beta",
            "historicalonly title",
            "blue",
            -7,
            2,
        ));
        assert!(replaced.is_none());
        let historical = historical.freeze(generation);
        index
            .publish_delta_table(vec![Arc::clone(&historical.snapshot)])
            .expect("publish historical E5.5 Delta");
        index
            .seal_delta_snapshot(
                cx,
                historical.snapshot,
                Vec::new(),
                e55_flush_input(E55_HISTORICAL_SEGMENT_ID),
            )
            .await
            .expect("seal historical E5.5 Delta");

        assert_eq!(
            index
                .search_snapshot()
                .materialize_document_id(historical_docid)
                .as_deref(),
            Some(E55_HISTORICAL_ID),
            "the sealed upsert source remains live until its replacement is staged"
        );
        let baseline_history_segments = index.snapshot().segments().len();
        (index, baseline_history_segments, historical_docid)
    }

    fn e55_tombstone_sealed_upsert_source(index: &QuillIndex, historical_docid: u32) -> QuillIndex {
        let committed = index.snapshot().clone();
        assert_eq!(
            committed
                .materialize_document_id(historical_docid)
                .as_deref(),
            Some(E55_HISTORICAL_ID),
            "sealed upsert begins from a live Keeper row"
        );
        let mut tombstoned_manifest = committed
            .next_manifest()
            .expect("stage sealed-upsert tombstone generation");
        assert!(
            committed
                .delete_document(&mut tombstoned_manifest, E55_HISTORICAL_ID)
                .expect("stage sealed-upsert source tombstone")
        );
        let tombstoned = committed
            .publish_owned_segments(&tombstoned_manifest, Vec::new())
            .expect("publish sealed-upsert source tombstone");
        assert_eq!(
            tombstoned.materialize_document_id(historical_docid),
            None,
            "sealed-upsert source is physically retained but publicly retired"
        );
        QuillIndex::from_in_memory_snapshot(tombstoned, e55_config())
            .expect("bind sealed-upsert Keeper successor")
    }

    fn e55_next_random(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    fn e55_add_seeded_documents(
        builder: &mut E55DeltaBuilder,
        seed: &mut u64,
        shard: usize,
        count: usize,
    ) {
        const TAGS: [&str; 6] = ["amber", "blue", "cyan", "green", "red", "violet"];
        const CONTENTS: [&str; 6] = [
            "alpha beta seeded",
            "alpha gamma seeded",
            "beta delta seeded",
            "gamma omega seeded",
            "alpha beta gamma seeded",
            "omega violet seeded",
        ];
        for ordinal in 0..count {
            let random = e55_next_random(seed);
            let tag = TAGS
                [usize::try_from(random % TAGS.len() as u64).expect("seeded tag index fits usize")];
            let content = CONTENTS[usize::try_from((random >> 8) % CONTENTS.len() as u64)
                .expect("seeded content index fits usize")];
            let signed_rank =
                i64::try_from((random >> 16) % 61).expect("seeded signed rank fits i64") - 30;
            let unsigned_rank = (random >> 24) % 41;
            builder.add(E55Document::new(
                format!("seeded-{shard}-{ordinal:05}"),
                content,
                format!("seeded title {}", random % 7),
                tag,
                signed_rank,
                unsigned_rank,
            ));
        }
    }

    struct E55LiveCorpus {
        first: Arc<DeltaSnapshot>,
        second: Arc<DeltaSnapshot>,
        expected_ranges: Vec<(u64, u64)>,
        expected_live: BTreeMap<String, u32>,
        retired_docids: Vec<u32>,
        replacement_docid: u32,
        corpus_hash: u64,
    }

    fn e55_corpus_hash(live: &BTreeMap<String, (u32, E55Document)>) -> u64 {
        let mut canonical = Vec::new();
        for (id, (global_docid, document)) in live {
            for value in [
                id.as_bytes(),
                document.content.as_bytes(),
                document.title.as_bytes(),
                document.tag.as_bytes(),
            ] {
                canonical.extend_from_slice(
                    &u64::try_from(value.len())
                        .expect("E5.5 fixture value length fits u64")
                        .to_be_bytes(),
                );
                canonical.extend_from_slice(value);
            }
            canonical.extend_from_slice(&global_docid.to_be_bytes());
            canonical.extend_from_slice(&document.signed_rank.to_be_bytes());
            canonical.extend_from_slice(&document.unsigned_rank.to_be_bytes());
        }
        xxh3_64(&canonical)
    }

    fn e55_build_live_corpus(
        index: &QuillIndex,
        seed: u64,
        extras_per_delta: usize,
        historical_docid: u32,
    ) -> E55LiveCorpus {
        let lease_base = index
            .snapshot()
            .loaded_manifest()
            .manifest
            .docid_high_watermark;
        assert_eq!(
            lease_base % u64::from(DOC_ORDS_PER_LEASE),
            0,
            "historical Keeper preserves a Q1-aligned successor lease"
        );
        let second_lease_base = lease_base
            .checked_add(u64::from(DOC_ORDS_PER_LEASE))
            .expect("second E5.5 lease base fits u64");
        assert!(
            extras_per_delta + 16 < DOC_ORDS_PER_LEASE as usize,
            "seeded E5.5 fixture stays within each Q1 lease"
        );
        let generation = index.search_snapshot().keeper_generation();
        let mut random = seed;

        let mut first = E55DeltaBuilder::new(lease_base);
        first.add(E55Document::new(
            "anchor-0-first",
            "alpha beta anchor",
            "alpha beta first",
            "amber",
            -20,
            0,
        ));
        assert_eq!(
            index
                .search_snapshot()
                .materialize_document_id(historical_docid)
                .as_deref(),
            Some(E55_HISTORICAL_ID),
            "the sealed row is live when its Delta replacement is staged"
        );
        let (replacement_docid, replaced) = first.add(E55Document::new(
            E55_HISTORICAL_ID,
            "alpha beta replacementlive",
            "replacementlive alpha",
            "blue",
            -7,
            2,
        ));
        assert!(
            replaced.is_none(),
            "cross-residency replacement has no older row in its new Delta"
        );
        assert_eq!(
            index
                .search_snapshot()
                .materialize_document_id(historical_docid)
                .as_deref(),
            Some(E55_HISTORICAL_ID),
            "staging the replacement does not retire the live sealed source early"
        );
        let (upsert_old, replaced) = first.add(E55Document::new(
            "delta-upsert",
            "abandoned alpha",
            "abandoned",
            "red",
            9,
            13,
        ));
        assert!(replaced.is_none());
        let (_, replaced) = first.add(E55Document::new(
            "delta-upsert",
            "alpha gamma upsertlive",
            "upsertlive alpha",
            "red",
            9,
            13,
        ));
        assert_eq!(replaced, Some(upsert_old));
        first.add(E55Document::new(
            "delta-delete-readd",
            "abandoned beta",
            "abandoned",
            "green",
            3,
            8,
        ));
        let delete_old = first.delete("delta-delete-readd");
        first.add(E55Document::new(
            "delta-delete-readd",
            "alpha beta readdlive",
            "readdlive beta",
            "green",
            3,
            8,
        ));
        first.add(E55Document::new(
            "range-middle",
            "beta range middle",
            "range alpha",
            "cyan",
            0,
            5,
        ));
        e55_add_seeded_documents(&mut first, &mut random, 0, extras_per_delta);
        first.add(E55Document::new(
            "anchor-0-last",
            "omega anchor",
            "omega final",
            "violet",
            20,
            21,
        ));
        let first = first.freeze(generation);

        let mut second = E55DeltaBuilder::new(second_lease_base);
        second.add(E55Document::new(
            "anchor-1-first",
            "alpha beta anchor",
            "alpha beta second",
            "amber",
            -11,
            1,
        ));
        second.add(E55Document::new(
            "second-blue",
            "alpha beta blue",
            "blue alpha",
            "blue",
            -7,
            2,
        ));
        second.add(E55Document::new(
            "second-green",
            "beta gamma green",
            "green beta",
            "green",
            7,
            8,
        ));
        second.add(E55Document::new(
            "second-red",
            "alpha delta red",
            "red alpha",
            "red",
            9,
            13,
        ));
        second.add(E55Document::new(
            "second-yellow",
            "omega yellow",
            "yellow omega",
            "yellow",
            30,
            34,
        ));
        e55_add_seeded_documents(&mut second, &mut random, 1, extras_per_delta);
        second.add(E55Document::new(
            "anchor-1-last",
            "alpha omega anchor",
            "omega final",
            "violet",
            25,
            40,
        ));
        let second = second.freeze(generation);

        let mut live = first.live.clone();
        for (id, row) in &second.live {
            assert!(live.insert(id.clone(), row.clone()).is_none());
        }
        let corpus_hash = e55_corpus_hash(&live);
        let expected_live = live
            .iter()
            .map(|(id, (global_docid, _))| (id.clone(), *global_docid))
            .collect();
        E55LiveCorpus {
            expected_ranges: vec![first.q1_range, second.q1_range],
            first: first.snapshot,
            second: second.snapshot,
            expected_live,
            retired_docids: vec![upsert_old, delete_old],
            replacement_docid,
            corpus_hash,
        }
    }

    fn e55_stats_witness(index: &QuillIndex) -> E55StatsWitness {
        let snapshot = index.search_snapshot();
        let fields = [
            E55_ID_FIELD,
            E55_CONTENT_FIELD,
            E55_TITLE_FIELD,
            E55_TAG_FIELD,
        ]
        .into_iter()
        .map(|field_ord| {
            let stats = snapshot
                .bm25_field_stats(field_ord)
                .expect("E5.5 indexed string field has composite statistics");
            E55FieldStatsWitness {
                field_ord,
                total_tokens: stats.total_tokens,
                doc_count: stats.doc_count,
            }
        })
        .collect();
        let terms: [(u16, &[u8]); 7] = [
            (E55_ID_FIELD, E55_HISTORICAL_ID.as_bytes()),
            (E55_CONTENT_FIELD, b"historicalonly"),
            (E55_CONTENT_FIELD, b"alpha"),
            (E55_CONTENT_FIELD, b"beta"),
            (E55_CONTENT_FIELD, b"abandoned"),
            (E55_TITLE_FIELD, b"alpha"),
            (E55_TAG_FIELD, b"blue"),
        ];
        let term_doc_freqs = terms
            .into_iter()
            .map(|(field_ord, term)| E55TermDfWitness {
                field_ord,
                term: String::from_utf8(term.to_vec()).expect("E5.5 witness terms are UTF-8"),
                doc_freq: snapshot
                    .bm25_doc_freq(field_ord, term)
                    .expect("collect E5.5 snapshot document frequency"),
            })
            .collect();
        E55StatsWitness {
            bm25_doc_count: snapshot.bm25_doc_count(),
            live_doc_count: snapshot.live_doc_count(),
            fields,
            term_doc_freqs,
        }
    }

    fn e55_live_leaf_ranges(index: &QuillIndex, baseline_dead_segments: usize) -> Vec<(u64, u64)> {
        let snapshot = index.search_snapshot();
        let mut ranges = snapshot
            .keeper_snapshot()
            .segments()
            .iter()
            .skip(baseline_dead_segments)
            .map(|segment| {
                let manifest = segment.manifest();
                (manifest.docid_lo, manifest.docid_hi)
            })
            .collect::<Vec<_>>();
        for delta in snapshot.delta_snapshots() {
            let live_docids = delta
                .live_documents()
                .map(|(global_docid, _)| global_docid)
                .collect::<Vec<_>>();
            let first = *live_docids
                .first()
                .expect("E5.5 published Delta has a live first anchor");
            let last = *live_docids
                .last()
                .expect("E5.5 published Delta has a live last anchor");
            ranges.push((u64::from(first), u64::from(last) + 1));
        }
        ranges.sort_unstable();
        ranges
    }

    fn e55_assert_identity_overlay(
        index: &QuillIndex,
        cx: &Cx,
        expected_live: &BTreeMap<String, u32>,
        retired_docids: &[u32],
        replacement_docid: u32,
    ) {
        let snapshot = index.search_snapshot();
        for (document_id, &global_docid) in expected_live {
            assert_eq!(
                snapshot
                    .materialize_document_id(global_docid)
                    .map(|value| value.to_string()),
                Some(document_id.clone()),
                "live Q1 materialization drifted for {document_id}"
            );
        }
        for &retired_docid in retired_docids {
            assert_eq!(
                snapshot.materialize_document_id(retired_docid),
                None,
                "retired Q1 row {retired_docid} became visible"
            );
        }
        let replacement_query = Query::Term {
            fields: vec![QueryField::new(E55_ID_FIELD, 1.0)],
            text: E55_HISTORICAL_ID.to_owned(),
        };
        assert_eq!(
            index
                .collect_preparsed_docids(cx, &replacement_query)
                .expect("resolve sealed-history replacement by external ID"),
            vec![replacement_docid],
            "the tombstoned Keeper row must not mask or duplicate its live replacement"
        );
    }

    struct E55CaptureContext<'a> {
        baseline_dead_segments: usize,
        expected_ranges: &'a [(u64, u64)],
        expected_live: &'a BTreeMap<String, u32>,
        retired_docids: &'a [u32],
        replacement_docid: u32,
        seed: u64,
        corpus_hash: u64,
        extras_per_delta: usize,
    }

    fn e55_capture_residency(
        index: &QuillIndex,
        cx: &Cx,
        state: &'static str,
        expected_new_keeper_segments: usize,
        expected_delta_leaves: usize,
        context: &E55CaptureContext<'_>,
    ) -> E55ResidencyEvidence {
        let snapshot = index.search_snapshot();
        let raw_keeper_segments = snapshot.keeper_snapshot().segments().len();
        let new_keeper_segments = raw_keeper_segments
            .checked_sub(context.baseline_dead_segments)
            .expect("E5.5 baseline segment count cannot exceed the current Keeper");
        let shape = E55ResidencyShape {
            baseline_dead_keeper_segments: context.baseline_dead_segments,
            new_keeper_segments,
            delta_leaves: snapshot.delta_count(),
            live_leaf_ranges: e55_live_leaf_ranges(index, context.baseline_dead_segments),
        };
        assert_eq!(
            shape.new_keeper_segments, expected_new_keeper_segments,
            "E5.5 {state} new Keeper residency shape"
        );
        assert_eq!(
            shape.delta_leaves, expected_delta_leaves,
            "E5.5 {state} Delta residency shape"
        );
        assert_eq!(
            shape.live_leaf_ranges, context.expected_ranges,
            "E5.5 {state} preserves the exact two-leaf Q1 geometry"
        );
        e55_assert_identity_overlay(
            index,
            cx,
            context.expected_live,
            context.retired_docids,
            context.replacement_docid,
        );

        let mut cases = BTreeMap::new();
        for query in e55_query_cases() {
            for mode in E55CollectorMode::ALL {
                let evidence = match mode {
                    E55CollectorMode::DocSet => e55_docset_evidence(index, cx, &query),
                    E55CollectorMode::Full
                    | E55CollectorMode::Paginated
                    | E55CollectorMode::ExactCount
                    | E55CollectorMode::ZeroLimit
                    | E55CollectorMode::BeyondTotal => e55_ranked_evidence(
                        index,
                        cx,
                        &query,
                        mode,
                        context.seed,
                        context.corpus_hash,
                    ),
                };
                let case_id = format!("{}::{}", query.id, mode.id());
                assert!(
                    cases.insert(case_id.clone(), evidence).is_none(),
                    "duplicate E5.5 matrix case {case_id}"
                );
            }
        }
        E55ResidencyEvidence {
            seed: format!("0x{:016x}", context.seed),
            corpus_hash: format!("{:016x}", context.corpus_hash),
            extras_per_delta: context.extras_per_delta,
            state,
            shape,
            stats: e55_stats_witness(index),
            cases,
        }
    }

    fn e410_capture_edge_state(
        index: &QuillIndex,
        cx: &Cx,
        state: &'static str,
        expected: E410EdgeStateShape,
    ) -> BTreeMap<String, E55CaseEvidence> {
        let snapshot = index.search_snapshot();
        let keeper = snapshot.keeper_snapshot();
        assert_eq!(
            keeper.segments().len(),
            expected.keeper_segments,
            "E4.10 {state} Keeper segment count",
        );
        assert_eq!(
            keeper.at_seal_doc_count(),
            expected.keeper_at_seal_documents,
            "E4.10 {state} Keeper physical row count",
        );
        assert_eq!(
            keeper.tombstone_count(),
            expected.keeper_tombstones,
            "E4.10 {state} Keeper tombstone count",
        );
        assert_eq!(
            snapshot.delta_count(),
            expected.delta_leaves,
            "E4.10 {state} Delta leaf count",
        );
        assert_eq!(
            snapshot
                .delta_snapshots()
                .iter()
                .map(|delta| delta.segment().physical_document_count())
                .sum::<usize>(),
            expected.delta_physical_documents,
            "E4.10 {state} Delta physical row count",
        );
        assert_eq!(
            snapshot
                .delta_snapshots()
                .iter()
                .map(|delta| delta.live_document_count())
                .sum::<usize>(),
            expected.delta_live_documents,
            "E4.10 {state} Delta live row count",
        );
        assert_eq!(
            snapshot.bm25_doc_count(),
            expected.keeper_at_seal_documents.saturating_add(
                u64::try_from(expected.delta_live_documents)
                    .expect("E4.10 Delta live row count fits u64"),
            ),
            "E4.10 {state} BM25 lifecycle population",
        );
        assert_eq!(
            snapshot.live_doc_count(),
            expected.live_documents,
            "E4.10 {state} live document count",
        );
        if let Some(global_docid) = expected.tombstoned_docid {
            assert!(
                keeper
                    .segments()
                    .iter()
                    .any(|segment| segment.is_tombstoned(global_docid)),
                "E4.10 {state} must physically retain tombstoned docid {global_docid}",
            );
        }
        let mut cases = BTreeMap::new();
        for query in e55_query_cases() {
            for mode in E55CollectorMode::ALL {
                let evidence = match mode {
                    E55CollectorMode::DocSet => e55_docset_evidence(index, cx, &query),
                    E55CollectorMode::Full
                    | E55CollectorMode::Paginated
                    | E55CollectorMode::ExactCount
                    | E55CollectorMode::ZeroLimit
                    | E55CollectorMode::BeyondTotal => e55_ranked_evidence(
                        index,
                        cx,
                        &query,
                        mode,
                        0xe410,
                        xxh3_64(state.as_bytes()),
                    ),
                };
                let result_count = evidence.observation.hits.len();
                tracing::info!(
                    target: "frankensearch.quill.gauntlet.e410",
                    state,
                    query_id = query.id,
                    collector = mode.id(),
                    expected_doc_count = expected.live_documents,
                    result_count,
                    "completed E4.10 edge-state query case",
                );
                assert_eq!(
                    evidence.observation.doc_count,
                    expected.live_documents,
                    "E4.10 {state} query={} collector={} doc_count",
                    query.id,
                    mode.id(),
                );
                let expected_matches =
                    u64::from(expected.live_documents == 1 && query.id != "empty");
                let expected_matching_hits =
                    usize::try_from(expected_matches).expect("E4.10 match count fits usize");
                let expected_hits = match mode {
                    E55CollectorMode::Full
                    | E55CollectorMode::ExactCount
                    | E55CollectorMode::DocSet => expected_matching_hits,
                    E55CollectorMode::Paginated
                    | E55CollectorMode::ZeroLimit
                    | E55CollectorMode::BeyondTotal => 0,
                };
                assert_eq!(
                    evidence.observation.hits.len(),
                    expected_hits,
                    "E4.10 {state} query={} collector={} result cardinality",
                    query.id,
                    mode.id(),
                );
                if expected_hits == 1 {
                    assert_eq!(
                        evidence.observation.hits[0].doc_id,
                        E55_HISTORICAL_ID,
                        "E4.10 {state} query={} collector={} returned the wrong row",
                        query.id,
                        mode.id(),
                    );
                }
                let expected_count = if matches!(
                    mode,
                    E55CollectorMode::ExactCount
                        | E55CollectorMode::ZeroLimit
                        | E55CollectorMode::DocSet
                ) {
                    CountState::Value(expected_matches)
                } else {
                    CountState::NotRequested
                };
                assert_eq!(
                    evidence.observation.match_count,
                    expected_count,
                    "E4.10 {state} query={} collector={} count evidence",
                    query.id,
                    mode.id(),
                );
                let case_id = format!("{}::{}", query.id, mode.id());
                assert!(
                    cases.insert(case_id.clone(), evidence).is_none(),
                    "duplicate E4.10 edge-state case {case_id}",
                );
            }
        }
        cases
    }

    fn e410_delta_only_index() -> QuillIndex {
        let index = QuillIndex::in_memory_with_schema(E55_SCHEMA, e55_config())
            .expect("construct strict E4.10 Delta-only index");
        let generation = index.search_snapshot().keeper_generation();
        let mut delta = E55DeltaBuilder::new(0);
        let (_, replaced) = delta.add(E55Document::new(
            E55_HISTORICAL_ID,
            "historicalonly alpha beta",
            "historicalonly title",
            "blue",
            -7,
            2,
        ));
        assert!(replaced.is_none());
        let delta = delta.freeze(generation);
        index
            .publish_delta_table(vec![delta.snapshot])
            .expect("publish strict E4.10 Delta-only snapshot");
        index
    }

    fn e55_first_native_key_divergence(
        subject: &EngineObservation,
        oracle: &EngineObservation,
    ) -> Option<String> {
        for (field, subject_hits, oracle_hits) in [
            ("hits", &subject.hits, &oracle.hits),
            (
                "cutoff_tie_group",
                &subject.cutoff_tie_group,
                &oracle.cutoff_tie_group,
            ),
            (
                "offset_tie_group",
                &subject.offset_tie_group,
                &oracle.offset_tie_group,
            ),
        ] {
            if subject_hits.len() != oracle_hits.len() {
                return Some(format!("/comparison/subject/{field}"));
            }
            if let Some((index, _)) = subject_hits.iter().zip(oracle_hits).enumerate().find(
                |(_, (subject_hit, oracle_hit))| {
                    subject_hit.native_tie_key != oracle_hit.native_tie_key
                },
            ) {
                return Some(format!(
                    "/comparison/subject/{field}/{index}/native_tie_key"
                ));
            }
        }
        None
    }

    fn e55_divergence_panic(
        pointer: &str,
        case_id: Option<&str>,
        baseline: &E55ResidencyEvidence,
        candidate: &E55ResidencyEvidence,
        report: Option<&ComparisonReport>,
        comparator_error: Option<&str>,
    ) -> ! {
        let payload = serde_json::json!({
            "campaign": "quill-e55-mixed-residency-v1",
            "case_id": case_id,
            "first_divergence": pointer,
            "comparison_report": report,
            "comparator_error": comparator_error,
            "state_lists": [baseline, candidate],
        });
        let encoded = serde_json::to_string_pretty(&payload)
            .expect("serialize structured E5.5 divergence evidence");
        panic!("E5.5 mixed-residency divergence\n{encoded}");
    }

    fn e55_assert_residency_exact(
        baseline: &E55ResidencyEvidence,
        candidate: &E55ResidencyEvidence,
    ) {
        if baseline.stats != candidate.stats {
            e55_divergence_panic("/residency/stats", None, baseline, candidate, None, None);
        }
        if baseline.shape.baseline_dead_keeper_segments
            != candidate.shape.baseline_dead_keeper_segments
            || baseline.shape.live_leaf_ranges != candidate.shape.live_leaf_ranges
        {
            e55_divergence_panic(
                "/residency/leaf_geometry",
                None,
                baseline,
                candidate,
                None,
                None,
            );
        }
        if baseline.cases.keys().ne(candidate.cases.keys()) {
            e55_divergence_panic("/residency/cases", None, baseline, candidate, None, None);
        }
        for (case_id, oracle) in &baseline.cases {
            let subject = candidate
                .cases
                .get(case_id)
                .expect("E5.5 state matrix keys were checked");
            if subject.diagnostics != oracle.diagnostics {
                e55_divergence_panic(
                    "/comparison/subject/diagnostics",
                    Some(case_id),
                    baseline,
                    candidate,
                    None,
                    None,
                );
            }
            let report = match compare_observations(
                subject.observation.clone(),
                oracle.observation.clone(),
                ComparatorConfig::default(),
            ) {
                Ok(report) => report,
                Err(error) => {
                    let error = error.to_string();
                    e55_divergence_panic(
                        "/comparison/comparator_contract",
                        Some(case_id),
                        baseline,
                        candidate,
                        None,
                        Some(&error),
                    );
                }
            };
            if report.status != ComparisonStatus::Exact
                || report.rank_class != RankClass::RankExact
                || !report.divergences.is_empty()
            {
                let pointer = report.first_divergence.as_deref().unwrap_or("/comparison");
                e55_divergence_panic(
                    pointer,
                    Some(case_id),
                    baseline,
                    candidate,
                    Some(&report),
                    None,
                );
            }
            if let Some(pointer) =
                e55_first_native_key_divergence(&subject.observation, &oracle.observation)
            {
                e55_divergence_panic(
                    &pointer,
                    Some(case_id),
                    baseline,
                    candidate,
                    Some(&report),
                    None,
                );
            }
        }
    }

    async fn e55_run_mixed_residency_campaign(cx: &Cx, seed: u64, extras_per_delta: usize) {
        let (index, baseline_dead_segments, historical_docid) =
            e55_index_with_live_history(cx).await;
        let mut corpus = e55_build_live_corpus(&index, seed, extras_per_delta, historical_docid);
        let index = e55_tombstone_sealed_upsert_source(&index, historical_docid);
        let successor_generation = index.search_snapshot().keeper_generation();
        corpus.first = Arc::new(corpus.first.rebind_keeper_generation(successor_generation));
        corpus.second = Arc::new(corpus.second.rebind_keeper_generation(successor_generation));
        let mut retired_docids = corpus.retired_docids.clone();
        retired_docids.push(historical_docid);
        retired_docids.sort_unstable();
        let context = E55CaptureContext {
            baseline_dead_segments,
            expected_ranges: &corpus.expected_ranges,
            expected_live: &corpus.expected_live,
            retired_docids: &retired_docids,
            replacement_docid: corpus.replacement_docid,
            seed,
            corpus_hash: corpus.corpus_hash,
            extras_per_delta,
        };
        tracing::info!(
            target: "frankensearch.quill.gauntlet.e55",
            seed,
            corpus_hash = %format_args!("{:016x}", corpus.corpus_hash),
            extras_per_delta,
            live_documents = corpus.expected_live.len(),
            baseline_dead_segments,
            "starting deterministic E5.5 mixed-residency campaign"
        );

        index
            .publish_delta_table(vec![Arc::clone(&corpus.first), Arc::clone(&corpus.second)])
            .expect("publish complete all-Delta E5.5 table");
        let all_delta = e55_capture_residency(&index, cx, "all_delta", 0, 2, &context);

        let mixed_generation = index
            .search_snapshot()
            .keeper_generation()
            .checked_add(1)
            .expect("mixed E5.5 Keeper generation fits u64");
        let surviving_second = Arc::new(corpus.second.rebind_keeper_generation(mixed_generation));
        index
            .seal_delta_snapshot(
                cx,
                Arc::clone(&corpus.first),
                vec![Arc::clone(&surviving_second)],
                e55_flush_input(E55_FIRST_SEGMENT_ID),
            )
            .await
            .expect("seal first E5.5 Delta into mixed residency");
        let mixed = e55_capture_residency(&index, cx, "mixed", 1, 1, &context);

        index
            .seal_delta_snapshot(
                cx,
                surviving_second,
                Vec::new(),
                e55_flush_input(E55_SECOND_SEGMENT_ID),
            )
            .await
            .expect("seal second E5.5 Delta into all-sealed residency");
        let all_sealed = e55_capture_residency(&index, cx, "all_sealed", 2, 0, &context);

        e55_assert_residency_exact(&all_delta, &mixed);
        e55_assert_residency_exact(&all_delta, &all_sealed);
        e55_assert_residency_exact(&mixed, &all_sealed);
        tracing::info!(
            target: "frankensearch.quill.gauntlet.e55",
            seed,
            corpus_hash = %format_args!("{:016x}", corpus.corpus_hash),
            case_count = all_delta.cases.len(),
            "completed exact E5.5 all-Delta to mixed to all-sealed campaign"
        );
    }

    #[test]
    fn e410_edge_state_query_matrix_is_total_and_residency_exact() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let empty = QuillIndex::in_memory_with_schema(E55_SCHEMA, e55_config())
                .expect("construct E4.10 empty index");
            let empty_cases = e410_capture_edge_state(
                &empty,
                &cx,
                "empty",
                E410EdgeStateShape {
                    keeper_segments: 0,
                    keeper_at_seal_documents: 0,
                    keeper_tombstones: 0,
                    delta_leaves: 0,
                    delta_physical_documents: 0,
                    delta_live_documents: 0,
                    live_documents: 0,
                    tombstoned_docid: None,
                },
            );

            let delta_only = e410_delta_only_index();
            let delta_cases = e410_capture_edge_state(
                &delta_only,
                &cx,
                "delta_only",
                E410EdgeStateShape {
                    keeper_segments: 0,
                    keeper_at_seal_documents: 0,
                    keeper_tombstones: 0,
                    delta_leaves: 1,
                    delta_physical_documents: 1,
                    delta_live_documents: 1,
                    live_documents: 1,
                    tombstoned_docid: None,
                },
            );

            let (single, _, _) = e55_index_with_live_history(&cx).await;
            let single_cases = e410_capture_edge_state(
                &single,
                &cx,
                "single_sealed",
                E410EdgeStateShape {
                    keeper_segments: 1,
                    keeper_at_seal_documents: 1,
                    keeper_tombstones: 0,
                    delta_leaves: 0,
                    delta_physical_documents: 0,
                    delta_live_documents: 0,
                    live_documents: 1,
                    tombstoned_docid: None,
                },
            );

            let (all_tombstoned_source, _, retired_docid) = e55_index_with_live_history(&cx).await;
            let all_tombstoned =
                e55_tombstone_sealed_upsert_source(&all_tombstoned_source, retired_docid);
            let tombstoned_cases = e410_capture_edge_state(
                &all_tombstoned,
                &cx,
                "all_tombstoned",
                E410EdgeStateShape {
                    keeper_segments: 1,
                    keeper_at_seal_documents: 1,
                    keeper_tombstones: 1,
                    delta_leaves: 0,
                    delta_physical_documents: 0,
                    delta_live_documents: 0,
                    live_documents: 0,
                    tombstoned_docid: Some(retired_docid),
                },
            );

            assert_eq!(
                delta_cases, single_cases,
                "strict Delta-only and single-sealed states must be bit-exact",
            );
            assert_eq!(
                empty_cases, tombstoned_cases,
                "empty and all-tombstoned states must expose the same public results",
            );
        });
    }

    #[test]
    fn e55_mixed_residency_conformance_is_exact() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            e55_run_mixed_residency_campaign(&cx, 0x55, 0).await;
        });
    }

    #[test]
    #[ignore = "nightly-only fixed-seed mixed-residency conformance campaign"]
    fn e55_seeded_mixed_residency_conformance_is_exact() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            e55_run_mixed_residency_campaign(&cx, E55_NIGHTLY_SEED, 512).await;
        });
    }

    #[test]
    fn live_subject_is_a_trait_object_with_quill_identity() {
        let subject: Box<dyn GauntletEngine> = Box::new(
            QuillSubject::in_memory(QuillConfig::default(), "test-revision", false)
                .expect("live Quill subject"),
        );
        assert_eq!(subject.descriptor().family, EngineFamily::Quill);
        assert_eq!(subject.descriptor().config_hash.len(), 16);
    }

    #[test]
    fn quill_config_hash_covers_every_public_knob() {
        let baseline_config = QuillConfig::default();
        let baseline_hash = quill_config_hash(&baseline_config);
        let variants = [
            (
                "scribe_shard_budget_bytes",
                QuillConfig {
                    scribe_shard_budget_bytes: baseline_config.scribe_shard_budget_bytes + 1,
                    ..baseline_config.clone()
                },
            ),
            (
                "delta_budget_bytes",
                QuillConfig {
                    delta_budget_bytes: baseline_config.delta_budget_bytes + 1,
                    ..baseline_config.clone()
                },
            ),
            (
                "tier_fanout",
                QuillConfig {
                    tier_fanout: baseline_config.tier_fanout + 1,
                    ..baseline_config.clone()
                },
            ),
            (
                "tier_small_max_docid_width",
                QuillConfig {
                    tier_small_max_docid_width: baseline_config.tier_small_max_docid_width + 1,
                    ..baseline_config.clone()
                },
            ),
            (
                "tier_medium_max_docid_width",
                QuillConfig {
                    tier_medium_max_docid_width: baseline_config.tier_medium_max_docid_width + 1,
                    ..baseline_config.clone()
                },
            ),
            (
                "bulk_load_mode",
                QuillConfig {
                    bulk_load_mode: !baseline_config.bulk_load_mode,
                    ..baseline_config.clone()
                },
            ),
            (
                "bulk_publish_segment_cadence",
                QuillConfig {
                    bulk_publish_segment_cadence: baseline_config.bulk_publish_segment_cadence + 1,
                    ..baseline_config.clone()
                },
            ),
            (
                "compaction_tombstone_density",
                QuillConfig {
                    compaction_tombstone_density: 0.21,
                    ..baseline_config.clone()
                },
            ),
            (
                "merge_max_hole_ratio",
                QuillConfig {
                    merge_max_hole_ratio: 0.51,
                    ..baseline_config.clone()
                },
            ),
            (
                "glob_expansion_limit",
                QuillConfig {
                    glob_expansion_limit: baseline_config.glob_expansion_limit + 1,
                    ..baseline_config.clone()
                },
            ),
            (
                "query_fuel_budget",
                QuillConfig {
                    query_fuel_budget: baseline_config.query_fuel_budget + 1,
                    ..baseline_config.clone()
                },
            ),
            (
                "max_ingest_shards",
                QuillConfig {
                    max_ingest_shards: baseline_config.max_ingest_shards + 1,
                    ..baseline_config.clone()
                },
            ),
            (
                "deterministic_ingest",
                QuillConfig {
                    deterministic_ingest: !baseline_config.deterministic_ingest,
                    ..baseline_config.clone()
                },
            ),
            (
                "max_visibility_lag_ms",
                QuillConfig {
                    max_visibility_lag_ms: baseline_config.max_visibility_lag_ms + 1,
                    ..baseline_config.clone()
                },
            ),
            (
                "quarantine_on_unrepairable",
                QuillConfig {
                    quarantine_on_unrepairable: !baseline_config.quarantine_on_unrepairable,
                    ..baseline_config.clone()
                },
            ),
        ];

        let mut observed_hashes = BTreeSet::from([baseline_hash.clone()]);
        for (field, variant) in variants {
            let variant_hash = quill_config_hash(&variant);
            assert_ne!(variant_hash, baseline_hash, "hash omitted {field}");
            assert!(
                observed_hashes.insert(variant_hash),
                "hash collision while mutating {field}"
            );
        }
    }

    #[test]
    fn case_shape_rejects_snippet_budget_at_every_entry_point() {
        let mut case = DifferentialCase::new("snippet-budget", "anything", 1);
        case.snippet_max_chars = Some(MAX_SNIPPET_CHARS + 1);
        assert!(matches!(
            case.validate_shape(),
            Err(GauntletError::InvalidCase { .. })
        ));
    }

    #[test]
    fn cross_engine_guard_rejects_family_even_when_configs_differ() {
        let first = EngineDescriptor {
            family: EngineFamily::Tantivy,
            implementation: "tantivy".to_owned(),
            crate_version: "0.26.1".to_owned(),
            source_revision: "a".repeat(40),
            source_dirty: false,
            config_hash: "one".to_owned(),
        };
        let mut second = first.clone();
        second.config_hash = "two".to_owned();
        assert!(matches!(
            EnginePairIdentity::new(ComparisonMode::CrossEngine, first, second),
            Err(GauntletError::EngineIdentityCollision { .. })
        ));
    }

    #[test]
    fn identity_guard_rejects_empty_subject_provenance() {
        let subject = EngineDescriptor {
            family: EngineFamily::Quill,
            implementation: String::new(),
            crate_version: "0.2.1".to_owned(),
            source_revision: "subject-revision".to_owned(),
            source_dirty: false,
            config_hash: "subject-config".to_owned(),
        };
        let oracle = EngineDescriptor {
            family: EngineFamily::Tantivy,
            implementation: "tantivy".to_owned(),
            crate_version: "0.26.1".to_owned(),
            source_revision: "oracle-revision".to_owned(),
            source_dirty: false,
            config_hash: "oracle-config".to_owned(),
        };
        assert!(matches!(
            EnginePairIdentity::new(ComparisonMode::CrossEngine, subject, oracle),
            Err(GauntletError::InvalidContract { .. })
        ));
    }

    #[test]
    fn identity_guard_rejects_before_engine_execution() {
        let observe_calls = Arc::new(AtomicUsize::new(0));
        let descriptor = EngineDescriptor {
            family: EngineFamily::Tantivy,
            implementation: "counting-tantivy".to_owned(),
            crate_version: "0.26.1".to_owned(),
            source_revision: "test".to_owned(),
            source_dirty: false,
            config_hash: "one".to_owned(),
        };
        let first = CountingEngine {
            descriptor: descriptor.clone(),
            observe_calls: Arc::clone(&observe_calls),
        };
        let mut second_descriptor = descriptor;
        second_descriptor.config_hash = "two".to_owned();
        let second = CountingEngine {
            descriptor: second_descriptor,
            observe_calls: Arc::clone(&observe_calls),
        };
        let harness = DifferentialHarness::default();
        let case = DifferentialCase::new("identity-preflight", "anything", 10);

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            assert!(matches!(
                harness.run(&cx, &first, &second, &case).await,
                Err(GauntletError::EngineIdentityCollision { .. })
            ));
        });
        assert_eq!(observe_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn quill_observation_rejects_any_count_free_rank_drift() {
        let evidence = QuillSearchResult {
            hits: vec![frankensearch_quill::QuillHit {
                document_id: "winner".to_owned(),
                global_docid: 7,
                score: 3.5,
            }],
            total_count: Some(1),
            doc_count: 1,
            diagnostics: Vec::new(),
        };
        let observed = QuillSearchResult {
            total_count: None,
            ..evidence.clone()
        };
        let expected_reason = "Quill observed and exhaustive collector pages differ";

        let mut wrong_external_id = observed.clone();
        wrong_external_id.hits[0].document_id = "other".to_owned();
        let mut wrong_native_tie_key = observed.clone();
        wrong_native_tie_key.hits[0].global_docid = 8;
        let mut wrong_score_bits = observed;
        wrong_score_bits.hits[0].score = f32::from_bits(3.5_f32.to_bits() + 1);

        for mismatch in [wrong_external_id, wrong_native_tie_key, wrong_score_bits] {
            assert!(matches!(
                quill_observation_from_results(&mismatch, &evidence, 1, 0, false),
                Err(GauntletError::InvalidObservation { reason }) if reason == expected_reason
            ));
        }
    }

    #[test]
    fn case_rejects_underfilled_and_overfilled_exact_top_k_evidence() {
        let subject_descriptor = EngineDescriptor {
            family: EngineFamily::Quill,
            implementation: "quill-test".to_owned(),
            crate_version: "0.2.1".to_owned(),
            source_revision: "test".to_owned(),
            source_dirty: false,
            config_hash: "subject".to_owned(),
        };
        let oracle_descriptor = EngineDescriptor {
            family: EngineFamily::Tantivy,
            implementation: "tantivy-test".to_owned(),
            crate_version: "0.26.1".to_owned(),
            source_revision: "test".to_owned(),
            source_dirty: false,
            config_hash: "oracle".to_owned(),
        };
        let engines = EnginePairIdentity::new(
            ComparisonMode::CrossEngine,
            subject_descriptor,
            oracle_descriptor,
        )
        .expect("distinct engines");
        let underfilled = EngineObservation {
            hits: Vec::new(),
            cutoff_tie_group: Vec::new(),
            cutoff_tie_complete: true,
            offset_tie_group: Vec::new(),
            offset_tie_complete: false,
            snippets: BTreeMap::new(),
            match_count: CountState::Value(2),
            doc_count: 2,
            ast_differences: Vec::new(),
        };
        let case = DifferentialCase::new("underfilled", "query", 10);
        assert!(
            case.validate_observations(&engines, &underfilled, &underfilled)
                .is_err()
        );

        let quill_hit = RankedHit {
            doc_id: "one".to_owned(),
            score_bits: 1.0_f32.to_bits(),
            native_tie_key: NativeTieKey::QuillDocId { doc_id: 1 },
        };
        let subject_overfilled = EngineObservation {
            hits: vec![quill_hit.clone()],
            cutoff_tie_group: vec![quill_hit],
            cutoff_tie_complete: true,
            offset_tie_group: Vec::new(),
            offset_tie_complete: false,
            snippets: BTreeMap::new(),
            match_count: CountState::Value(1),
            doc_count: 1,
            ast_differences: Vec::new(),
        };
        let oracle_empty = EngineObservation {
            hits: Vec::new(),
            cutoff_tie_group: Vec::new(),
            cutoff_tie_complete: true,
            offset_tie_group: Vec::new(),
            offset_tie_complete: false,
            snippets: BTreeMap::new(),
            match_count: CountState::Value(0),
            doc_count: 1,
            ast_differences: Vec::new(),
        };
        let zero_limit = DifferentialCase::new("overfilled", "query", 0);
        assert!(
            zero_limit
                .validate_observations(&engines, &subject_overfilled, &oracle_empty)
                .is_err()
        );

        let malformed_offset = EngineObservation {
            hits: vec![RankedHit {
                doc_id: "page".to_owned(),
                score_bits: 1.0_f32.to_bits(),
                native_tie_key: NativeTieKey::QuillDocId { doc_id: 2 },
            }],
            cutoff_tie_group: Vec::new(),
            cutoff_tie_complete: true,
            offset_tie_group: vec![RankedHit {
                doc_id: "prefix".to_owned(),
                score_bits: 2.0_f32.to_bits(),
                native_tie_key: NativeTieKey::QuillDocId { doc_id: 1 },
            }],
            offset_tie_complete: true,
            snippets: BTreeMap::new(),
            match_count: CountState::Value(2),
            doc_count: 2,
            ast_differences: Vec::new(),
        };
        let mut paginated = DifferentialCase::new("malformed-offset", "query", 1);
        paginated.offset = 1;
        assert!(
            paginated
                .validate_observations(&engines, &malformed_offset, &malformed_offset)
                .is_err()
        );
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn separate_tantivy_instances_fail_before_execution() {
        let revision = oracle_version_contract()
            .expect("version contract")
            .lexical_git_revision;
        let first = TantivyOracle::in_memory(&revision, false).expect("first oracle");
        let second = TantivyOracle::in_memory(&revision, false).expect("second oracle");
        let harness = DifferentialHarness::default();
        let case = DifferentialCase::new("identity-guard", "anything", 10);

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let error = harness
                .run(&cx, &first, &second, &case)
                .await
                .expect_err("oracle-vs-oracle must fail before observation");
            assert!(matches!(
                error,
                GauntletError::EngineIdentityCollision { .. }
            ));
        });
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn oracle_observation_retains_full_tie_evidence_and_exact_count() {
        let revision = oracle_version_contract()
            .expect("version contract")
            .lexical_git_revision;
        let mut oracle = TantivyOracle::in_memory(&revision, false).expect("oracle");
        let documents = vec![
            frankensearch_core::IndexableDocument::new("a", "shared token"),
            frankensearch_core::IndexableDocument::new("b", "shared token"),
            frankensearch_core::IndexableDocument::new("c", "shared token"),
        ];
        let mut case = DifferentialCase::new("oracle-observation", "shared", 2);
        case.tie_expansion_limit = 8;
        let mut exhausted_case = case.clone();
        exhausted_case.fixture_id = "oracle-exhausted-tie-expansion".to_owned();
        exhausted_case.tie_expansion_limit = 0;
        let mut zero_limit_case = case.clone();
        zero_limit_case.fixture_id = "oracle-zero-limit-count".to_owned();
        zero_limit_case.limit = 0;
        let mut paginated_case = case.clone();
        paginated_case.fixture_id = "oracle-offset-tie".to_owned();
        paginated_case.offset = 1;
        paginated_case.limit = 1;

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            oracle
                .index_documents(&cx, &documents)
                .await
                .expect("index oracle corpus");
            let observation = oracle.observe(&cx, &case).await.expect("observe query");
            assert_eq!(observation.hits.len(), 2);
            assert_eq!(observation.cutoff_tie_group.len(), 3);
            assert!(observation.cutoff_tie_complete);
            assert_eq!(observation.match_count, CountState::Value(3));
            assert_eq!(observation.doc_count, 3);
            assert!(
                observation.hits.iter().all(|hit| matches!(
                    hit.native_tie_key,
                    NativeTieKey::TantivyDocAddress { .. }
                ))
            );

            let exhausted = oracle
                .observe(&cx, &exhausted_case)
                .await
                .expect("observe exhausted tie expansion");
            assert_eq!(exhausted.hits.len(), 2);
            assert_eq!(exhausted.cutoff_tie_group.len(), 2);
            assert!(!exhausted.cutoff_tie_complete);
            assert_eq!(exhausted.match_count, CountState::Value(3));

            let zero_limit = oracle
                .observe(&cx, &zero_limit_case)
                .await
                .expect("observe zero-limit exact count");
            assert!(zero_limit.hits.is_empty());
            assert!(zero_limit.cutoff_tie_group.is_empty());
            assert!(zero_limit.cutoff_tie_complete);
            assert_eq!(zero_limit.match_count, CountState::Value(3));

            let paginated = oracle
                .observe(&cx, &paginated_case)
                .await
                .expect("observe offset inside tie");
            assert_eq!(paginated.hits.len(), 1);
            assert_eq!(paginated.offset_tie_group.len(), 3);
            assert!(paginated.offset_tie_complete);
            assert_eq!(paginated.match_count, CountState::Value(3));
        });
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn oracle_lower_score_sentinel_completes_cutoff_tie_group() {
        let revision = oracle_version_contract()
            .expect("version contract")
            .lexical_git_revision;
        let mut oracle = TantivyOracle::in_memory(&revision, false).expect("oracle");
        let documents = vec![
            frankensearch_core::IndexableDocument::new("a", "alpha beta"),
            frankensearch_core::IndexableDocument::new("b", "alpha beta"),
            frankensearch_core::IndexableDocument::new("c", "alpha"),
            frankensearch_core::IndexableDocument::new("d", "alpha"),
            frankensearch_core::IndexableDocument::new("e", "alpha"),
        ];
        let mut case = DifferentialCase::new("oracle-lower-score-sentinel", "alpha beta", 1);
        case.tie_expansion_limit = 2;

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            oracle
                .index_documents(&cx, &documents)
                .await
                .expect("index oracle corpus");
            let observation = oracle.observe(&cx, &case).await.expect("observe query");
            assert_eq!(observation.hits.len(), 1);
            assert_eq!(observation.cutoff_tie_group.len(), 2);
            assert!(observation.cutoff_tie_complete);
            assert_eq!(observation.match_count, CountState::Value(5));
        });
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn fieldnorm_codec_matches_tantivy_decodes_and_encode_boundaries() {
        use frankensearch_lexical::tantivy_crate::fieldnorm::FieldNormReader;
        use frankensearch_quill::contract::{FIELD_NORMS_TABLE, fieldnorm_to_id, id_to_fieldnorm};

        for id in 0..=u8::MAX {
            assert_eq!(
                id_to_fieldnorm(id),
                FieldNormReader::id_to_fieldnorm(id),
                "decode id={id}"
            );
        }
        // The encoder is constant between consecutive table boundaries, so
        // each boundary and both adjacent values cover every output interval.
        let mut probes = vec![0, u32::MAX];
        for &boundary in &FIELD_NORMS_TABLE {
            probes.push(boundary);
            probes.push(boundary.saturating_sub(1));
            probes.push(boundary.saturating_add(1));
        }
        probes.sort_unstable();
        probes.dedup();
        for length in probes {
            assert_eq!(
                fieldnorm_to_id(length),
                FieldNormReader::fieldnorm_to_id(length),
                "encode length={length}"
            );
        }
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn e410_controlled_public_search_semantics_match_oracle() {
        let revision = oracle_version_contract()
            .expect("oracle version contract")
            .lexical_git_revision;
        let mut subject = QuillSubject::in_memory(e55_config(), "e410-subject", false)
            .expect("E4.10 Quill subject");
        let mut oracle =
            TantivyOracle::in_memory_scalar_g1a(&revision, false).expect("E4.10 Tantivy oracle");
        let documents = vec![
            frankensearch_core::IndexableDocument::new("title-hit", "quiet filler")
                .with_title("Needle"),
            frankensearch_core::IndexableDocument::new(
                "content-hit",
                "needle filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler",
            )
            .with_title("quiet"),
            frankensearch_core::IndexableDocument::new(
                "hyphen-hit",
                "ERR-404 troubleshooting guide",
            ),
            frankensearch_core::IndexableDocument::new("case-hit", "MiXeDcAsE identifier"),
            frankensearch_core::IndexableDocument::new("special-hit", "C++ interop"),
        ];

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            subject
                .claim_fresh_campaign()
                .expect("claim E4.10 subject campaign");
            subject
                .index_mut()
                .expect("E4.10 subject index")
                .index_documents(&cx, &documents)
                .await
                .expect("index E4.10 subject corpus");
            subject
                .index_mut()
                .expect("E4.10 subject index")
                .commit(&cx)
                .await
                .expect("commit E4.10 subject corpus");
            subject
                .mark_committed()
                .expect("publish E4.10 subject campaign");

            oracle
                .claim_fresh_campaign()
                .expect("claim E4.10 oracle campaign");
            oracle
                .index_documents(&cx, &documents)
                .await
                .expect("index E4.10 oracle corpus");
            oracle
                .mark_committed()
                .expect("publish E4.10 oracle campaign");

            let harness = DifferentialHarness::default();
            let mut casefold_hits = None;
            for (id, query) in [
                ("title-boost", "needle"),
                ("casefold-lower", "mixedcase"),
                ("casefold-mixed", "MiXeDcAsE"),
                ("hyphen", "ERR-404"),
                ("special-chars", "C++"),
                ("empty-query", ""),
            ] {
                let mut case = DifferentialCase::new(format!("e410-{id}"), query, 10);
                case.snippet_max_chars = None;
                let run = harness
                    .run(&cx, &subject, &oracle, &case)
                    .await
                    .unwrap_or_else(|error| panic!("E4.10 case {id} failed: {error}"));
                assert_eq!(
                    run.comparison.status,
                    ComparisonStatus::Exact,
                    "E4.10 case {id}: {:?}",
                    run.comparison.divergences,
                );
                assert_eq!(run.comparison.rank_class, RankClass::RankExact);
                if id == "title-boost" {
                    assert_eq!(
                        run.comparison
                            .subject
                            .hits
                            .first()
                            .map(|hit| hit.doc_id.as_str()),
                        Some("title-hit"),
                        "title-field boost must outrank a content-only hit",
                    );
                }
                let ids = run
                    .comparison
                    .subject
                    .hits
                    .iter()
                    .map(|hit| hit.doc_id.clone())
                    .collect::<Vec<_>>();
                if id.starts_with("casefold-") {
                    assert_eq!(
                        ids,
                        vec!["case-hit".to_owned()],
                        "case-folded query must retrieve the intended mixed-case document",
                    );
                    if let Some(expected) = &casefold_hits {
                        assert_eq!(&ids, expected, "case-folded queries changed the hit set");
                    } else {
                        casefold_hits = Some(ids.clone());
                    }
                }
                if id == "hyphen" {
                    assert!(
                        ids.iter().any(|doc_id| doc_id == "hyphen-hit"),
                        "hyphenated query must retrieve the intended document: {ids:?}",
                    );
                }
                if id == "special-chars" {
                    assert_eq!(
                        ids,
                        vec!["special-hit".to_owned()],
                        "special-character query must retrieve the intended document",
                    );
                }
                if id == "empty-query" {
                    assert!(ids.is_empty(), "empty query must return no hits");
                }
            }
        });
    }

    /// E4.10 limit/count/order semantics ported from the lexical engine's
    /// public-surface tests (`search_respects_limit`,
    /// `zero_limit_returns_empty_without_collector_panic`,
    /// `no_results_for_unmatched_query`, `search_scores_are_descending`,
    /// `doc_count_accurate_after_operations`): every case must stay
    /// rank-exact against the oracle (bd-quill-e4-argus-3ycz.10).
    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn e410_limit_count_and_order_semantics_match_oracle() {
        let revision = oracle_version_contract()
            .expect("oracle version contract")
            .lexical_git_revision;
        let mut subject = QuillSubject::in_memory(e55_config(), "e410-limits-subject", false)
            .expect("E4.10 limits Quill subject");
        let mut oracle = TantivyOracle::in_memory_scalar_g1a(&revision, false)
            .expect("E4.10 limits Tantivy oracle");
        // Every document carries "shared" exactly once at a distinct document
        // length, so the counted match-all case has five distinct scores;
        // "rust" matches exactly two documents at distinct (tf, |d|) pairs.
        let documents = vec![
            frankensearch_core::IndexableDocument::new(
                "doc-1",
                "rust is a systems programming language shared",
            )
            .with_title("borrow checker"),
            frankensearch_core::IndexableDocument::new(
                "doc-2",
                "machine learning with neural networks shared",
            )
            .with_title("gradient guide"),
            frankensearch_core::IndexableDocument::new("doc-3", "rust rust rust ownership shared")
                .with_title("moved values"),
            frankensearch_core::IndexableDocument::new("doc-4", "databases and storage shared")
                .with_title("storage primer"),
            frankensearch_core::IndexableDocument::new(
                "doc-5",
                "distributed consensus algorithms paxos raft quorum vault shared",
            )
            .with_title("consensus notes"),
        ];

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            subject
                .claim_fresh_campaign()
                .expect("claim E4.10 limits subject campaign");
            subject
                .index_mut()
                .expect("E4.10 limits subject index")
                .index_documents(&cx, &documents)
                .await
                .expect("index E4.10 limits subject corpus");
            subject
                .index_mut()
                .expect("E4.10 limits subject index")
                .commit(&cx)
                .await
                .expect("commit E4.10 limits subject corpus");
            subject
                .mark_committed()
                .expect("publish E4.10 limits subject campaign");

            oracle
                .claim_fresh_campaign()
                .expect("claim E4.10 limits oracle campaign");
            oracle
                .index_documents(&cx, &documents)
                .await
                .expect("index E4.10 limits oracle corpus");
            oracle
                .mark_committed()
                .expect("publish E4.10 limits oracle campaign");

            let harness = DifferentialHarness::default();
            for (id, query, limit, expected_hits) in [
                ("limit-zero", "rust", 0_u64, 0_usize),
                ("limit-one", "rust", 1, 1),
                ("limit-exact", "rust", 2, 2),
                ("limit-headroom", "rust", 10, 2),
                ("no-match", "zzzabsent", 10, 0),
                ("match-all-counted", "shared", 10, 5),
            ] {
                let mut case = DifferentialCase::new(format!("e410-{id}"), query, limit);
                case.snippet_max_chars = None;
                let run = harness
                    .run(&cx, &subject, &oracle, &case)
                    .await
                    .unwrap_or_else(|error| panic!("E4.10 limits case {id} failed: {error}"));
                assert_eq!(
                    run.comparison.status,
                    ComparisonStatus::Exact,
                    "E4.10 limits case {id}: {:?}",
                    run.comparison.divergences,
                );
                assert_eq!(run.comparison.rank_class, RankClass::RankExact);
                assert_eq!(
                    run.comparison.subject.hits.len(),
                    expected_hits,
                    "E4.10 limits case {id} returned the wrong page size",
                );
                if id == "match-all-counted" {
                    assert_eq!(
                        run.comparison.subject.match_count,
                        crate::comparator::CountState::Value(5),
                        "an exact count over the shared term must see every live document",
                    );
                    let scores = run
                        .comparison
                        .subject
                        .hits
                        .iter()
                        .map(|hit| f32::from_bits(hit.score_bits))
                        .collect::<Vec<_>>();
                    assert!(
                        scores.windows(2).all(|pair| pair[0] >= pair[1]),
                        "public ranking must be non-increasing in score: {scores:?}",
                    );
                }
            }
        });
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn doclen_and_raw_stats_match_tantivy_before_and_after_delete() {
        use frankensearch_lexical::tantivy_crate::indexer::NoMergePolicy;
        use frankensearch_lexical::tantivy_crate::query::Bm25StatisticsProvider;
        use frankensearch_lexical::tantivy_crate::schema::{STORED, STRING, Schema, TEXT};
        use frankensearch_lexical::tantivy_crate::{Index, Term, doc};
        use frankensearch_quill::contract::id_to_fieldnorm;
        use frankensearch_quill::quiver::{
            DocLenFieldInput, EncodedDocLenSection, EncodedStatsSection, FieldStats,
            aggregate_field_stats,
        };

        fn tokens(count: usize) -> String {
            std::iter::repeat_n("x", count)
                .collect::<Vec<_>>()
                .join(" ")
        }

        assert_eq!(
            oracle_version_contract()
                .expect("version contract")
                .tantivy_version,
            "0.26.1"
        );
        let raw_lengths = [41_u32, 42, 65];
        let mut schema_builder = Schema::builder();
        let id = schema_builder.add_text_field("id", STRING | STORED);
        let content = schema_builder.add_text_field("content", TEXT | STORED);
        let index = Index::create_in_ram(schema_builder.build());
        let mut writer = index
            .writer_with_num_threads(1, 50_000_000)
            .expect("single-segment oracle writer");
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for (document_index, &length) in raw_lengths.iter().enumerate() {
            writer
                .add_document(doc!(
                    id => format!("stats-{document_index}"),
                    content => tokens(usize::try_from(length).unwrap_or(usize::MAX)),
                ))
                .expect("add oracle document");
        }
        writer.commit().expect("commit oracle fixture");
        let reader = index.reader().expect("oracle reader");
        reader.reload().expect("reload committed oracle");
        let searcher = reader.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);

        let oracle_tokens = Bm25StatisticsProvider::total_num_tokens(&searcher, content)
            .expect("oracle token count");
        let oracle_docs =
            Bm25StatisticsProvider::total_num_docs(&searcher).expect("oracle document count");
        assert_eq!(oracle_tokens, 148);
        assert_eq!(oracle_docs, 3);
        let mut oracle_ids = searcher
            .segment_readers()
            .iter()
            .flat_map(|segment| {
                let norms = segment
                    .get_fieldnorms_reader(content)
                    .expect("content fieldnorms");
                (0..segment.max_doc())
                    .map(move |doc| norms.fieldnorm_id(doc))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        oracle_ids.sort_unstable();

        let lengths = raw_lengths.map(Some);
        let inputs = [DocLenFieldInput::new(1, &lengths)];
        let encoded_doclen =
            EncodedDocLenSection::encode(0, 3, &[1], &inputs).expect("Quill DOCLEN");
        let mut quill_ids = encoded_doclen
            .section(&[1])
            .expect("parse Quill DOCLEN")
            .field(1)
            .expect("Quill content field")
            .fieldnorm_ids()
            .to_vec();
        quill_ids.sort_unstable();
        assert_eq!(quill_ids, oracle_ids);

        let segment_stats = searcher
            .segment_readers()
            .iter()
            .map(|segment| {
                let row = [FieldStats::new(
                    1,
                    segment
                        .inverted_index(content)
                        .expect("inverted index")
                        .total_num_tokens(),
                    segment.max_doc(),
                )];
                EncodedStatsSection::encode(&[1], &row, segment.max_doc())
                    .expect("encode segment STATS")
                    .section(&[1])
                    .expect("parse segment STATS")
            })
            .collect::<Vec<_>>();
        let aggregate = aggregate_field_stats(segment_stats.iter())
            .expect("aggregate multi-segment Quill STATS");
        let raw_avgdl = aggregate[0]
            .average_field_length()
            .expect("non-empty average");
        assert_eq!(raw_avgdl.to_bits(), (148.0_f32 / 3.0).to_bits());
        let decoded_avgdl =
            oracle_ids.iter().copied().map(id_to_fieldnorm).sum::<u32>() as f32 / 3.0;
        assert_ne!(raw_avgdl.to_bits(), decoded_avgdl.to_bits());

        drop(searcher);
        writer.delete_term(Term::from_field_text(id, "stats-1"));
        writer.commit().expect("commit oracle delete");
        reader.reload().expect("reload oracle deletion");
        let deleted_searcher = reader.searcher();
        assert_eq!(deleted_searcher.num_docs(), 2);
        assert_eq!(
            Bm25StatisticsProvider::total_num_docs(&deleted_searcher)
                .expect("post-delete oracle document count"),
            3
        );
        assert_eq!(
            Bm25StatisticsProvider::total_num_tokens(&deleted_searcher, content)
                .expect("post-delete oracle token count"),
            148
        );
        let mut post_delete_ids = deleted_searcher
            .segment_readers()
            .iter()
            .flat_map(|segment| {
                let norms = segment
                    .get_fieldnorms_reader(content)
                    .expect("post-delete fieldnorms");
                (0..segment.max_doc())
                    .map(move |doc| norms.fieldnorm_id(doc))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        post_delete_ids.sort_unstable();
        assert_eq!(post_delete_ids, oracle_ids);
        assert!(
            deleted_searcher
                .segment_readers()
                .iter()
                .any(|segment| { (0..segment.max_doc()).any(|doc| segment.is_deleted(doc)) })
        );

        let post_delete_sections = deleted_searcher
            .segment_readers()
            .iter()
            .map(|segment| {
                let row = [FieldStats::new(
                    1,
                    segment
                        .inverted_index(content)
                        .expect("post-delete inverted index")
                        .total_num_tokens(),
                    segment.max_doc(),
                )];
                EncodedStatsSection::encode(&[1], &row, segment.max_doc())
                    .expect("encode post-delete STATS")
                    .section(&[1])
                    .expect("parse post-delete STATS")
            })
            .collect::<Vec<_>>();
        let post_delete = aggregate_field_stats(post_delete_sections.iter())
            .expect("aggregate post-delete STATS");
        assert_eq!(post_delete[0].total_tokens, 148);
        assert_eq!(post_delete[0].doc_count, 3);
        assert_eq!(
            post_delete[0].average_field_length().map(f32::to_bits),
            Some(raw_avgdl.to_bits())
        );
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn oracle_constructor_rejects_dirty_or_mismatched_source() {
        let revision = oracle_version_contract()
            .expect("version contract")
            .lexical_git_revision;
        assert!(TantivyOracle::in_memory(&revision, true).is_err());
        assert!(TantivyOracle::in_memory(&"0".repeat(40), false).is_err());
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn oracle_rejects_oversized_case_before_query_execution() {
        let revision = oracle_version_contract()
            .expect("version contract")
            .lexical_git_revision;
        let oracle = TantivyOracle::in_memory(&revision, false).expect("oracle");
        let mut case = DifferentialCase::new("oversized", "anything", MAX_ORACLE_LIMIT + 1);
        case.tie_expansion_limit = 0;

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            assert!(matches!(
                oracle.observe(&cx, &case).await,
                Err(GauntletError::InvalidCase { .. })
            ));
        });
    }
}
