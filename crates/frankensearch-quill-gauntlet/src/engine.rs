use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;

use asupersync::Cx;
#[cfg(feature = "tantivy-oracle")]
use frankensearch_quill::scribe::{
    CassAnalyzer, ColumnarAccumulator, DOC_ORDS_PER_LEASE, FlushDocumentInput, FlushMode,
    FlushSegmentInput, IndexedFieldValue, IndexedNumericValue, StoredFieldValue,
    flush_accumulator_with_mode,
};
#[cfg(feature = "tantivy-oracle")]
use frankensearch_quill::{
    CASS_SEMANTIC_SCHEMA, CURRENT_ENGINE_VERSION, EncodedSegment, KeeperSnapshot,
    ManifestFieldStats, ManifestSegment, TombstoneSet,
};
use frankensearch_quill::{QuillConfig, QuillIndex, QuillSearchResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
#[cfg(any(test, feature = "tantivy-oracle"))]
use xxhash_rust::xxh3::xxh3_64;

use crate::GauntletError;
use crate::artifact::GauntletProducerBuildIdentity;
use crate::comparator::{
    ComparatorConfig, ComparisonReport, CountState, EngineObservation, NativeTieKey, RankedHit,
    compare_observations,
};
#[cfg(feature = "tantivy-oracle")]
use crate::generator::GeneratedDocument;
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
pub const CASS_TANTIVY_ORACLE_CONFIG_HASH: &str = "cass-schema-and-parser-v1";
const BUILT_IN_PROFILE_V1_QUILL_CRATE_VERSION: &str = "0.2.1";
const BUILT_IN_PROFILE_V1_LEXICAL_CRATE_VERSION: &str = "0.2.1";
const BUILT_IN_PROFILE_V1_DEFAULT_ANALYZER_HASH: &str =
    "7425c0f2d0a909ca4103bd20f439b6282d3ce00ab3c9f6784ec7333398197041";
const BUILT_IN_PROFILE_V1_DEFAULT_SCHEMA_HASH: &str =
    "9fed22a53e5060243e9528fbbf40605a0df8ea120b3d74ac41ecbb097c2df571";
const BUILT_IN_PROFILE_V1_SCALAR_G1A_SCHEMA_HASH: &str =
    "31c57f7e822289f5d1b685b3d92a75ab66697e3c4846ebb9315cc96e75dd9f53";
const BUILT_IN_PROFILE_V1_CASS_ANALYZER_HASH: &str =
    "8db8c441617927a16604df40ff17f57a5478996eaa2b0c7b4018dfac1340edcf";
const BUILT_IN_PROFILE_V1_CASS_SCHEMA_HASH: &str =
    "24e54284be158fe39dfa4bf0def76dba6dd9d50d8c59f7cb75f24e52b0cccae4";
const BUILT_IN_PROFILE_V1_SCALAR_ORACLE_CONFIG_HASH: &str = "shipping-schema-and-parser-v1";
const BUILT_IN_PROFILE_V1_CASS_ORACLE_CONFIG_HASH: &str = "cass-schema-and-parser-v1";

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
#[serde(deny_unknown_fields)]
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

fn is_canonical_git_revision(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_recorded_producer_source(
    observed_revision: &str,
    source_dirty: bool,
) -> Result<(), GauntletError> {
    if !(is_canonical_git_revision(observed_revision)
        || observed_revision == "unavailable" && source_dirty)
    {
        return Err(GauntletError::InvalidContract {
            reason: "recorded producer revision must be a canonical lowercase 40-hex Git identity or the conservative dirty unavailable sentinel"
                .to_owned(),
        });
    }
    Ok(())
}

fn canonical_f64_bits(value: f64) -> u64 {
    if value == 0.0 { 0 } else { value.to_bits() }
}

fn usize_to_receipt_u64(value: usize, field: &str) -> u64 {
    u64::try_from(value).unwrap_or_else(|error| {
        panic!("Quill config field {field} exceeds the portable u64 receipt domain: {error}")
    })
}

fn receipt_u64_to_usize(value: u64, field: &str) -> Result<usize, GauntletError> {
    usize::try_from(value).map_err(|_| GauntletError::InvalidContract {
        reason: format!(
            "stored Quill runtime configuration field {field} does not fit this target's usize"
        ),
    })
}

fn sha256_lower_hex(bytes: &[u8]) -> String {
    const DOMAIN: &[u8] = b"frankensearch-quill-config-receipt-v1\0";
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN);
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

/// Canonical, replayable preimage of every public Quill runtime knob.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuillConfigReceipt {
    schema_version: u32,
    scribe_shard_budget_bytes: u64,
    delta_budget_bytes: u64,
    tier_fanout: u64,
    tier_small_max_docid_width: u64,
    tier_medium_max_docid_width: u64,
    bulk_load_mode: bool,
    bulk_publish_segment_cadence: u64,
    compaction_tombstone_density_bits: u64,
    merge_max_hole_ratio_bits: u64,
    glob_expansion_limit: u64,
    query_fuel_budget: u64,
    max_ingest_shards: u64,
    deterministic_ingest: bool,
    max_visibility_lag_ms: u64,
    quarantine_on_unrepairable: bool,
}

impl QuillConfigReceipt {
    const V1_SCHEMA_VERSION: u32 = 1;
    const CURRENT_SCHEMA_VERSION: u32 = Self::V1_SCHEMA_VERSION;

    fn from_config(config: &QuillConfig) -> Self {
        let QuillConfig {
            scribe_shard_budget_bytes,
            delta_budget_bytes,
            tier_fanout,
            tier_small_max_docid_width,
            tier_medium_max_docid_width,
            bulk_load_mode,
            bulk_publish_segment_cadence,
            compaction_tombstone_density,
            merge_max_hole_ratio,
            glob_expansion_limit,
            query_fuel_budget,
            max_ingest_shards,
            deterministic_ingest,
            max_visibility_lag_ms,
            quarantine_on_unrepairable,
        } = config.clone();
        Self {
            schema_version: Self::CURRENT_SCHEMA_VERSION,
            scribe_shard_budget_bytes: usize_to_receipt_u64(
                scribe_shard_budget_bytes,
                "scribe_shard_budget_bytes",
            ),
            delta_budget_bytes: usize_to_receipt_u64(delta_budget_bytes, "delta_budget_bytes"),
            tier_fanout: usize_to_receipt_u64(tier_fanout, "tier_fanout"),
            tier_small_max_docid_width,
            tier_medium_max_docid_width,
            bulk_load_mode,
            bulk_publish_segment_cadence: usize_to_receipt_u64(
                bulk_publish_segment_cadence,
                "bulk_publish_segment_cadence",
            ),
            compaction_tombstone_density_bits: canonical_f64_bits(compaction_tombstone_density),
            merge_max_hole_ratio_bits: canonical_f64_bits(merge_max_hole_ratio),
            glob_expansion_limit: usize_to_receipt_u64(
                glob_expansion_limit,
                "glob_expansion_limit",
            ),
            query_fuel_budget,
            max_ingest_shards: usize_to_receipt_u64(max_ingest_shards, "max_ingest_shards"),
            deterministic_ingest,
            max_visibility_lag_ms,
            quarantine_on_unrepairable,
        }
    }

    fn to_config(&self) -> Result<QuillConfig, GauntletError> {
        Ok(QuillConfig {
            scribe_shard_budget_bytes: receipt_u64_to_usize(
                self.scribe_shard_budget_bytes,
                "scribe_shard_budget_bytes",
            )?,
            delta_budget_bytes: receipt_u64_to_usize(
                self.delta_budget_bytes,
                "delta_budget_bytes",
            )?,
            tier_fanout: receipt_u64_to_usize(self.tier_fanout, "tier_fanout")?,
            tier_small_max_docid_width: self.tier_small_max_docid_width,
            tier_medium_max_docid_width: self.tier_medium_max_docid_width,
            bulk_load_mode: self.bulk_load_mode,
            bulk_publish_segment_cadence: receipt_u64_to_usize(
                self.bulk_publish_segment_cadence,
                "bulk_publish_segment_cadence",
            )?,
            compaction_tombstone_density: f64::from_bits(self.compaction_tombstone_density_bits),
            merge_max_hole_ratio: f64::from_bits(self.merge_max_hole_ratio_bits),
            glob_expansion_limit: receipt_u64_to_usize(
                self.glob_expansion_limit,
                "glob_expansion_limit",
            )?,
            query_fuel_budget: self.query_fuel_budget,
            max_ingest_shards: receipt_u64_to_usize(self.max_ingest_shards, "max_ingest_shards")?,
            deterministic_ingest: self.deterministic_ingest,
            max_visibility_lag_ms: self.max_visibility_lag_ms,
            quarantine_on_unrepairable: self.quarantine_on_unrepairable,
        })
    }

    fn canonical_preimage_v1(&self) -> String {
        format!(
            "quill-config-v1;scribe={};delta={};fanout={};tier_small={};tier_medium={};bulk={};bulk_cadence={};compact={:016x};holes={:016x};glob={};fuel={};shards={};deterministic={};visibility_ms={};quarantine={}",
            self.scribe_shard_budget_bytes,
            self.delta_budget_bytes,
            self.tier_fanout,
            self.tier_small_max_docid_width,
            self.tier_medium_max_docid_width,
            self.bulk_load_mode,
            self.bulk_publish_segment_cadence,
            self.compaction_tombstone_density_bits,
            self.merge_max_hole_ratio_bits,
            self.glob_expansion_limit,
            self.query_fuel_budget,
            self.max_ingest_shards,
            self.deterministic_ingest,
            self.max_visibility_lag_ms,
            self.quarantine_on_unrepairable
        )
    }

    fn descriptor_hash_v1(&self) -> String {
        sha256_lower_hex(self.canonical_preimage_v1().as_bytes())
    }

    fn validate_stored_v1(&self) -> Result<(), GauntletError> {
        let compaction_tombstone_density = f64::from_bits(self.compaction_tombstone_density_bits);
        let merge_max_hole_ratio = f64::from_bits(self.merge_max_hole_ratio_bits);
        if self.schema_version != Self::V1_SCHEMA_VERSION
            || self.scribe_shard_budget_bytes == 0
            || self.delta_budget_bytes == 0
            || self.tier_fanout < 2
            || self.tier_small_max_docid_width == 0
            || self.tier_medium_max_docid_width <= self.tier_small_max_docid_width
            || self.bulk_publish_segment_cadence == 0
            || !compaction_tombstone_density.is_finite()
            || !(0.0..=1.0).contains(&compaction_tombstone_density)
            || compaction_tombstone_density == 0.0
            || !merge_max_hole_ratio.is_finite()
            || !(0.0..=1.0).contains(&merge_max_hole_ratio)
            || self.merge_max_hole_ratio_bits == (-0.0_f64).to_bits()
            || self.glob_expansion_limit == 0
            || self.query_fuel_budget == 0
            || self.max_ingest_shards == 0
            || self.max_visibility_lag_ms == 0
        {
            return Err(GauntletError::InvalidContract {
                reason: "stored Quill runtime configuration receipt v1 is invalid".to_owned(),
            });
        }
        Ok(())
    }

    fn validate_creation(&self) -> Result<(), GauntletError> {
        self.validate_stored_v1()?;
        self.to_config()?
            .validate()
            .map_err(|error| GauntletError::InvalidContract {
                reason: format!(
                    "Quill runtime configuration receipt is invalid under the current engine: {error}"
                ),
            })
    }
}

/// Exact built-in adapter/profile lane selected by a typed execution path.
///
/// This enum is a configuration identity, not authentication or admission
/// authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuiltInEngineProfile {
    ScalarShipping,
    ScalarG1a,
    Cass,
}

/// Stored profile receipt binding the adapter role to the full Quill config.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuiltInEngineProfileReceipt {
    schema_version: u32,
    profile: BuiltInEngineProfile,
    subject_config: QuillConfigReceipt,
}

impl BuiltInEngineProfileReceipt {
    const V1_SCHEMA_VERSION: u32 = 1;
    #[cfg_attr(
        not(any(test, feature = "tantivy-oracle")),
        expect(
            dead_code,
            reason = "typed built-in receipts are constructed only by oracle-backed or test lanes"
        )
    )]
    const CURRENT_SCHEMA_VERSION: u32 = Self::V1_SCHEMA_VERSION;

    #[cfg_attr(
        not(any(test, feature = "tantivy-oracle")),
        expect(
            dead_code,
            reason = "typed built-in receipts are constructed only by oracle-backed or test lanes"
        )
    )]
    pub(crate) fn new(profile: BuiltInEngineProfile, subject_config: &QuillConfig) -> Self {
        Self {
            schema_version: Self::CURRENT_SCHEMA_VERSION,
            profile,
            subject_config: QuillConfigReceipt::from_config(subject_config),
        }
    }

    fn current_semantic_contract(&self) -> crate::runner::SemanticContract {
        match self.profile {
            BuiltInEngineProfile::ScalarShipping => {
                crate::runner::SemanticContract::shipping_default()
            }
            BuiltInEngineProfile::ScalarG1a => crate::runner::SemanticContract::scalar_g1a(),
            BuiltInEngineProfile::Cass => crate::runner::SemanticContract::cass(),
        }
    }

    fn validate_stored(&self, engines: &EnginePairIdentity) -> Result<(), GauntletError> {
        match self.schema_version {
            1 => self.validate_stored_v1(engines),
            _ => Err(GauntletError::InvalidContract {
                reason: "built-in engine profile receipt schema is unsupported".to_owned(),
            }),
        }
    }

    fn stored_semantic_contract_v1(&self) -> crate::runner::SemanticContract {
        let (analyzer_contract_hash, schema_contract_hash) = match self.profile {
            BuiltInEngineProfile::ScalarShipping => (
                BUILT_IN_PROFILE_V1_DEFAULT_ANALYZER_HASH,
                BUILT_IN_PROFILE_V1_DEFAULT_SCHEMA_HASH,
            ),
            BuiltInEngineProfile::ScalarG1a => (
                BUILT_IN_PROFILE_V1_DEFAULT_ANALYZER_HASH,
                BUILT_IN_PROFILE_V1_SCALAR_G1A_SCHEMA_HASH,
            ),
            BuiltInEngineProfile::Cass => (
                BUILT_IN_PROFILE_V1_CASS_ANALYZER_HASH,
                BUILT_IN_PROFILE_V1_CASS_SCHEMA_HASH,
            ),
        };
        crate::runner::SemanticContract {
            analyzer_contract_hash: analyzer_contract_hash.to_owned(),
            schema_contract_hash: schema_contract_hash.to_owned(),
        }
    }

    fn validate_stored_v1(&self, engines: &EnginePairIdentity) -> Result<(), GauntletError> {
        self.subject_config.validate_stored_v1()?;
        let (subject_implementation, subject_hash, oracle_hash) = match self.profile {
            BuiltInEngineProfile::ScalarShipping => (
                "frankensearch-quill/scalar-index",
                self.subject_config.descriptor_hash_v1(),
                BUILT_IN_PROFILE_V1_SCALAR_ORACLE_CONFIG_HASH,
            ),
            BuiltInEngineProfile::ScalarG1a => (
                "frankensearch-quill/scalar-index",
                self.subject_config.descriptor_hash_v1(),
                BUILT_IN_PROFILE_V1_SCALAR_ORACLE_CONFIG_HASH,
            ),
            BuiltInEngineProfile::Cass => (
                "frankensearch-quill/cass-index",
                format!(
                    "cass-semantic-v1:{}",
                    self.subject_config.descriptor_hash_v1()
                ),
                BUILT_IN_PROFILE_V1_CASS_ORACLE_CONFIG_HASH,
            ),
        };
        if self.schema_version != Self::V1_SCHEMA_VERSION
            || engines.comparison_mode != ComparisonMode::CrossEngine
            || engines.subject.implementation != subject_implementation
            || engines.subject.crate_version != BUILT_IN_PROFILE_V1_QUILL_CRATE_VERSION
            || engines.subject.config_hash != subject_hash
            || engines.oracle.implementation != "frankensearch-lexical/tantivy-index"
            || engines.oracle.crate_version != BUILT_IN_PROFILE_V1_LEXICAL_CRATE_VERSION
            || engines.oracle.config_hash != oracle_hash
            || engines.semantic_contract.as_ref() != Some(&self.stored_semantic_contract_v1())
            || engines.subject.source_revision != engines.oracle.source_revision
            || engines.subject.source_dirty != engines.oracle.source_dirty
        {
            return Err(GauntletError::InvalidContract {
                reason: "built-in engine profile receipt does not match its stored adapter identities and semantic contract"
                    .to_owned(),
            });
        }
        validate_recorded_producer_source(
            &engines.subject.source_revision,
            engines.subject.source_dirty,
        )?;
        Ok(())
    }

    fn validate_creation(&self, engines: &EnginePairIdentity) -> Result<(), GauntletError> {
        self.validate_stored(engines)?;
        self.subject_config.validate_creation()?;
        let current_config = self.subject_config.to_config()?;
        let oracle_version = oracle_version_contract()?;
        let (expected_subject_hash, expected_oracle_hash) = match self.profile {
            BuiltInEngineProfile::ScalarShipping | BuiltInEngineProfile::ScalarG1a => (
                quill_config_hash(&current_config),
                TANTIVY_ORACLE_CONFIG_HASH,
            ),
            BuiltInEngineProfile::Cass => (
                format!("cass-semantic-v1:{}", quill_config_hash(&current_config)),
                CASS_TANTIVY_ORACLE_CONFIG_HASH,
            ),
        };
        if engines.subject.crate_version != frankensearch_quill::FRANKENSEARCH_QUILL_CRATE_VERSION
            || engines.subject.config_hash != expected_subject_hash
            || engines.oracle.crate_version != oracle_version.lexical_package_version
            || engines.oracle.config_hash != expected_oracle_hash
            || engines.semantic_contract.as_ref() != Some(&self.current_semantic_contract())
        {
            return Err(GauntletError::InvalidContract {
                reason: "built-in engine profile receipt does not match the current adapter versions and semantic contract"
                    .to_owned(),
            });
        }
        Ok(())
    }
}

/// Subject/oracle pair with mode-specific distinctness validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnginePairIdentity {
    pub comparison_mode: ComparisonMode,
    pub subject: EngineDescriptor,
    pub oracle: EngineDescriptor,
    /// Shared campaign semantic contract declared by both adapters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_contract: Option<crate::runner::SemanticContract>,
    /// Typed evidence lane and complete Quill runtime configuration preimage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    built_in_profile: Option<BuiltInEngineProfileReceipt>,
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
            built_in_profile: None,
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

    pub(crate) fn bind_builtin_profile(
        &mut self,
        receipt: BuiltInEngineProfileReceipt,
    ) -> Result<(), GauntletError> {
        if self.built_in_profile.is_some() {
            return Err(GauntletError::InvalidContract {
                reason: "built-in engine profile receipt may be bound only once".to_owned(),
            });
        }
        receipt.validate_stored(self)?;
        self.built_in_profile = Some(receipt);
        Ok(())
    }

    pub(crate) const fn has_builtin_profile(&self) -> bool {
        self.built_in_profile.is_some()
    }

    /// Revalidate the live adapters against this exact stored identity.
    ///
    /// The built-in profile is part of the identity. Reconstructing a live
    /// pair from only descriptors and semantics would silently discard that
    /// receipt and make every valid built-in campaign fail its mid-run drift
    /// checks. Rebinding the already-validated receipt also proves that it
    /// remains compatible with the observed descriptors.
    pub(crate) fn validate_runtime_state(
        &self,
        subject: EngineDescriptor,
        oracle: EngineDescriptor,
        subject_semantics: &crate::runner::SemanticContract,
        oracle_semantics: &crate::runner::SemanticContract,
    ) -> Result<(), GauntletError> {
        let expected_semantics =
            self.semantic_contract
                .as_ref()
                .ok_or_else(|| GauntletError::InvalidContract {
                    reason: "runtime engine identity is missing its semantic contract".to_owned(),
                })?;
        if subject_semantics != expected_semantics || oracle_semantics != expected_semantics {
            return Err(GauntletError::InvalidContract {
                reason: "engine semantic contract changed during campaign execution".to_owned(),
            });
        }

        let mut observed = Self::new(self.comparison_mode, subject, oracle)?;
        observed.bind_semantic_contract(expected_semantics.clone())?;
        if let Some(receipt) = &self.built_in_profile {
            observed.bind_builtin_profile(receipt.clone())?;
        }
        if &observed != self {
            return Err(GauntletError::InvalidContract {
                reason: "engine identity changed during campaign execution".to_owned(),
            });
        }
        Ok(())
    }

    /// Validate only the identity pair's recorded, engine-neutral invariants.
    ///
    /// Diagnostic adapters may come from a different source revision than the
    /// gauntlet binary, so stored/diagnostic validation must not consult the
    /// currently linked oracle or rewrite their identities.
    pub(crate) fn validate_stored_contract(&self) -> Result<(), GauntletError> {
        let mut rebuilt = Self::new(
            self.comparison_mode,
            self.subject.clone(),
            self.oracle.clone(),
        )?;
        if let Some(semantic_contract) = &self.semantic_contract {
            rebuilt.bind_semantic_contract(semantic_contract.clone())?;
        }
        if let Some(receipt) = &self.built_in_profile {
            rebuilt.bind_builtin_profile(receipt.clone())?;
        }
        if &rebuilt != self {
            return Err(GauntletError::InvalidContract {
                reason: "engine identity is not self-consistent".to_owned(),
            });
        }
        Ok(())
    }

    /// Validate the concrete Quill/Tantivy adapters linked into this producer.
    ///
    /// This creation-time gate is intentionally stronger than diagnostic
    /// validation and may consult the current oracle dependency contract.
    pub(crate) fn validate_builtin_contract(&self) -> Result<(), GauntletError> {
        self.validate_stored_contract()?;
        let Some(receipt) = &self.built_in_profile else {
            return Err(GauntletError::InvalidContract {
                reason: "built-in evidence is missing its typed adapter/profile receipt".to_owned(),
            });
        };
        receipt.validate_creation(self)?;
        for descriptor in [&self.subject, &self.oracle] {
            validate_recorded_producer_source(
                &descriptor.source_revision,
                descriptor.source_dirty,
            )?;
        }
        if self.comparison_mode == ComparisonMode::CrossEngine {
            let oracle_version = oracle_version_contract()?;
            let expected_config_hash = if self.semantic_contract.as_ref()
                == Some(&crate::runner::SemanticContract::cass())
            {
                CASS_TANTIVY_ORACLE_CONFIG_HASH
            } else {
                TANTIVY_ORACLE_CONFIG_HASH
            };
            if self.oracle.implementation != "frankensearch-lexical/tantivy-index"
                || self.oracle.crate_version != oracle_version.lexical_package_version
                || self.subject.crate_version
                    != frankensearch_quill::FRANKENSEARCH_QUILL_CRATE_VERSION
                || self.oracle.source_revision != self.subject.source_revision
                || self.oracle.source_dirty != self.subject.source_dirty
                || self.oracle.config_hash != expected_config_hash
            {
                return Err(GauntletError::InvalidContract {
                    reason: "oracle descriptor does not bind the shared producer and the independently pinned lexical dependency contract"
                        .to_owned(),
                });
            }
        }
        Ok(())
    }
}

/// Engine-neutral query case consumed by both adapters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub struct HarnessRun {
    /// Identity-bearing schema. Decode-only pre-identity runs deserialize to
    /// zero and are rejected before they can be persisted.
    #[serde(default)]
    pub schema_version: u32,
    /// Exact binary identity captured before either adapter was observed.
    #[serde(default)]
    pub producer_build_identity: GauntletProducerBuildIdentity,
    pub engines: EnginePairIdentity,
    pub case: DifferentialCase,
    pub comparator_config: ComparatorConfig,
    pub comparison: ComparisonReport,
}

/// Current serialized harness-run schema.
pub const HARNESS_RUN_SCHEMA_VERSION: u32 = 2;

impl HarnessRun {
    pub(crate) fn validate_diagnostic_creation(&self) -> Result<(), GauntletError> {
        if self.schema_version != HARNESS_RUN_SCHEMA_VERSION {
            return Err(GauntletError::InvalidContract {
                reason: "legacy or unknown harness-run schema cannot produce current evidence"
                    .to_owned(),
            });
        }
        self.producer_build_identity.validate_matches_compiled()?;
        self.engines.validate_stored_contract()?;
        Ok(())
    }

    pub(crate) fn validate_builtin_evidence_creation(&self) -> Result<(), GauntletError> {
        self.validate_diagnostic_creation()?;
        self.producer_build_identity
            .validate_builtin_engines(&self.engines)?;
        self.engines.validate_builtin_contract()?;
        Ok(())
    }
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
        self.run_internal(cx, subject, oracle, case, HarnessAdmission::Diagnostic)
            .await
    }

    #[cfg(test)]
    pub(crate) async fn run_builtin_evidence(
        &self,
        cx: &Cx,
        subject: &dyn GauntletEngine,
        oracle: &dyn GauntletEngine,
        case: &DifferentialCase,
        profile: BuiltInEngineProfileReceipt,
    ) -> Result<HarnessRun, GauntletError> {
        self.run_internal(
            cx,
            subject,
            oracle,
            case,
            HarnessAdmission::BuiltInEvidence(profile),
        )
        .await
    }

    async fn run_internal(
        &self,
        cx: &Cx,
        subject: &dyn GauntletEngine,
        oracle: &dyn GauntletEngine,
        case: &DifferentialCase,
        admission: HarnessAdmission,
    ) -> Result<HarnessRun, GauntletError> {
        let producer_build_identity = GauntletProducerBuildIdentity::compiled()?;
        let engines = EnginePairIdentity::new(
            self.comparison_mode,
            subject.descriptor(),
            oracle.descriptor(),
        )?;
        #[cfg(test)]
        let mut engines = engines;
        #[cfg(test)]
        if let HarnessAdmission::BuiltInEvidence(profile) = admission {
            engines.bind_semantic_contract(profile.current_semantic_contract())?;
            engines.bind_builtin_profile(profile)?;
            producer_build_identity.validate_builtin_engines(&engines)?;
            engines.validate_builtin_contract()?;
        } else {
            engines.validate_stored_contract()?;
        }
        #[cfg(not(test))]
        {
            let _ = admission;
            engines.validate_stored_contract()?;
        }
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
            schema_version: HARNESS_RUN_SCHEMA_VERSION,
            producer_build_identity,
            engines,
            case: case.clone(),
            comparator_config: self.comparator_config,
            comparison,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum HarnessAdmission {
    Diagnostic,
    #[cfg(test)]
    BuiltInEvidence(BuiltInEngineProfileReceipt),
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
    pub fn in_memory(config: QuillConfig) -> Result<Self, GauntletError> {
        let producer = GauntletProducerBuildIdentity::compiled()?;
        Self::in_memory_with_source(
            config,
            producer.source_git_revision,
            producer.source_git_dirty,
        )
    }

    pub(crate) fn in_memory_with_source(
        config: QuillConfig,
        source_revision: impl Into<String>,
        source_dirty: bool,
    ) -> Result<Self, GauntletError> {
        let config_hash = quill_config_hash(&config);
        let descriptor = EngineDescriptor {
            family: EngineFamily::Quill,
            implementation: "frankensearch-quill/scalar-index".to_owned(),
            crate_version: frankensearch_quill::FRANKENSEARCH_QUILL_CRATE_VERSION.to_owned(),
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
    // Preserve the production count-free collector for both the requested
    // page and its expanded tie evidence. Exact count is an independent
    // observation: enabling it must never switch the collector that supplies
    // ranked score bits or native tie keys.
    let observed = index.search_paginated(cx, &case.query, limit, offset, false)?;
    let evidence = index.search_paginated(cx, &case.query, fetch_limit, 0, false)?;
    let count_evidence = index.search_paginated(cx, &case.query, 0, 0, true)?;
    quill_native_observation_from_results(
        &observed,
        &evidence,
        &count_evidence,
        limit,
        offset,
        case.count_requested,
    )
}

fn quill_native_observation_from_results(
    observed: &QuillSearchResult,
    evidence: &QuillSearchResult,
    count_evidence: &QuillSearchResult,
    limit: usize,
    offset: usize,
    count_requested: bool,
) -> Result<EngineObservation, GauntletError> {
    if observed.total_count.is_some() || evidence.total_count.is_some() {
        return Err(GauntletError::InvalidObservation {
            reason: "Quill native ranked observations unexpectedly executed exact-count work"
                .to_owned(),
        });
    }
    if !count_evidence.hits.is_empty() {
        return Err(GauntletError::InvalidObservation {
            reason: "Quill count-only evidence unexpectedly returned ranked hits".to_owned(),
        });
    }
    if observed.doc_count != evidence.doc_count || observed.doc_count != count_evidence.doc_count {
        return Err(GauntletError::InvalidObservation {
            reason: "Quill native ranked and count-only observations disagreed on the committed document count"
                .to_owned(),
        });
    }
    if observed.diagnostics != evidence.diagnostics
        || observed.diagnostics != count_evidence.diagnostics
    {
        return Err(GauntletError::InvalidObservation {
            reason:
                "Quill native ranked and count-only observations disagreed on parser diagnostics"
                    .to_owned(),
        });
    }
    let total_count =
        count_evidence
            .total_count
            .ok_or_else(|| GauntletError::InvalidObservation {
                reason: "Quill count-only evidence omitted its exact count".to_owned(),
            })?;
    let match_count = if count_requested {
        CountState::Value(total_count)
    } else {
        CountState::NotRequested
    };
    quill_observation_from_validated_results(
        observed,
        evidence,
        total_count,
        match_count,
        limit,
        offset,
    )
}

#[cfg(any(feature = "tantivy-oracle", test))]
fn quill_observation_from_results(
    observed: &QuillSearchResult,
    evidence: &QuillSearchResult,
    limit: usize,
    offset: usize,
    count_requested: bool,
) -> Result<EngineObservation, GauntletError> {
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
    quill_observation_from_validated_results(
        observed,
        evidence,
        total_count,
        match_count,
        limit,
        offset,
    )
}

fn quill_observation_from_validated_results(
    observed: &QuillSearchResult,
    evidence: &QuillSearchResult,
    total_count: u64,
    match_count: CountState,
    limit: usize,
    offset: usize,
) -> Result<EngineObservation, GauntletError> {
    let page_end = offset
        .checked_add(limit)
        .ok_or_else(|| GauntletError::InvalidObservation {
            reason: "Quill observation page boundary does not fit usize".to_owned(),
        })?;
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
            reason: "Quill observed and expanded collector pages differ".to_owned(),
        });
    }
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

pub fn quill_config_hash(config: &QuillConfig) -> String {
    QuillConfigReceipt::from_config(config).descriptor_hash_v1()
}

#[cfg(feature = "tantivy-oracle")]
#[derive(Debug)]
struct CassFlushIdentity {
    doc_ord: u32,
    document_id: String,
    content_hash: u64,
}

/// Fresh one-shot Quill subject bound to the durable CASS semantic schema.
#[cfg(feature = "tantivy-oracle")]
pub struct CassQuillSubject {
    config: QuillConfig,
    descriptor: EngineDescriptor,
    parser: frankensearch_quill::CassQueryParser,
    index: Option<frankensearch_quill::index::PreparsedQuillIndex>,
    accumulator: Option<ColumnarAccumulator<CassAnalyzer>>,
    flush_identities: Vec<CassFlushIdentity>,
    encoded_segments: Vec<EncodedSegment>,
    manifest_segments: Vec<ManifestSegment>,
    field_stats: BTreeMap<u16, (u64, u32)>,
    current_lease_base: u64,
    document_count: u64,
    state: QuillCampaignState,
}

#[cfg(feature = "tantivy-oracle")]
impl CassQuillSubject {
    /// Construct a fresh CASS subject without publishing any index state.
    ///
    /// # Errors
    ///
    /// Returns a typed schema, analyzer, parser, configuration, or descriptor
    /// failure before a campaign can claim the adapter.
    pub fn in_memory(config: QuillConfig) -> Result<Self, GauntletError> {
        let producer = GauntletProducerBuildIdentity::compiled()?;
        Self::in_memory_with_source(
            config,
            producer.source_git_revision,
            producer.source_git_dirty,
        )
    }

    pub(crate) fn in_memory_with_source(
        config: QuillConfig,
        source_revision: impl Into<String>,
        source_dirty: bool,
    ) -> Result<Self, GauntletError> {
        let parser =
            frankensearch_quill::CassQueryParser::new(CASS_SEMANTIC_SCHEMA).map_err(|error| {
                GauntletError::InvalidContract {
                    reason: format!("cannot bind the Quill CASS parser: {error}"),
                }
            })?;
        let descriptor = EngineDescriptor {
            family: EngineFamily::Quill,
            implementation: "frankensearch-quill/cass-index".to_owned(),
            crate_version: frankensearch_quill::FRANKENSEARCH_QUILL_CRATE_VERSION.to_owned(),
            source_revision: source_revision.into(),
            source_dirty,
            config_hash: format!("cass-semantic-v1:{}", quill_config_hash(&config)),
        };
        descriptor.validate()?;
        let accumulator =
            ColumnarAccumulator::with_analyzer(CASS_SEMANTIC_SCHEMA, CassAnalyzer::default())
                .map_err(frankensearch_quill::QuillIndexError::from)?;
        Ok(Self {
            config,
            descriptor,
            parser,
            index: None,
            accumulator: Some(accumulator),
            flush_identities: Vec::new(),
            encoded_segments: Vec::new(),
            manifest_segments: Vec::new(),
            field_stats: BTreeMap::new(),
            current_lease_base: 0,
            document_count: 0,
            state: QuillCampaignState::Fresh,
        })
    }

    pub(crate) const fn config(&self) -> &QuillConfig {
        &self.config
    }

    pub(crate) fn claim_fresh_campaign(&mut self) -> Result<(), GauntletError> {
        if self.state != QuillCampaignState::Fresh
            || self.index.is_some()
            || self.document_count != 0
            || !self.encoded_segments.is_empty()
        {
            return Err(GauntletError::InvalidCampaign {
                reason: "Quill CASS subject may execute only one fresh campaign".to_owned(),
            });
        }
        self.state = QuillCampaignState::Ingesting;
        Ok(())
    }

    fn require_ingesting(&self) -> Result<(), GauntletError> {
        if self.state != QuillCampaignState::Ingesting {
            return Err(GauntletError::InvalidCampaign {
                reason: "Quill CASS indexing and commit require an active ingest session"
                    .to_owned(),
            });
        }
        Ok(())
    }

    fn require_committed(&self) -> Result<(), GauntletError> {
        if self.state != QuillCampaignState::Committed {
            return Err(GauntletError::InvalidCampaign {
                reason: "Quill CASS observation requires a committed one-shot snapshot".to_owned(),
            });
        }
        Ok(())
    }

    fn index(&self) -> Result<&frankensearch_quill::index::PreparsedQuillIndex, GauntletError> {
        self.index
            .as_ref()
            .ok_or_else(|| GauntletError::SubjectUnavailable {
                reason: "Quill CASS campaign subject has no committed snapshot".to_owned(),
            })
    }

    fn finish_segment(&mut self) -> Result<(), GauntletError> {
        let accumulator =
            self.accumulator
                .take()
                .ok_or_else(|| GauntletError::InvalidCampaign {
                    reason: "Quill CASS accumulator is unavailable".to_owned(),
                })?;
        if accumulator.document_count() == 0 {
            self.accumulator = Some(accumulator);
            return Ok(());
        }
        let documents = self
            .flush_identities
            .iter()
            .map(|identity| {
                FlushDocumentInput::new(
                    identity.doc_ord,
                    &identity.document_id,
                    identity.content_hash,
                )
            })
            .collect::<Vec<_>>();
        let seal_seq = u64::try_from(self.manifest_segments.len())
            .ok()
            .and_then(|count| count.checked_add(1))
            .ok_or_else(|| GauntletError::InvalidCampaign {
                reason: "Quill CASS seal sequence exhausted".to_owned(),
            })?;
        let segment_id = 0xca55_0000_0000_0000_u64
            .checked_add(seal_seq)
            .ok_or_else(|| GauntletError::InvalidCampaign {
                reason: "Quill CASS segment identity exhausted".to_owned(),
            })?;
        let encoded = flush_accumulator_with_mode(
            &accumulator,
            FlushSegmentInput {
                segment_id,
                lease_docid_base: self.current_lease_base,
                created_unix_s: 0,
                engine_version: CURRENT_ENGINE_VERSION,
                documents: &documents,
            },
            FlushMode::Scalar,
        )
        .map_err(frankensearch_quill::QuillIndexError::from)?;
        let segment_document_count = u32::try_from(accumulator.document_count()).map_err(|_| {
            GauntletError::InvalidCampaign {
                reason: "Quill CASS segment document count does not fit u32".to_owned(),
            }
        })?;
        for field in accumulator.fields() {
            let entry = self.field_stats.entry(field.field_ord()).or_insert((0, 0));
            entry.0 = entry.0.checked_add(field.total_tokens()).ok_or_else(|| {
                GauntletError::InvalidCampaign {
                    reason: "Quill CASS field token count overflow".to_owned(),
                }
            })?;
            entry.1 = entry.1.checked_add(segment_document_count).ok_or_else(|| {
                GauntletError::InvalidCampaign {
                    reason: "Quill CASS field document count overflow".to_owned(),
                }
            })?;
        }
        let header = encoded.header();
        self.manifest_segments.push(ManifestSegment {
            segment_id: header.segment_id,
            seal_seq,
            file_len: encoded.file_len(),
            file_xxh3: encoded.file_xxh3(),
            docid_lo: header.docid_lo,
            docid_hi: header.docid_hi,
            doc_count: header.doc_count,
            tombstones: TombstoneSet::new(),
        });
        self.encoded_segments.push(encoded);
        self.flush_identities.clear();
        self.accumulator = Some(
            ColumnarAccumulator::with_analyzer(CASS_SEMANTIC_SCHEMA, CassAnalyzer::default())
                .map_err(frankensearch_quill::QuillIndexError::from)?,
        );
        Ok(())
    }

    pub(crate) fn index_generated_batch(
        &mut self,
        cx: &Cx,
        documents: &[GeneratedDocument],
    ) -> Result<(), GauntletError> {
        self.require_ingesting()?;
        for document in documents {
            require_active_cx(cx, "Quill CASS indexing")?;
            let local_doc_ord = u32::try_from(self.document_count % u64::from(DOC_ORDS_PER_LEASE))
                .map_err(|_| GauntletError::InvalidCampaign {
                    reason: "Quill CASS local document ordinal does not fit u32".to_owned(),
                })?;
            if local_doc_ord == 0 && self.document_count > 0 {
                self.finish_segment()?;
                self.current_lease_base = self.document_count;
            }
            let cass = document
                .cass
                .as_ref()
                .ok_or_else(|| GauntletError::InvalidCampaign {
                    reason: format!(
                        "CASS campaign document {:?} has no typed CASS fields",
                        document.id
                    ),
                })?;
            let document_id = frankensearch_lexical::cass_compat::cass_document_identity_parts(
                &cass.source_id,
                cass.message_index,
            );
            if document_id.is_empty() || document_id.len() > MAX_DOCUMENT_ID_BYTES {
                return Err(GauntletError::InvalidCampaign {
                    reason: "CASS contract identity is empty or exceeds the document-ID bound"
                        .to_owned(),
                });
            }
            let title_prefix = document
                .title
                .as_deref()
                .map(frankensearch_quill::scribe::cass_generate_edge_ngrams);
            let content_prefix = frankensearch_quill::scribe::cass_generate_edge_ngrams(
                cass_content_prefix_source(&document.content),
            );
            let preview = frankensearch_quill::scribe::cass_build_preview(&document.content, 400);
            let mut indexed = vec![
                IndexedFieldValue::new(0, &cass.agent),
                IndexedFieldValue::new(1, &cass.workspace),
                IndexedFieldValue::new(7, &document.content),
                IndexedFieldValue::new(9, &content_prefix),
                IndexedFieldValue::new(11, &cass.source_id),
                IndexedFieldValue::new(12, &cass.origin_kind),
            ];
            if let Some(title) = document.title.as_deref() {
                indexed.push(IndexedFieldValue::new(6, title));
            }
            if let Some(prefix) = title_prefix.as_deref() {
                indexed.push(IndexedFieldValue::new(8, prefix));
            }
            let numeric = [
                IndexedNumericValue::u64(4, cass.message_index),
                IndexedNumericValue::i64(5, document.created_at_ms),
            ];
            let stored = [
                StoredFieldValue::new(3, cass.source_path.as_bytes()),
                StoredFieldValue::new(10, preview.as_bytes()),
            ];
            self.accumulator
                .as_mut()
                .ok_or_else(|| GauntletError::InvalidCampaign {
                    reason: "Quill CASS accumulator is unavailable".to_owned(),
                })?
                .add_document_with_values(local_doc_ord, &indexed, &numeric, &stored)
                .map_err(frankensearch_quill::QuillIndexError::from)?;
            let canonical = serde_json::to_vec(document)?;
            self.flush_identities.push(CassFlushIdentity {
                doc_ord: local_doc_ord,
                document_id,
                content_hash: xxh3_64(&canonical),
            });
            self.document_count = self.document_count.checked_add(1).ok_or_else(|| {
                GauntletError::InvalidCampaign {
                    reason: "Quill CASS document count overflow".to_owned(),
                }
            })?;
        }
        Ok(())
    }

    pub(crate) fn commit_corpus(&mut self, cx: &Cx) -> Result<usize, GauntletError> {
        self.require_ingesting()?;
        require_active_cx(cx, "Quill CASS commit")?;
        self.finish_segment()?;
        let genesis = KeeperSnapshot::in_memory(CASS_SEMANTIC_SCHEMA)
            .map_err(frankensearch_quill::QuillIndexError::from)?;
        let snapshot = if self.manifest_segments.is_empty() {
            genesis
        } else {
            let mut manifest = genesis
                .next_manifest()
                .map_err(frankensearch_quill::QuillIndexError::from)?;
            manifest.segments = std::mem::take(&mut self.manifest_segments);
            manifest.docid_high_watermark = self
                .current_lease_base
                .checked_add(u64::from(DOC_ORDS_PER_LEASE))
                .ok_or_else(|| GauntletError::InvalidCampaign {
                    reason: "Quill CASS document-ID watermark overflow".to_owned(),
                })?;
            manifest.field_stats = self
                .field_stats
                .iter()
                .map(
                    |(&field_ord, &(total_tokens, doc_count))| ManifestFieldStats {
                        field_ord,
                        total_tokens,
                        doc_count,
                    },
                )
                .collect();
            genesis
                .publish_owned_segments(&manifest, std::mem::take(&mut self.encoded_segments))
                .map_err(frankensearch_quill::QuillIndexError::from)?
        };
        self.index = Some(
            frankensearch_quill::index::PreparsedQuillIndex::from_in_memory_snapshot(
                snapshot,
                self.config.clone(),
            )?,
        );
        self.accumulator = None;
        self.state = QuillCampaignState::Committed;
        usize::try_from(self.document_count).map_err(|_| GauntletError::InvalidCampaign {
            reason: "Quill CASS document count does not fit usize".to_owned(),
        })
    }

    pub(crate) fn observe_cass(
        &self,
        cx: &Cx,
        case: &DifferentialCase,
        filters: &frankensearch_quill::CassQueryFilters,
    ) -> Result<EngineObservation, GauntletError> {
        self.require_committed()?;
        case.validate_shape()?;
        if case.snippet_max_chars.is_some() {
            return Err(GauntletError::InvalidCase {
                reason: "the CASS Quill adapter requires snippets to be disabled".to_owned(),
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
                    reason: "expanded Quill CASS observation window does not fit usize".to_owned(),
                })?;
        let parsed = self.parser.parse(&case.query, filters);
        let mut observed = self.index()?.search_preparsed_paginated(
            cx,
            &parsed.query,
            limit,
            offset,
            case.count_requested,
        )?;
        let mut evidence =
            self.index()?
                .search_preparsed_paginated(cx, &parsed.query, fetch_limit, 0, true)?;
        observed.diagnostics.clone_from(&parsed.diagnostics);
        evidence.diagnostics = parsed.diagnostics;
        quill_observation_from_results(&observed, &evidence, limit, offset, case.count_requested)
    }

    pub(crate) fn descriptor(&self) -> EngineDescriptor {
        self.descriptor.clone()
    }

    pub(crate) fn abort(&mut self) {
        self.state = QuillCampaignState::Aborted;
        self.index = None;
        self.accumulator = None;
        self.flush_identities.clear();
        self.encoded_segments.clear();
        self.manifest_segments.clear();
        self.field_stats.clear();
    }
}

#[cfg(feature = "tantivy-oracle")]
fn cass_content_prefix_source(content: &str) -> &str {
    const MAX_BYTES: usize = 4 * 1024;
    if content.len() <= MAX_BYTES {
        return content;
    }
    let mut boundary = MAX_BYTES;
    while !content.is_char_boundary(boundary) {
        boundary -= 1;
    }
    &content[..boundary]
}

#[cfg(feature = "tantivy-oracle")]
fn require_active_cx(cx: &Cx, phase: &str) -> Result<(), GauntletError> {
    if cx.is_cancel_requested() {
        return Err(frankensearch_core::SearchError::Cancelled {
            phase: phase.to_owned(),
            reason: "gauntlet campaign context requested cancellation".to_owned(),
        }
        .into());
    }
    Ok(())
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
    pub fn in_memory() -> Result<Self, GauntletError> {
        let producer = GauntletProducerBuildIdentity::compiled()?;
        Self::in_memory_with_source(&producer.source_git_revision, producer.source_git_dirty)
    }

    pub(crate) fn in_memory_with_source(
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
    pub fn in_memory_scalar_g1a() -> Result<Self, GauntletError> {
        let producer = GauntletProducerBuildIdentity::compiled()?;
        Self::in_memory_scalar_g1a_with_source(
            &producer.source_git_revision,
            producer.source_git_dirty,
        )
    }

    pub(crate) fn in_memory_scalar_g1a_with_source(
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
    pub fn from_index(index: frankensearch_lexical::TantivyIndex) -> Result<Self, GauntletError> {
        let producer = GauntletProducerBuildIdentity::compiled()?;
        Self::from_index_with_source(
            index,
            &producer.source_git_revision,
            producer.source_git_dirty,
        )
    }

    pub(crate) fn from_index_with_source(
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
        validate_recorded_producer_source(observed_lexical_revision, source_dirty)?;
        Ok(Self {
            index,
            descriptor: EngineDescriptor {
                family: EngineFamily::Tantivy,
                implementation: "frankensearch-lexical/tantivy-index".to_owned(),
                crate_version: contract.lexical_package_version,
                source_revision: observed_lexical_revision.to_owned(),
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
        use frankensearch_core::LexicalWrite;

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

/// Build one committed scalar Quill/Tantivy pair for an external fuzzing run.
///
/// This constructor is feature-gated because the pair is deliberately
/// one-shot: each corpus receives fresh engines, both are committed before a
/// query may execute, and the normal production API does not expose campaign
/// lifecycle controls.
///
/// # Errors
///
/// Returns the first engine construction, indexing, or commit failure.
#[cfg(feature = "fuzz-harness")]
pub async fn scalar_g1a_fuzz_pair(
    cx: &Cx,
    documents: &[frankensearch_core::IndexableDocument],
) -> Result<(QuillSubject, TantivyOracle), GauntletError> {
    use frankensearch_core::LexicalRead;

    let config = QuillConfig {
        deterministic_ingest: true,
        ..QuillConfig::default()
    };
    let mut subject = QuillSubject::in_memory(config)?;
    let mut oracle = TantivyOracle::in_memory_scalar_g1a()?;
    subject.claim_fresh_campaign()?;
    oracle.claim_fresh_campaign()?;
    subject.index_mut()?.index_documents(cx, documents).await?;
    subject.index_mut()?.commit(cx).await?;
    oracle.index().index_documents(cx, documents).await?;
    oracle.index().commit(cx).await?;
    subject.mark_committed()?;
    oracle.mark_committed()?;
    Ok((subject, oracle))
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

#[cfg(feature = "tantivy-oracle")]
fn oracle_observation_from_results(
    observation: frankensearch_lexical::OracleQueryObservation,
    case: &DifferentialCase,
) -> Result<EngineObservation, GauntletError> {
    let limit = usize::try_from(case.limit).map_err(|_| GauntletError::InvalidCase {
        reason: "limit does not fit usize".to_owned(),
    })?;
    let offset = usize::try_from(case.offset).map_err(|_| GauntletError::InvalidCase {
        reason: "offset does not fit usize".to_owned(),
    })?;
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
    let hits = observation
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
        .collect::<Vec<_>>();
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
}

/// Fresh one-shot Tantivy oracle bound to the shipping CASS compatibility path.
#[cfg(feature = "tantivy-oracle")]
pub struct CassTantivyOracle {
    index: frankensearch_lexical::CassTantivyIndex,
    descriptor: EngineDescriptor,
    document_count: usize,
    state: TantivyCampaignState,
}

#[cfg(feature = "tantivy-oracle")]
impl CassTantivyOracle {
    /// Construct a fresh RAM-backed CASS oracle with one deterministic writer.
    ///
    /// # Errors
    ///
    /// Returns a version-contract or Tantivy setup failure before the campaign
    /// can claim the adapter.
    pub fn in_memory() -> Result<Self, GauntletError> {
        let producer = GauntletProducerBuildIdentity::compiled()?;
        Self::in_memory_with_source(&producer.source_git_revision, producer.source_git_dirty)
    }

    pub(crate) fn in_memory_with_source(
        observed_lexical_revision: &str,
        source_dirty: bool,
    ) -> Result<Self, GauntletError> {
        let contract = oracle_version_contract()?;
        validate_recorded_producer_source(observed_lexical_revision, source_dirty)?;
        let descriptor = EngineDescriptor {
            family: EngineFamily::Tantivy,
            implementation: "frankensearch-lexical/tantivy-index".to_owned(),
            crate_version: contract.lexical_package_version,
            source_revision: observed_lexical_revision.to_owned(),
            source_dirty,
            config_hash: CASS_TANTIVY_ORACLE_CONFIG_HASH.to_owned(),
        };
        descriptor.validate()?;
        Ok(Self {
            index: frankensearch_lexical::CassTantivyIndex::in_memory_single_threaded_oracle()?,
            descriptor,
            document_count: 0,
            state: TantivyCampaignState::Fresh,
        })
    }

    pub(crate) fn claim_fresh_campaign(&mut self) -> Result<(), GauntletError> {
        if self.state != TantivyCampaignState::Fresh || self.document_count != 0 {
            return Err(GauntletError::InvalidCampaign {
                reason: "Tantivy CASS oracle may execute only one fresh campaign".to_owned(),
            });
        }
        self.state = TantivyCampaignState::Ingesting;
        Ok(())
    }

    fn require_ingesting(&self) -> Result<(), GauntletError> {
        if self.state != TantivyCampaignState::Ingesting {
            return Err(GauntletError::InvalidCampaign {
                reason: "Tantivy CASS indexing and commit require an active ingest session"
                    .to_owned(),
            });
        }
        Ok(())
    }

    fn require_committed(&self) -> Result<(), GauntletError> {
        if self.state != TantivyCampaignState::Committed {
            return Err(GauntletError::InvalidCampaign {
                reason: "Tantivy CASS observation requires a committed one-shot index".to_owned(),
            });
        }
        Ok(())
    }

    pub(crate) fn index_generated_batch(
        &mut self,
        cx: &Cx,
        documents: &[GeneratedDocument],
    ) -> Result<(), GauntletError> {
        self.require_ingesting()?;
        require_active_cx(cx, "Tantivy CASS indexing")?;
        let mut lowered = Vec::with_capacity(documents.len());
        for document in documents {
            let cass = document
                .cass
                .as_ref()
                .ok_or_else(|| GauntletError::InvalidCampaign {
                    reason: format!(
                        "CASS campaign document {:?} has no typed CASS fields",
                        document.id
                    ),
                })?;
            lowered.push(frankensearch_lexical::CassDocument {
                agent: cass.agent.clone(),
                workspace: Some(cass.workspace.clone()),
                workspace_original: None,
                source_path: cass.source_path.clone(),
                msg_idx: cass.message_index,
                created_at: Some(document.created_at_ms),
                title: document.title.clone(),
                content: document.content.clone(),
                source_id: cass.source_id.clone(),
                origin_kind: cass.origin_kind.clone(),
                origin_host: None,
                conversation_id: None,
            });
        }
        self.index.add_cass_documents(&lowered)?;
        self.document_count = self
            .document_count
            .checked_add(lowered.len())
            .ok_or_else(|| GauntletError::InvalidCampaign {
                reason: "Tantivy CASS document count overflow".to_owned(),
            })?;
        Ok(())
    }

    pub(crate) fn commit_corpus(&mut self, cx: &Cx) -> Result<usize, GauntletError> {
        self.require_ingesting()?;
        require_active_cx(cx, "Tantivy CASS commit")?;
        self.index.commit()?;
        self.state = TantivyCampaignState::Committed;
        Ok(self.document_count)
    }

    pub(crate) fn observe_cass(
        &self,
        cx: &Cx,
        case: &DifferentialCase,
        filters: &frankensearch_lexical::CassQueryFilters,
    ) -> Result<EngineObservation, GauntletError> {
        self.require_committed()?;
        require_active_cx(cx, "Tantivy CASS search")?;
        case.validate_shape()?;
        if case.snippet_max_chars.is_some() {
            return Err(GauntletError::InvalidCase {
                reason: "the CASS Tantivy adapter requires snippets to be disabled".to_owned(),
            });
        }
        let limit = usize::try_from(case.limit).map_err(|_| GauntletError::InvalidCase {
            reason: "limit does not fit usize".to_owned(),
        })?;
        let offset = usize::try_from(case.offset).map_err(|_| GauntletError::InvalidCase {
            reason: "offset does not fit usize".to_owned(),
        })?;
        let fetch_limit = offset
            .checked_add(limit)
            .ok_or_else(|| GauntletError::InvalidCase {
                reason: "offset plus limit does not fit usize".to_owned(),
            })?;
        let tie_expansion_limit =
            usize::try_from(case.tie_expansion_limit).map_err(|_| GauntletError::InvalidCase {
                reason: "tie expansion limit does not fit usize".to_owned(),
            })?;
        let observation = self.index.cass_oracle_observe_query(
            &case.query,
            filters,
            fetch_limit,
            tie_expansion_limit,
        )?;
        oracle_observation_from_results(observation, case)
    }

    pub(crate) fn descriptor(&self) -> EngineDescriptor {
        self.descriptor.clone()
    }

    pub(crate) fn abort(&mut self) {
        self.state = TantivyCampaignState::Aborted;
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
    // Every `DivergenceClass` reference in this module sits inside a
    // `perf-harness` block, so the import carries the same gate.
    #[cfg(feature = "perf-harness")]
    use crate::comparator::DivergenceClass;

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

    #[cfg(feature = "perf-harness")]
    const QG_POSITIONLESS_FIELDS: [FieldDescriptor; 5] = [
        FieldDescriptor {
            id: 0,
            name: "id",
            kind: FieldKind::Keyword,
            stored: true,
        },
        FieldDescriptor {
            id: 1,
            name: "content",
            kind: FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault,
                positions: false,
            },
            stored: true,
        },
        FieldDescriptor {
            id: 2,
            name: "title",
            kind: FieldKind::Text {
                analyzer: Analyzer::FrankensearchDefault,
                positions: false,
            },
            stored: true,
        },
        FieldDescriptor {
            id: 3,
            name: "metadata_json",
            kind: FieldKind::StoredOnly,
            stored: true,
        },
        FieldDescriptor {
            id: 4,
            name: "ord",
            kind: FieldKind::U64 {
                indexed: false,
                fast: true,
            },
            stored: true,
        },
    ];

    #[cfg(feature = "perf-harness")]
    const QG_POSITIONLESS_SCHEMA: SchemaDescriptor = SchemaDescriptor {
        name: "frankensearch-default-no-positions-v1",
        fields: &QG_POSITIONLESS_FIELDS,
    };

    fn test_producer_source() -> (String, bool) {
        let identity = GauntletProducerBuildIdentity::compiled().expect("compiled producer");
        (identity.source_git_revision, identity.source_git_dirty)
    }

    fn test_scalar_g1a_profile() -> BuiltInEngineProfileReceipt {
        BuiltInEngineProfileReceipt::new(BuiltInEngineProfile::ScalarG1a, &QuillConfig::default())
    }

    fn stored_profile_pair(
        profile: BuiltInEngineProfile,
        config: &QuillConfig,
    ) -> EnginePairIdentity {
        let receipt = BuiltInEngineProfileReceipt::new(profile, config);
        let semantic_contract = receipt.stored_semantic_contract_v1();
        let (subject_implementation, subject_config_hash, oracle_config_hash) = match profile {
            BuiltInEngineProfile::ScalarShipping | BuiltInEngineProfile::ScalarG1a => (
                "frankensearch-quill/scalar-index",
                receipt.subject_config.descriptor_hash_v1(),
                BUILT_IN_PROFILE_V1_SCALAR_ORACLE_CONFIG_HASH,
            ),
            BuiltInEngineProfile::Cass => (
                "frankensearch-quill/cass-index",
                format!(
                    "cass-semantic-v1:{}",
                    receipt.subject_config.descriptor_hash_v1()
                ),
                BUILT_IN_PROFILE_V1_CASS_ORACLE_CONFIG_HASH,
            ),
        };
        let producer_revision = "a".repeat(40);
        let mut pair = EnginePairIdentity::new(
            ComparisonMode::CrossEngine,
            EngineDescriptor {
                family: EngineFamily::Quill,
                implementation: subject_implementation.to_owned(),
                crate_version: BUILT_IN_PROFILE_V1_QUILL_CRATE_VERSION.to_owned(),
                source_revision: producer_revision.clone(),
                source_dirty: false,
                config_hash: subject_config_hash,
            },
            EngineDescriptor {
                family: EngineFamily::Tantivy,
                implementation: "frankensearch-lexical/tantivy-index".to_owned(),
                crate_version: BUILT_IN_PROFILE_V1_LEXICAL_CRATE_VERSION.to_owned(),
                source_revision: producer_revision,
                source_dirty: false,
                config_hash: oracle_config_hash.to_owned(),
            },
        )
        .expect("frozen v1 profile pair");
        pair.bind_semantic_contract(semantic_contract)
            .expect("frozen v1 semantic contract");
        pair.bind_builtin_profile(receipt)
            .expect("frozen v1 profile receipt");
        pair
    }

    #[cfg(feature = "perf-harness")]
    fn qg_position_mode_subject(positions: bool) -> QuillSubject {
        qg_position_mode_subject_with_config(positions, e55_config())
    }

    #[cfg(feature = "perf-harness")]
    fn qg_position_mode_subject_with_config(positions: bool, config: QuillConfig) -> QuillSubject {
        let (producer_revision, producer_dirty) = test_producer_source();
        let schema = if positions {
            frankensearch_quill::DEFAULT_SCHEMA
        } else {
            QG_POSITIONLESS_SCHEMA
        };
        let descriptor = EngineDescriptor {
            family: EngineFamily::Quill,
            implementation: "frankensearch-quill/scalar-index".to_owned(),
            crate_version: frankensearch_quill::FRANKENSEARCH_QUILL_CRATE_VERSION.to_owned(),
            source_revision: producer_revision,
            source_dirty: producer_dirty,
            config_hash: format!(
                "{}-positions_{}",
                quill_config_hash(&config),
                if positions { "on" } else { "off" }
            ),
        };
        descriptor
            .validate()
            .expect("QG position-mode subject descriptor");
        QuillSubject {
            index: Some(
                QuillIndex::in_memory_with_schema(schema, config.clone())
                    .expect("QG position-mode Quill index"),
            ),
            config,
            descriptor,
            state: QuillCampaignState::Fresh,
        }
    }

    #[cfg(feature = "perf-harness")]
    fn qg_position_mode_oracle(positions: bool) -> TantivyOracle {
        let (revision, dirty) = test_producer_source();
        let index = frankensearch_lexical::TantivyIndex::in_memory_with_benchmark_config(
            50_000_000, 1, positions,
        )
        .expect("QG position-mode Tantivy index");
        TantivyOracle::from_index_with_campaign_freshness(
            index,
            &revision,
            dirty,
            true,
            crate::runner::SemanticContract::scalar_g1a(),
        )
        .expect("QG position-mode Tantivy oracle")
    }

    /// E6.3 law runner. Each caller supplies its generator identity, while
    /// every returned observable is projected through the normal differential
    /// harness so this does not compare an invented side channel.
    #[cfg(feature = "perf-harness")]
    async fn e63_observations(
        cx: &Cx,
        documents: &[frankensearch_core::IndexableDocument],
        cases: &[(&str, &str)],
        seed: u64,
        generator_id: &str,
    ) -> Vec<(String, EngineObservation, EngineObservation)> {
        e63_observations_with_config(cx, documents, cases, seed, generator_id, e55_config()).await
    }

    #[cfg(feature = "perf-harness")]
    async fn e63_observations_with_config(
        cx: &Cx,
        documents: &[frankensearch_core::IndexableDocument],
        cases: &[(&str, &str)],
        seed: u64,
        generator_id: &str,
        subject_config: QuillConfig,
    ) -> Vec<(String, EngineObservation, EngineObservation)> {
        e63_observations_with_config_and_batch_size(
            cx,
            documents,
            cases,
            seed,
            generator_id,
            subject_config,
            documents.len(),
        )
        .await
    }

    /// E6.3 observation runner variant that fixes the ingest batch schedule.
    /// A non-zero batch size lets lifecycle laws exercise publication
    /// boundaries without changing the corpus or query projection.
    #[cfg(feature = "perf-harness")]
    async fn e63_observations_with_config_and_batch_size(
        cx: &Cx,
        documents: &[frankensearch_core::IndexableDocument],
        cases: &[(&str, &str)],
        seed: u64,
        generator_id: &str,
        subject_config: QuillConfig,
        batch_size: usize,
    ) -> Vec<(String, EngineObservation, EngineObservation)> {
        assert!(batch_size > 0, "E6.3 ingest batch size must be non-zero");
        let mut subject = qg_position_mode_subject_with_config(true, subject_config);
        let mut oracle = qg_position_mode_oracle(true);
        subject
            .claim_fresh_campaign()
            .expect("E6.3 claim Quill input-order campaign");
        for batch in documents.chunks(batch_size) {
            subject
                .index_mut()
                .expect("E6.3 open Quill input-order campaign")
                .index_documents(cx, batch)
                .await
                .expect("E6.3 index Quill input-order fixture");
        }
        subject
            .index_mut()
            .expect("E6.3 open Quill input-order campaign")
            .commit(cx)
            .await
            .expect("E6.3 commit Quill input-order fixture");
        subject
            .mark_committed()
            .expect("E6.3 publish Quill input-order campaign");

        oracle
            .claim_fresh_campaign()
            .expect("E6.3 claim Tantivy input-order campaign");
        oracle
            .index_documents(cx, documents)
            .await
            .expect("E6.3 index Tantivy input-order fixture");
        oracle
            .mark_committed()
            .expect("E6.3 publish Tantivy input-order campaign");

        let harness = DifferentialHarness::default();
        let mut observations = Vec::with_capacity(cases.len());
        for &(case_id, query) in cases {
            let mut case = DifferentialCase::new(format!("e63-{case_id}"), query, 16);
            case.snippet_max_chars = None;
            case.tie_expansion_limit = 64;
            case.metadata.generator_id = Some(generator_id.to_owned());
            case.metadata.generator_seed = Some(seed);
            let run = harness
                .run(cx, &subject, &oracle, &case)
                .await
                .unwrap_or_else(|error| panic!("E6.3 case {case_id} failed: {error}"));
            assert_eq!(
                run.comparison.status,
                ComparisonStatus::Exact,
                "E6.3 cross-engine case {case_id}: {:?}",
                run.comparison.divergences,
            );
            assert_eq!(
                run.comparison.rank_class,
                RankClass::RankExact,
                "E6.3 cross-engine case {case_id}: {:?}",
                run.comparison.divergences,
            );
            observations.push((
                case_id.to_owned(),
                run.comparison.subject,
                run.comparison.oracle,
            ));
        }
        observations
    }

    #[cfg(feature = "perf-harness")]
    fn e63_seeded_input_permutation(len: usize, seed: u64) -> Vec<usize> {
        let mut permutation = (0..len).collect::<Vec<_>>();
        let mut state = seed;
        for position in (1..len).rev() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let width = u64::try_from(position + 1).expect("E6.3 permutation width fits u64");
            let selected =
                usize::try_from(state % width).expect("E6.3 permutation index fits usize");
            permutation.swap(position, selected);
        }
        permutation
    }

    /// The replay identity includes the exact v1 LCG and Fisher-Yates swap
    /// schedule. Determinism alone would not detect a changed generator that
    /// silently maps a historical seed to a different ingest order.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_input_permutation_seed_schedule_v1_is_exact_and_preserves_small_domains() {
        assert_eq!(e63_seeded_input_permutation(0, 0), Vec::<usize>::new());
        assert_eq!(e63_seeded_input_permutation(1, 0), vec![0]);
        assert_eq!(
            e63_seeded_input_permutation(5, 0xe63_1a00_5eed_0001),
            vec![4, 1, 0, 3, 2]
        );
    }

    #[cfg(feature = "perf-harness")]
    fn e63_seeded_ascii_query_normalization(term: &str, seed: u64) -> String {
        match seed % 3 {
            0 => format!(" \t{term}\n"),
            1 => term.to_ascii_uppercase(),
            _ => format!("\t{}  ", term.to_ascii_uppercase()),
        }
    }

    /// Pin the versioned query-normalization seed mapping used in replay
    /// artifacts. The campaign already proves each transformed query is
    /// equivalent, but a remapped seed would otherwise reproduce a different
    /// transform under the same historical identity.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_ascii_query_normalization_seed_schedule_v1_is_exact_and_periodic() {
        assert_eq!(
            e63_seeded_ascii_query_normalization("alpha", 0),
            " \talpha\n"
        );
        assert_eq!(e63_seeded_ascii_query_normalization("alpha", 1), "ALPHA");
        assert_eq!(
            e63_seeded_ascii_query_normalization("alpha", 2),
            "\tALPHA  "
        );
        assert_eq!(
            e63_seeded_ascii_query_normalization("alpha", 3),
            " \talpha\n"
        );
        assert_eq!(
            e63_seeded_ascii_query_normalization("alpha", u64::MAX),
            " \talpha\n"
        );
    }

    #[cfg(feature = "perf-harness")]
    fn e63_seeded_flush_batch_size(seed: u64) -> usize {
        match seed % 3 {
            0 => 1,
            1 => 2,
            _ => 3,
        }
    }

    /// The v1 flush-batch generator is a three-state replay contract, not
    /// merely a non-zero batch-size source. Pinning every residue and the
    /// period wrap prevents a changed schedule from silently relabelling a
    /// historical E6.3 seed.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_flush_batch_seed_schedule_v1_is_exact_and_periodic() {
        assert_eq!(e63_seeded_flush_batch_size(0), 1);
        assert_eq!(e63_seeded_flush_batch_size(1), 2);
        assert_eq!(e63_seeded_flush_batch_size(2), 3);
        assert_eq!(e63_seeded_flush_batch_size(3), 1);
        assert_eq!(e63_seeded_flush_batch_size(u64::MAX), 1);
    }

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

    struct ExactDiagnosticEngine {
        descriptor: EngineDescriptor,
        observe_calls: Arc<AtomicUsize>,
    }

    impl GauntletEngine for ExactDiagnosticEngine {
        fn descriptor(&self) -> EngineDescriptor {
            self.descriptor.clone()
        }

        fn observe<'a>(&'a self, _cx: &'a Cx, _case: &'a DifferentialCase) -> GauntletFuture<'a> {
            Box::pin(async move {
                self.observe_calls.fetch_add(1, Ordering::Relaxed);
                Ok(EngineObservation {
                    hits: Vec::new(),
                    cutoff_tie_group: Vec::new(),
                    cutoff_tie_complete: true,
                    offset_tie_group: Vec::new(),
                    offset_tie_complete: false,
                    snippets: BTreeMap::new(),
                    match_count: CountState::Value(0),
                    doc_count: 0,
                    ast_differences: Vec::new(),
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
            QuillSubject::in_memory_with_source(QuillConfig::default(), "test-revision", false)
                .expect("live Quill subject"),
        );
        assert_eq!(subject.descriptor().family, EngineFamily::Quill);
        assert_eq!(subject.descriptor().config_hash.len(), 64);
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
    fn built_in_profile_v1_accepts_every_frozen_lane_and_current_creation_policy() {
        for profile in [
            BuiltInEngineProfile::ScalarShipping,
            BuiltInEngineProfile::ScalarG1a,
            BuiltInEngineProfile::Cass,
        ] {
            let pair = stored_profile_pair(profile, &QuillConfig::default());
            pair.validate_stored_contract()
                .expect("frozen receipt must validate without mutable current policy");
            pair.validate_builtin_contract()
                .expect("current adapters must remain compatible with receipt v1");
        }
    }

    #[test]
    fn quill_config_receipt_v1_rejects_every_invalid_boundary() {
        let baseline = QuillConfigReceipt::from_config(&QuillConfig::default());
        macro_rules! reject_receipt_mutation {
            ($label:literal, $mutation:expr) => {{
                let mut candidate = baseline.clone();
                $mutation(&mut candidate);
                assert!(
                    candidate.validate_stored_v1().is_err(),
                    "stored receipt accepted invalid {}",
                    $label
                );
            }};
        }

        reject_receipt_mutation!("schema version", |receipt: &mut QuillConfigReceipt| {
            receipt.schema_version = 2;
        });
        reject_receipt_mutation!("scribe budget", |receipt: &mut QuillConfigReceipt| {
            receipt.scribe_shard_budget_bytes = 0;
        });
        reject_receipt_mutation!("delta budget", |receipt: &mut QuillConfigReceipt| {
            receipt.delta_budget_bytes = 0;
        });
        reject_receipt_mutation!("tier fanout", |receipt: &mut QuillConfigReceipt| {
            receipt.tier_fanout = 1;
        });
        reject_receipt_mutation!("small tier width", |receipt: &mut QuillConfigReceipt| {
            receipt.tier_small_max_docid_width = 0;
        });
        reject_receipt_mutation!("tier ordering", |receipt: &mut QuillConfigReceipt| {
            receipt.tier_medium_max_docid_width = receipt.tier_small_max_docid_width;
        });
        reject_receipt_mutation!("bulk cadence", |receipt: &mut QuillConfigReceipt| {
            receipt.bulk_publish_segment_cadence = 0;
        });
        reject_receipt_mutation!(
            "zero compaction density",
            |receipt: &mut QuillConfigReceipt| {
                receipt.compaction_tombstone_density_bits = 0.0_f64.to_bits();
            }
        );
        reject_receipt_mutation!(
            "NaN compaction density",
            |receipt: &mut QuillConfigReceipt| {
                receipt.compaction_tombstone_density_bits = f64::NAN.to_bits();
            }
        );
        reject_receipt_mutation!(
            "large compaction density",
            |receipt: &mut QuillConfigReceipt| {
                receipt.compaction_tombstone_density_bits = 1.1_f64.to_bits();
            }
        );
        reject_receipt_mutation!(
            "NaN merge hole ratio",
            |receipt: &mut QuillConfigReceipt| {
                receipt.merge_max_hole_ratio_bits = f64::NAN.to_bits();
            }
        );
        reject_receipt_mutation!(
            "large merge hole ratio",
            |receipt: &mut QuillConfigReceipt| {
                receipt.merge_max_hole_ratio_bits = 1.1_f64.to_bits();
            }
        );
        reject_receipt_mutation!(
            "negative-zero merge hole ratio",
            |receipt: &mut QuillConfigReceipt| {
                receipt.merge_max_hole_ratio_bits = (-0.0_f64).to_bits();
            }
        );
        reject_receipt_mutation!("glob limit", |receipt: &mut QuillConfigReceipt| {
            receipt.glob_expansion_limit = 0;
        });
        reject_receipt_mutation!("query fuel", |receipt: &mut QuillConfigReceipt| {
            receipt.query_fuel_budget = 0;
        });
        reject_receipt_mutation!("ingest shards", |receipt: &mut QuillConfigReceipt| {
            receipt.max_ingest_shards = 0;
        });
        reject_receipt_mutation!("visibility lag", |receipt: &mut QuillConfigReceipt| {
            receipt.max_visibility_lag_ms = 0;
        });
    }

    #[test]
    fn built_in_profile_v1_rejects_every_bound_identity_mutation() {
        let baseline =
            stored_profile_pair(BuiltInEngineProfile::ScalarG1a, &QuillConfig::default());
        macro_rules! reject_pair_mutation {
            ($label:literal, $mutation:expr) => {{
                let mut candidate = baseline.clone();
                $mutation(&mut candidate);
                assert!(
                    candidate.validate_stored_contract().is_err(),
                    "stored profile accepted mutated {}",
                    $label
                );
            }};
        }

        reject_pair_mutation!("comparison mode", |pair: &mut EnginePairIdentity| {
            pair.comparison_mode = ComparisonMode::InternalDifferential;
        });
        reject_pair_mutation!("subject family", |pair: &mut EnginePairIdentity| {
            pair.subject.family = EngineFamily::Tantivy;
        });
        reject_pair_mutation!("subject implementation", |pair: &mut EnginePairIdentity| {
            pair.subject.implementation = "frankensearch-quill/cass-index".to_owned();
        });
        reject_pair_mutation!("subject crate version", |pair: &mut EnginePairIdentity| {
            pair.subject.crate_version = "0.2.2".to_owned();
        });
        reject_pair_mutation!("subject config hash", |pair: &mut EnginePairIdentity| {
            pair.subject.config_hash = "mutated".to_owned();
        });
        reject_pair_mutation!("oracle family", |pair: &mut EnginePairIdentity| {
            pair.oracle.family = EngineFamily::Quill;
        });
        reject_pair_mutation!("oracle implementation", |pair: &mut EnginePairIdentity| {
            pair.oracle.implementation = "tantivy/direct".to_owned();
        });
        reject_pair_mutation!("oracle crate version", |pair: &mut EnginePairIdentity| {
            pair.oracle.crate_version = "0.2.2".to_owned();
        });
        reject_pair_mutation!("oracle config hash", |pair: &mut EnginePairIdentity| {
            pair.oracle.config_hash = "mutated".to_owned();
        });
        reject_pair_mutation!(
            "producer revision mismatch",
            |pair: &mut EnginePairIdentity| {
                pair.oracle.source_revision = "b".repeat(40);
            }
        );
        reject_pair_mutation!(
            "producer dirty mismatch",
            |pair: &mut EnginePairIdentity| {
                pair.oracle.source_dirty = true;
            }
        );
        reject_pair_mutation!(
            "invalid shared producer",
            |pair: &mut EnginePairIdentity| {
                pair.subject.source_revision = "not-a-git-revision".to_owned();
                pair.oracle.source_revision = "not-a-git-revision".to_owned();
            }
        );
        reject_pair_mutation!("semantic contract", |pair: &mut EnginePairIdentity| {
            pair.semantic_contract = Some(crate::runner::SemanticContract::shipping_default());
        });
        reject_pair_mutation!("profile schema", |pair: &mut EnginePairIdentity| {
            pair.built_in_profile
                .as_mut()
                .expect("profile")
                .schema_version = 2;
        });
        reject_pair_mutation!("profile kind", |pair: &mut EnginePairIdentity| {
            pair.built_in_profile.as_mut().expect("profile").profile = BuiltInEngineProfile::Cass;
        });
        reject_pair_mutation!("profile config", |pair: &mut EnginePairIdentity| {
            pair.built_in_profile
                .as_mut()
                .expect("profile")
                .subject_config
                .query_fuel_budget += 1;
        });
        let mut missing_profile = baseline.clone();
        missing_profile.built_in_profile = None;
        assert!(
            missing_profile.validate_stored_contract().is_ok(),
            "engine-neutral stored identities may omit built-in execution authority"
        );
        assert!(matches!(
            missing_profile.validate_builtin_contract(),
            Err(GauntletError::InvalidContract { ref reason })
                if reason.contains("missing its typed adapter/profile receipt")
        ));
        reject_pair_mutation!(
            "missing semantic contract",
            |pair: &mut EnginePairIdentity| {
                pair.semantic_contract = None;
            }
        );

        let encoded = serde_json::to_value(
            baseline
                .built_in_profile
                .as_ref()
                .expect("baseline profile"),
        )
        .expect("serialize profile receipt");
        let mut unknown_field = encoded;
        unknown_field
            .as_object_mut()
            .expect("receipt object")
            .insert("unknown".to_owned(), serde_json::json!(true));
        assert!(
            serde_json::from_value::<BuiltInEngineProfileReceipt>(unknown_field).is_err(),
            "profile receipt must reject unknown fields"
        );
    }

    #[test]
    fn literal_engine_profile_v1_fixture_pins_archival_bytes_preimage_and_hash() {
        const PROFILE_JSON: &str = "{\"schema_version\":1,\"profile\":\"scalar_g1a\",\"subject_config\":{\"schema_version\":1,\"scribe_shard_budget_bytes\":123,\"delta_budget_bytes\":456,\"tier_fanout\":3,\"tier_small_max_docid_width\":7,\"tier_medium_max_docid_width\":9,\"bulk_load_mode\":true,\"bulk_publish_segment_cadence\":11,\"compaction_tombstone_density_bits\":4598175219545276416,\"merge_max_hole_ratio_bits\":4604930618986332160,\"glob_expansion_limit\":13,\"query_fuel_budget\":17,\"max_ingest_shards\":19,\"deterministic_ingest\":true,\"max_visibility_lag_ms\":23,\"quarantine_on_unrepairable\":true}}";
        const CONFIG_PREIMAGE: &str = "quill-config-v1;scribe=123;delta=456;fanout=3;tier_small=7;tier_medium=9;bulk=true;bulk_cadence=11;compact=3fd0000000000000;holes=3fe8000000000000;glob=13;fuel=17;shards=19;deterministic=true;visibility_ms=23;quarantine=true";
        const CONFIG_SHA256: &str =
            "514677086bef61af511a70172bbabc1996fc3e4d933653c21f2e127f7c463c44";

        let fixture_bytes = include_bytes!("../fixtures/engine-pair-profile-v1.json");
        assert_eq!(fixture_bytes.last(), Some(&b'\n'));
        let pair: EnginePairIdentity =
            serde_json::from_slice(fixture_bytes).expect("literal frozen v1 engine-pair fixture");
        assert_eq!(
            serde_json::to_vec(&pair).expect("canonical engine-pair JSON"),
            fixture_bytes
                .strip_suffix(b"\n")
                .expect("fixture ends in exactly one LF"),
            "the entire frozen engine-pair receipt must remain byte-for-byte canonical",
        );
        pair.validate_stored_contract()
            .expect("literal v1 bytes remain archive-valid");
        pair.validate_builtin_contract()
            .expect("current creation policy remains compatible with v1");
        let profile = pair.built_in_profile.as_ref().expect("profile receipt");
        assert_eq!(
            serde_json::to_string(profile).expect("profile JSON"),
            PROFILE_JSON
        );
        assert_eq!(
            profile.subject_config.canonical_preimage_v1(),
            CONFIG_PREIMAGE
        );
        assert_eq!(profile.subject_config.descriptor_hash_v1(), CONFIG_SHA256);

        let mut nested_unknown: serde_json::Value =
            serde_json::from_slice(fixture_bytes).expect("fixture JSON value");
        nested_unknown["built_in_profile"]["subject_config"]
            .as_object_mut()
            .expect("subject config object")
            .insert("unknown".to_owned(), serde_json::json!(true));
        assert!(
            serde_json::from_value::<EnginePairIdentity>(nested_unknown).is_err(),
            "nested receipt fields must fail closed"
        );
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
    fn cass_identity_requires_the_cass_oracle_config_hash() {
        let version = oracle_version_contract().expect("oracle version contract");
        let producer_revision = "a".repeat(40);
        let subject_config = QuillConfig::default();
        let subject = EngineDescriptor {
            family: EngineFamily::Quill,
            implementation: "frankensearch-quill/cass-index".to_owned(),
            crate_version: frankensearch_quill::FRANKENSEARCH_QUILL_CRATE_VERSION.to_owned(),
            source_revision: producer_revision.clone(),
            source_dirty: false,
            config_hash: format!("cass-semantic-v1:{}", quill_config_hash(&subject_config)),
        };
        let oracle = EngineDescriptor {
            family: EngineFamily::Tantivy,
            implementation: "frankensearch-lexical/tantivy-index".to_owned(),
            crate_version: version.lexical_package_version,
            source_revision: producer_revision,
            source_dirty: false,
            config_hash: CASS_TANTIVY_ORACLE_CONFIG_HASH.to_owned(),
        };
        let mut pair =
            EnginePairIdentity::new(ComparisonMode::CrossEngine, subject, oracle.clone())
                .expect("well-shaped cross-engine identity");
        pair.bind_semantic_contract(crate::runner::SemanticContract::cass())
            .expect("CASS semantic contract");
        pair.bind_builtin_profile(BuiltInEngineProfileReceipt::new(
            BuiltInEngineProfile::Cass,
            &subject_config,
        ))
        .expect("CASS profile receipt");
        pair.validate_builtin_contract()
            .expect("the exact CASS oracle identity is admissible");

        pair.oracle = EngineDescriptor {
            config_hash: TANTIVY_ORACLE_CONFIG_HASH.to_owned(),
            ..oracle.clone()
        };
        assert!(matches!(
            pair.validate_builtin_contract(),
            Err(GauntletError::InvalidContract { .. })
        ));

        pair.oracle = oracle;
        pair.validate_builtin_contract()
            .expect("CASS oracle identity clears only with the CASS config hash");
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
    fn contract_guard_rejects_before_engine_execution() {
        let observe_calls = Arc::new(AtomicUsize::new(0));
        let (producer_revision, producer_dirty) = test_producer_source();
        let subject = CountingEngine {
            descriptor: EngineDescriptor {
                family: EngineFamily::Quill,
                implementation: "counting-quill".to_owned(),
                crate_version: frankensearch_quill::FRANKENSEARCH_QUILL_CRATE_VERSION.to_owned(),
                source_revision: producer_revision.clone(),
                source_dirty: producer_dirty,
                config_hash: "quill-config".to_owned(),
            },
            observe_calls: Arc::clone(&observe_calls),
        };
        let oracle = CountingEngine {
            descriptor: EngineDescriptor {
                family: EngineFamily::Tantivy,
                implementation: "frankensearch-lexical/tantivy-index".to_owned(),
                crate_version: oracle_version_contract()
                    .expect("oracle version contract")
                    .lexical_package_version,
                source_revision: producer_revision,
                source_dirty: producer_dirty,
                config_hash: "hostile-wrong-oracle-config".to_owned(),
            },
            observe_calls: Arc::clone(&observe_calls),
        };
        let harness = DifferentialHarness::default();
        let case = DifferentialCase::new("contract-preflight", "anything", 10);

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            assert!(matches!(
                harness
                    .run_builtin_evidence(&cx, &subject, &oracle, &case, test_scalar_g1a_profile(),)
                    .await,
                Err(GauntletError::InvalidContract { .. })
            ));
        });
        assert_eq!(observe_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn independent_engine_revisions_are_diagnostic_but_cannot_be_promoted() {
        let observe_calls = Arc::new(AtomicUsize::new(0));
        let subject = ExactDiagnosticEngine {
            descriptor: EngineDescriptor {
                family: EngineFamily::Quill,
                implementation: "external-quill-diagnostic".to_owned(),
                crate_version: "9.1.0".to_owned(),
                source_revision: "external-subject-revision".to_owned(),
                source_dirty: true,
                config_hash: "external-subject-config".to_owned(),
            },
            observe_calls: Arc::clone(&observe_calls),
        };
        let oracle = ExactDiagnosticEngine {
            descriptor: EngineDescriptor {
                family: EngineFamily::Tantivy,
                implementation: "external-oracle-diagnostic".to_owned(),
                crate_version: "7.2.0".to_owned(),
                source_revision: "independent-oracle-revision".to_owned(),
                source_dirty: false,
                config_hash: "external-oracle-config".to_owned(),
            },
            observe_calls: Arc::clone(&observe_calls),
        };
        let harness = DifferentialHarness::default();
        let case = DifferentialCase::new("independent-diagnostic", "", 0);

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let run = harness
                .run(&cx, &subject, &oracle, &case)
                .await
                .expect("independent adapters remain usable as diagnostics");
            assert_eq!(
                run.engines.subject.source_revision,
                "external-subject-revision"
            );
            assert_eq!(
                run.engines.oracle.source_revision,
                "independent-oracle-revision"
            );
            assert_eq!(run.comparison.status, crate::ComparisonStatus::Exact);
            assert_eq!(observe_calls.load(Ordering::Relaxed), 2);

            let error = harness
                .run_builtin_evidence(&cx, &subject, &oracle, &case, test_scalar_g1a_profile())
                .await
                .expect_err("diagnostic adapters cannot be promoted to built-in evidence");
            assert!(matches!(error, GauntletError::InvalidContract { .. }));
            assert_eq!(
                observe_calls.load(Ordering::Relaxed),
                2,
                "promotion rejection must happen before either adapter executes again"
            );
        });
    }

    #[test]
    fn producer_guard_rejects_a_canonical_fabrication_before_engine_execution() {
        let observe_calls = Arc::new(AtomicUsize::new(0));
        let compiled = GauntletProducerBuildIdentity::compiled().expect("compiled producer");
        let fabricated_revision = if compiled.source_git_revision == "f".repeat(40) {
            "e".repeat(40)
        } else {
            "f".repeat(40)
        };
        let subject = CountingEngine {
            descriptor: EngineDescriptor {
                family: EngineFamily::Quill,
                implementation: "counting-quill".to_owned(),
                crate_version: frankensearch_quill::FRANKENSEARCH_QUILL_CRATE_VERSION.to_owned(),
                source_revision: fabricated_revision.clone(),
                source_dirty: false,
                config_hash: "quill-config".to_owned(),
            },
            observe_calls: Arc::clone(&observe_calls),
        };
        let oracle = CountingEngine {
            descriptor: EngineDescriptor {
                family: EngineFamily::Tantivy,
                implementation: "frankensearch-lexical/tantivy-index".to_owned(),
                crate_version: oracle_version_contract()
                    .expect("oracle version contract")
                    .lexical_package_version,
                source_revision: fabricated_revision,
                source_dirty: false,
                config_hash: TANTIVY_ORACLE_CONFIG_HASH.to_owned(),
            },
            observe_calls: Arc::clone(&observe_calls),
        };
        let harness = DifferentialHarness::default();
        let case = DifferentialCase::new("producer-preflight", "anything", 10);

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            assert!(matches!(
                harness
                    .run_builtin_evidence(&cx, &subject, &oracle, &case, test_scalar_g1a_profile(),)
                    .await,
                Err(GauntletError::InvalidContract { .. })
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
        let expected_reason = "Quill observed and expanded collector pages differ";

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
    fn quill_native_observation_keeps_ranked_scores_separate_from_exact_count() {
        let native_score = f32::from_bits(0x4005_5fc7);
        let ranked = QuillSearchResult {
            hits: vec![frankensearch_quill::QuillHit {
                document_id: "test-cooking-015".to_owned(),
                global_docid: 15,
                score: native_score,
            }],
            total_count: None,
            doc_count: 120,
            diagnostics: Vec::new(),
        };
        let count_evidence = QuillSearchResult {
            hits: Vec::new(),
            total_count: Some(110),
            doc_count: 120,
            diagnostics: Vec::new(),
        };

        let counted =
            quill_native_observation_from_results(&ranked, &ranked, &count_evidence, 1, 0, true)
                .expect("native ranking plus independent count evidence");
        assert_eq!(counted.hits[0].score_bits, native_score.to_bits());
        assert_eq!(counted.match_count, CountState::Value(110));

        let count_free =
            quill_native_observation_from_results(&ranked, &ranked, &count_evidence, 1, 0, false)
                .expect("internal count evidence stays hidden for a count-free case");
        assert_eq!(count_free.hits[0].score_bits, native_score.to_bits());
        assert_eq!(count_free.match_count, CountState::NotRequested);
    }

    #[test]
    fn quill_native_observation_rejects_collector_contract_mixups() {
        let ranked = QuillSearchResult {
            hits: vec![frankensearch_quill::QuillHit {
                document_id: "winner".to_owned(),
                global_docid: 7,
                score: 3.5,
            }],
            total_count: None,
            doc_count: 3,
            diagnostics: Vec::new(),
        };
        let count_evidence = QuillSearchResult {
            hits: Vec::new(),
            total_count: Some(1),
            doc_count: 3,
            diagnostics: Vec::new(),
        };

        let mut counted_ranked = ranked.clone();
        counted_ranked.total_count = Some(1);
        assert!(matches!(
            quill_native_observation_from_results(
                &counted_ranked,
                &ranked,
                &count_evidence,
                1,
                0,
                true,
            ),
            Err(GauntletError::InvalidObservation { reason })
                if reason == "Quill native ranked observations unexpectedly executed exact-count work"
        ));

        let mut scored_count = count_evidence.clone();
        scored_count.hits.clone_from(&ranked.hits);
        assert!(matches!(
            quill_native_observation_from_results(
                &ranked,
                &ranked,
                &scored_count,
                1,
                0,
                true,
            ),
            Err(GauntletError::InvalidObservation { reason })
                if reason == "Quill count-only evidence unexpectedly returned ranked hits"
        ));

        let mut wrong_doc_count = count_evidence.clone();
        wrong_doc_count.doc_count = 4;
        assert!(matches!(
            quill_native_observation_from_results(
                &ranked,
                &ranked,
                &wrong_doc_count,
                1,
                0,
                true,
            ),
            Err(GauntletError::InvalidObservation { reason })
                if reason == "Quill native ranked and count-only observations disagreed on the committed document count"
        ));

        let mut wrong_diagnostics = count_evidence.clone();
        wrong_diagnostics
            .diagnostics
            .push(frankensearch_quill::QueryDiagnostic {
                kind: frankensearch_quill::QueryDiagnosticKind::SyntaxRecovery,
                message: "different parser result".to_owned(),
                byte_offset: None,
                fragment: None,
            });
        assert!(matches!(
            quill_native_observation_from_results(
                &ranked,
                &ranked,
                &wrong_diagnostics,
                1,
                0,
                true,
            ),
            Err(GauntletError::InvalidObservation { reason })
                if reason == "Quill native ranked and count-only observations disagreed on parser diagnostics"
        ));

        let mut missing_count = count_evidence;
        missing_count.total_count = None;
        assert!(matches!(
            quill_native_observation_from_results(
                &ranked,
                &ranked,
                &missing_count,
                1,
                0,
                true,
            ),
            Err(GauntletError::InvalidObservation { reason })
                if reason == "Quill count-only evidence omitted its exact count"
        ));
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
            .lexical_contract_audit_revision;
        let first = TantivyOracle::in_memory_with_source(&revision, false).expect("first oracle");
        let second = TantivyOracle::in_memory_with_source(&revision, false).expect("second oracle");
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
            .lexical_contract_audit_revision;
        let mut oracle = TantivyOracle::in_memory_with_source(&revision, false).expect("oracle");
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
            .lexical_contract_audit_revision;
        let mut oracle = TantivyOracle::in_memory_with_source(&revision, false).expect("oracle");
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

    /// QG position-free cells are a schema mode, not a different query
    /// language. Before release measurement, both engines must construct the
    /// mode and preserve the registered exact law for operators that never
    /// consume a position list. Quill's separate capability tests cover the
    /// typed `PositionsRequired` failure for phrases.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn qg_position_modes_are_cross_engine_exact_for_position_independent_queries() {
        use frankensearch_core::IndexableDocument;

        let documents = vec![
            IndexableDocument::new("doc-1", "alpha beta beta").with_title("guide"),
            IndexableDocument::new("doc-2", "alpha gamma").with_title("alpha overview"),
            IndexableDocument::new("doc-3", "beta gamma gamma gamma").with_title("alpha"),
            IndexableDocument::new("doc-4", "alpha beta gamma delta").with_title("reference"),
            IndexableDocument::new("doc-5", "delta epsilon").with_title("quiet"),
        ];
        let queries = [
            ("bare-term", "alpha"),
            ("repeated-term", "beta"),
            ("boolean-and", "alpha AND beta"),
            ("boolean-or", "alpha OR gamma"),
            ("fielded-term", "title:alpha"),
        ];
        type PositionModeEvidence = Vec<(String, Vec<(String, u32)>, CountState)>;

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let mut quill_mode_evidence = Vec::new();
            let mut tantivy_mode_evidence = Vec::new();

            for positions in [true, false] {
                let mode = if positions {
                    "positions_on"
                } else {
                    "positions_off"
                };
                let mut subject = qg_position_mode_subject(positions);
                let mut oracle = qg_position_mode_oracle(positions);

                subject
                    .claim_fresh_campaign()
                    .unwrap_or_else(|error| panic!("{mode}: claim Quill campaign: {error}"));
                subject
                    .index_mut()
                    .unwrap_or_else(|error| panic!("{mode}: open Quill campaign: {error}"))
                    .index_documents(&cx, &documents)
                    .await
                    .unwrap_or_else(|error| panic!("{mode}: index Quill fixture: {error}"));
                subject
                    .index_mut()
                    .unwrap_or_else(|error| panic!("{mode}: open Quill campaign: {error}"))
                    .commit(&cx)
                    .await
                    .unwrap_or_else(|error| panic!("{mode}: commit Quill fixture: {error}"));
                subject
                    .mark_committed()
                    .unwrap_or_else(|error| panic!("{mode}: publish Quill campaign: {error}"));

                oracle
                    .claim_fresh_campaign()
                    .unwrap_or_else(|error| panic!("{mode}: claim Tantivy campaign: {error}"));
                oracle
                    .index_documents(&cx, &documents)
                    .await
                    .unwrap_or_else(|error| panic!("{mode}: index Tantivy fixture: {error}"));
                oracle
                    .mark_committed()
                    .unwrap_or_else(|error| panic!("{mode}: publish Tantivy campaign: {error}"));

                let harness = DifferentialHarness::default();
                let mut quill_queries = Vec::new();
                let mut tantivy_queries = Vec::new();
                for (query_id, query) in queries {
                    let mut case =
                        DifferentialCase::new(format!("qg-{mode}-{query_id}"), query, 16);
                    case.snippet_max_chars = None;
                    case.tie_expansion_limit = 64;
                    let run = harness
                        .run(&cx, &subject, &oracle, &case)
                        .await
                        .unwrap_or_else(|error| {
                            panic!("{mode}/{query_id}: cross-engine observation failed: {error}")
                        });
                    assert_eq!(
                        run.comparison.status,
                        ComparisonStatus::Exact,
                        "{mode}/{query_id}: {:?}",
                        run.comparison.divergences,
                    );
                    assert_eq!(
                        run.comparison.rank_class,
                        RankClass::RankExact,
                        "{mode}/{query_id}: {:?}",
                        run.comparison.divergences,
                    );

                    let summarize = |observation: &EngineObservation| {
                        (
                            query_id.to_owned(),
                            observation
                                .hits
                                .iter()
                                .map(|hit| (hit.doc_id.clone(), hit.score_bits))
                                .collect::<Vec<_>>(),
                            observation.match_count,
                        )
                    };
                    quill_queries.push(summarize(&run.comparison.subject));
                    tantivy_queries.push(summarize(&run.comparison.oracle));
                }
                quill_mode_evidence.push(quill_queries);
                tantivy_mode_evidence.push(tantivy_queries);
            }

            let stable_membership = |evidence: &PositionModeEvidence| {
                evidence
                    .iter()
                    .map(|(query_id, hits, count)| {
                        let mut doc_ids = hits
                            .iter()
                            .map(|(doc_id, _score_bits)| doc_id.clone())
                            .collect::<Vec<_>>();
                        doc_ids.sort();
                        (query_id.clone(), doc_ids, *count)
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                stable_membership(&quill_mode_evidence[0]),
                stable_membership(&quill_mode_evidence[1]),
                "Quill position-independent membership or counts changed with positions",
            );
            assert_eq!(
                stable_membership(&tantivy_mode_evidence[0]),
                stable_membership(&tantivy_mode_evidence[1]),
                "Tantivy position-independent membership or counts changed with positions",
            );

            let repeated_term_scores = |evidence: &PositionModeEvidence| {
                evidence
                    .iter()
                    .find(|(query_id, _, _)| query_id == "repeated-term")
                    .map(|(_, hits, _)| {
                        let mut score_bits_by_doc = hits
                            .iter()
                            .map(|(doc_id, score_bits)| (doc_id.clone(), *score_bits))
                            .collect::<Vec<_>>();
                        score_bits_by_doc.sort_by(|left, right| left.0.cmp(&right.0));
                        score_bits_by_doc
                    })
                    .expect("repeated-term evidence")
            };
            assert_ne!(
                repeated_term_scores(&quill_mode_evidence[0]),
                repeated_term_scores(&quill_mode_evidence[1]),
                "Quill positioned fields retain frequencies while Basic fields clamp tf to one",
            );
            assert_ne!(
                repeated_term_scores(&tantivy_mode_evidence[0]),
                repeated_term_scores(&tantivy_mode_evidence[1]),
                "Tantivy positioned fields retain frequencies while Basic fields clamp tf to one",
            );
        });
    }

    /// E6.3 law: with stable external document IDs, ingest order is not an
    /// observable part of scalar lexical semantics. Equal-score ties may use
    /// engine-local document addresses, so the law explicitly permits only a
    /// tie-order classification. The negative fixture keeps the IDs and order
    /// transform but changes one document's content, proving that the law does
    /// not accept an arbitrary corpus rewrite.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_seeded_input_order_permutation_is_exact_and_content_mutation_is_not() {
        use frankensearch_core::IndexableDocument;

        const SEED: u64 = 0xe63_1a00_5eed_0001;
        let canonical = vec![
            IndexableDocument::new("doc-1", "alpha beta beta").with_title("guide"),
            IndexableDocument::new("doc-2", "alpha gamma").with_title("alpha overview"),
            IndexableDocument::new("doc-3", "beta gamma gamma gamma").with_title("alpha"),
            IndexableDocument::new("doc-4", "alpha beta gamma delta").with_title("reference"),
            IndexableDocument::new("doc-5", "delta epsilon").with_title("quiet"),
        ];
        let queries = [
            ("bare-term", "alpha"),
            ("repeated-term", "beta"),
            ("boolean-and", "alpha AND beta"),
            ("fielded-term", "title:alpha"),
            ("negative-sentinel", "saffron"),
        ];
        let permutation = e63_seeded_input_permutation(canonical.len(), SEED);
        assert_ne!(
            permutation,
            (0..canonical.len()).collect::<Vec<_>>(),
            "E6.3 seed must exercise a real ingest-order transform",
        );
        assert_eq!(
            permutation,
            e63_seeded_input_permutation(canonical.len(), SEED),
            "E6.3 seed must replay byte-identically",
        );
        let permuted = permutation
            .iter()
            .map(|&index| canonical[index].clone())
            .collect::<Vec<_>>();
        let mut content_mutated = permuted.clone();
        content_mutated[permutation
            .iter()
            .position(|&index| index == 3)
            .expect("E6.3 permutation retains doc-4")] =
            IndexableDocument::new("doc-4", "alpha beta saffron").with_title("reference");

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let baseline = e63_observations(
                &cx,
                &canonical,
                &queries,
                SEED,
                "e6.3-input-order-permutation-v1",
            )
            .await;
            let transformed = e63_observations(
                &cx,
                &permuted,
                &queries,
                SEED,
                "e6.3-input-order-permutation-v1",
            )
            .await;
            let invalid = e63_observations(
                &cx,
                &content_mutated,
                &queries,
                SEED,
                "e6.3-input-order-permutation-v1",
            )
            .await;

            for (
                (baseline_id, baseline_quill, baseline_tantivy),
                (transformed_id, transformed_quill, transformed_tantivy),
            ) in baseline.iter().zip(&transformed)
            {
                assert_eq!(
                    baseline_id, transformed_id,
                    "E6.3 replay case identity drifted"
                );
                for (engine, before, after) in [
                    ("Quill", baseline_quill, transformed_quill),
                    ("Tantivy", baseline_tantivy, transformed_tantivy),
                ] {
                    let comparison = compare_observations(
                        before.clone(),
                        after.clone(),
                        ComparatorConfig::default(),
                    )
                    .unwrap_or_else(|error| {
                        panic!("E6.3 {engine} {baseline_id} permutation comparison failed: {error}")
                    });
                    assert!(
                        matches!(
                            comparison.rank_class,
                            RankClass::RankExact | RankClass::TieOrder
                        ) && comparison
                            .divergences
                            .iter()
                            .all(|divergence| divergence.class == DivergenceClass::TieOrder),
                        "E6.3 {engine} {baseline_id} produced a non-tie divergence under an input-order-only transform: {:?}",
                        comparison.divergences,
                    );
                }
            }

            let baseline_sentinel = baseline
                .iter()
                .find(|(case_id, _, _)| case_id == "negative-sentinel")
                .expect("E6.3 baseline negative fixture");
            let invalid_sentinel = invalid
                .iter()
                .find(|(case_id, _, _)| case_id == "negative-sentinel")
                .expect("E6.3 invalid negative fixture");
            for (engine, before, after) in [
                ("Quill", &baseline_sentinel.1, &invalid_sentinel.1),
                ("Tantivy", &baseline_sentinel.2, &invalid_sentinel.2),
            ] {
                let comparison = compare_observations(
                    before.clone(),
                    after.clone(),
                    ComparatorConfig::default(),
                )
                .unwrap_or_else(|error| {
                    panic!("E6.3 {engine} invalid-transform comparison failed: {error}")
                });
                assert_eq!(
                    comparison.status,
                    ComparisonStatus::Failed,
                    "E6.3 {engine} incorrectly accepted a content mutation as an input-order permutation",
                );
            }
        });
    }

    /// E6.3 seeded property campaign for the qualified input-order law. Each
    /// seed must replay exactly and exercise a nonidentity stable-ID
    /// permutation; the paired single-seed law supplies the intentionally
    /// invalid content-mutation control. This stays bounded for PR execution
    /// while making the generator rather than one hand-picked permutation the
    /// unit under test.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_input_order_seed_matrix_replays_live_observations() {
        use frankensearch_core::IndexableDocument;

        const SEEDS: [u64; 3] = [
            0xe63_1a00_5eed_0001,
            0xe63_1a00_5eed_0002,
            0xe63_1a00_5eed_0003,
        ];
        let canonical = vec![
            IndexableDocument::new("doc-1", "alpha beta beta").with_title("guide"),
            IndexableDocument::new("doc-2", "alpha gamma").with_title("alpha overview"),
            IndexableDocument::new("doc-3", "beta gamma gamma gamma").with_title("alpha"),
            IndexableDocument::new("doc-4", "alpha beta gamma delta").with_title("reference"),
            IndexableDocument::new("doc-5", "delta epsilon").with_title("quiet"),
        ];
        let queries = [("bare-term", "alpha"), ("boolean-and", "alpha AND beta")];

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for seed in SEEDS {
                let permutation = e63_seeded_input_permutation(canonical.len(), seed);
                assert_ne!(
                    permutation,
                    (0..canonical.len()).collect::<Vec<_>>(),
                    "E6.3 seed {seed:#x} must exercise a real ingest-order transform",
                );
                assert_eq!(
                    permutation,
                    e63_seeded_input_permutation(canonical.len(), seed),
                    "E6.3 seed {seed:#x} must replay byte-identically",
                );
                let permuted = permutation
                    .iter()
                    .map(|&index| canonical[index].clone())
                    .collect::<Vec<_>>();
                let baseline = e63_observations(
                    &cx,
                    &canonical,
                    &queries,
                    seed,
                    "e6.3-input-order-permutation-v1",
                )
                .await;
                let transformed = e63_observations(
                    &cx,
                    &permuted,
                    &queries,
                    seed,
                    "e6.3-input-order-permutation-v1",
                )
                .await;

                for (
                    (baseline_id, baseline_quill, baseline_tantivy),
                    (transformed_id, transformed_quill, transformed_tantivy),
                ) in baseline.iter().zip(&transformed)
                {
                    assert_eq!(
                        baseline_id, transformed_id,
                        "E6.3 seed {seed:#x} replay case identity drifted"
                    );
                    for (engine, before, after) in [
                        ("Quill", baseline_quill, transformed_quill),
                        ("Tantivy", baseline_tantivy, transformed_tantivy),
                    ] {
                        let comparison = compare_observations(
                            before.clone(),
                            after.clone(),
                            ComparatorConfig::default(),
                        )
                        .unwrap_or_else(|error| {
                            panic!("E6.3 {engine} seed {seed:#x} {baseline_id} replay comparison failed: {error}")
                        });
                        assert!(
                            matches!(
                                comparison.rank_class,
                                RankClass::RankExact | RankClass::TieOrder
                            ) && comparison
                                .divergences
                                .iter()
                                .all(|divergence| divergence.class == DivergenceClass::TieOrder),
                            "E6.3 {engine} seed {seed:#x} {baseline_id} produced a non-tie divergence: {:?}",
                            comparison.divergences,
                        );
                    }
                }
            }
        });
    }

    /// E6.3 law: segment boundaries are an implementation detail when the
    /// corpus, stable IDs, and scalar configuration contract are unchanged.
    /// The tight budget below is intentionally small enough to exercise the
    /// flush/segment path; it does not change analyzer, scoring, or query
    /// policy. Changing a document payload is the invalid control, so this
    /// does not disguise a corpus mutation as a geometry perturbation.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_tight_segment_geometry_preserves_observations_but_content_mutation_does_not() {
        use frankensearch_core::IndexableDocument;

        const SEED: u64 = 0xe63_5e90_5eed_0001;
        let documents = vec![
            IndexableDocument::new("doc-1", "alpha beta beta").with_title("guide"),
            IndexableDocument::new("doc-2", "alpha gamma").with_title("alpha overview"),
            IndexableDocument::new("doc-3", "beta gamma gamma gamma").with_title("alpha"),
            IndexableDocument::new("doc-4", "alpha beta gamma delta").with_title("reference"),
            IndexableDocument::new("doc-5", "delta epsilon").with_title("quiet"),
        ];
        let queries = [
            ("bare-term", "alpha"),
            ("repeated-term", "beta"),
            ("boolean-and", "alpha AND beta"),
            ("negative-sentinel", "saffron"),
        ];
        let tight_geometry = QuillConfig {
            scribe_shard_budget_bytes: 1,
            delta_budget_bytes: 1,
            tier_fanout: 2,
            ..e55_config()
        };
        let mut content_mutated = documents.clone();
        content_mutated[3] =
            IndexableDocument::new("doc-4", "alpha beta saffron").with_title("reference");

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let baseline = e63_observations(
                &cx,
                &documents,
                &queries,
                SEED,
                "e6.3-tight-segment-geometry-v1",
            )
            .await;
            let transformed = e63_observations_with_config(
                &cx,
                &documents,
                &queries,
                SEED,
                "e6.3-tight-segment-geometry-v1",
                tight_geometry,
            )
            .await;
            let invalid = e63_observations_with_config(
                &cx,
                &content_mutated,
                &queries,
                SEED,
                "e6.3-tight-segment-geometry-v1",
                e55_config(),
            )
            .await;

            for (
                (baseline_id, baseline_quill, baseline_tantivy),
                (geometry_id, geometry_quill, geometry_tantivy),
            ) in baseline.iter().zip(&transformed)
            {
                assert_eq!(
                    baseline_id, geometry_id,
                    "E6.3 geometry case identity drifted"
                );
                for (engine, before, after) in [
                    ("Quill", baseline_quill, geometry_quill),
                    ("Tantivy", baseline_tantivy, geometry_tantivy),
                ] {
                    let comparison = compare_observations(
                        before.clone(),
                        after.clone(),
                        ComparatorConfig::default(),
                    )
                    .unwrap_or_else(|error| {
                        panic!("E6.3 {engine} {baseline_id} geometry comparison failed: {error}")
                    });
                    assert!(
                        matches!(
                            comparison.rank_class,
                            RankClass::RankExact | RankClass::TieOrder
                        ) && comparison
                            .divergences
                            .iter()
                            .all(|divergence| divergence.class == DivergenceClass::TieOrder),
                        "E6.3 {engine} {baseline_id} produced a non-tie divergence under tight segment geometry: {:?}",
                        comparison.divergences,
                    );
                }
            }

            let baseline_sentinel = baseline
                .iter()
                .find(|(case_id, _, _)| case_id == "negative-sentinel")
                .expect("E6.3 baseline geometry negative fixture");
            let invalid_sentinel = invalid
                .iter()
                .find(|(case_id, _, _)| case_id == "negative-sentinel")
                .expect("E6.3 invalid geometry negative fixture");
            for (engine, before, after) in [
                ("Quill", &baseline_sentinel.1, &invalid_sentinel.1),
                ("Tantivy", &baseline_sentinel.2, &invalid_sentinel.2),
            ] {
                let comparison = compare_observations(
                    before.clone(),
                    after.clone(),
                    ComparatorConfig::default(),
                )
                .unwrap_or_else(|error| {
                    panic!("E6.3 {engine} invalid geometry comparison failed: {error}")
                });
                assert_eq!(
                    comparison.status,
                    ComparisonStatus::Failed,
                    "E6.3 {engine} incorrectly accepted a content mutation as segment geometry",
                );
            }
        });
    }

    /// E6.3 law: bulk publication cadence changes only the intermediate
    /// manifest schedule. With the same stable-ID corpus and scalar contract,
    /// it must not change the committed query observation. Both arms use bulk
    /// mode and identical tight shard budgets so the only transformed setting
    /// is the cadence. A content mutation remains the invalid control.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_bulk_publish_cadence_preserves_observations_but_content_mutation_does_not() {
        use frankensearch_core::IndexableDocument;

        const SEED: u64 = 0xe63_b011_5eed_0001;
        let documents = vec![
            IndexableDocument::new("doc-1", "alpha beta beta").with_title("guide"),
            IndexableDocument::new("doc-2", "alpha gamma").with_title("alpha overview"),
            IndexableDocument::new("doc-3", "beta gamma gamma gamma").with_title("alpha"),
            IndexableDocument::new("doc-4", "alpha beta gamma delta").with_title("reference"),
            IndexableDocument::new("doc-5", "delta epsilon").with_title("quiet"),
        ];
        let queries = [
            ("bare-term", "alpha"),
            ("repeated-term", "beta"),
            ("boolean-and", "alpha AND beta"),
            ("negative-sentinel", "saffron"),
        ];
        let baseline_config = QuillConfig {
            scribe_shard_budget_bytes: 1,
            delta_budget_bytes: 1,
            tier_fanout: 2,
            bulk_load_mode: true,
            bulk_publish_segment_cadence: 1,
            ..e55_config()
        };
        let transformed_config = QuillConfig {
            bulk_publish_segment_cadence: 3,
            ..baseline_config.clone()
        };
        let mut content_mutated = documents.clone();
        content_mutated[3] =
            IndexableDocument::new("doc-4", "alpha beta saffron").with_title("reference");

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let baseline = e63_observations_with_config_and_batch_size(
                &cx,
                &documents,
                &queries,
                SEED,
                "e6.3-bulk-publish-cadence-v1",
                baseline_config,
                1,
            )
            .await;
            let transformed = e63_observations_with_config_and_batch_size(
                &cx,
                &documents,
                &queries,
                SEED,
                "e6.3-bulk-publish-cadence-v1",
                transformed_config,
                1,
            )
            .await;
            let invalid = e63_observations_with_config_and_batch_size(
                &cx,
                &content_mutated,
                &queries,
                SEED,
                "e6.3-bulk-publish-cadence-v1",
                QuillConfig {
                    scribe_shard_budget_bytes: 1,
                    delta_budget_bytes: 1,
                    tier_fanout: 2,
                    bulk_load_mode: true,
                    bulk_publish_segment_cadence: 1,
                    ..e55_config()
                },
                1,
            )
            .await;

            for (
                (baseline_id, baseline_quill, baseline_tantivy),
                (transformed_id, transformed_quill, transformed_tantivy),
            ) in baseline.iter().zip(&transformed)
            {
                assert_eq!(
                    baseline_id, transformed_id,
                    "E6.3 bulk cadence case identity drifted"
                );
                for (engine, before, after) in [
                    ("Quill", baseline_quill, transformed_quill),
                    ("Tantivy", baseline_tantivy, transformed_tantivy),
                ] {
                    let comparison = compare_observations(
                        before.clone(),
                        after.clone(),
                        ComparatorConfig::default(),
                    )
                    .unwrap_or_else(|error| {
                        panic!(
                            "E6.3 {engine} {baseline_id} bulk cadence comparison failed: {error}"
                        )
                    });
                    assert!(
                        matches!(
                            comparison.rank_class,
                            RankClass::RankExact | RankClass::TieOrder
                        ) && comparison
                            .divergences
                            .iter()
                            .all(|divergence| divergence.class == DivergenceClass::TieOrder),
                        "E6.3 {engine} {baseline_id} produced a non-tie divergence under bulk cadence: {:?}",
                        comparison.divergences,
                    );
                }
            }

            let baseline_sentinel = baseline
                .iter()
                .find(|(case_id, _, _)| case_id == "negative-sentinel")
                .expect("E6.3 baseline bulk cadence negative fixture");
            let invalid_sentinel = invalid
                .iter()
                .find(|(case_id, _, _)| case_id == "negative-sentinel")
                .expect("E6.3 invalid bulk cadence negative fixture");
            for (engine, before, after) in [
                ("Quill", &baseline_sentinel.1, &invalid_sentinel.1),
                ("Tantivy", &baseline_sentinel.2, &invalid_sentinel.2),
            ] {
                let comparison = compare_observations(
                    before.clone(),
                    after.clone(),
                    ComparatorConfig::default(),
                )
                .unwrap_or_else(|error| {
                    panic!("E6.3 {engine} invalid bulk cadence comparison failed: {error}")
                });
                assert_eq!(
                    comparison.status,
                    ComparisonStatus::Failed,
                    "E6.3 {engine} incorrectly accepted a content mutation as bulk cadence",
                );
            }
        });
    }

    /// E6.3 seeded property campaign for the qualified flush-batch schedule
    /// law. A tight segment geometry makes every batch boundary observable to
    /// the writer lifecycle, while the final corpus, stable IDs, query policy,
    /// and scalar scoring contract stay fixed. The batch schedule is generated
    /// from a replayable seed rather than selected per fixture. A changed
    /// document payload remains an intentionally invalid control for each arm.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_flush_batch_seed_matrix_preserves_observations_but_content_mutation_does_not() {
        use frankensearch_core::IndexableDocument;

        const SEEDS: [u64; 3] = [
            0xe63_f1a5_5eed_0001,
            0xe63_f1a5_5eed_0002,
            0xe63_f1a5_5eed_0003,
        ];
        let documents = vec![
            IndexableDocument::new("doc-1", "alpha beta beta").with_title("guide"),
            IndexableDocument::new("doc-2", "alpha gamma").with_title("alpha overview"),
            IndexableDocument::new("doc-3", "beta gamma gamma gamma").with_title("alpha"),
            IndexableDocument::new("doc-4", "alpha beta gamma delta").with_title("reference"),
            IndexableDocument::new("doc-5", "delta epsilon").with_title("quiet"),
        ];
        let queries = [
            ("bare-term", "alpha"),
            ("boolean-and", "alpha AND beta"),
            ("negative-sentinel", "saffron"),
        ];
        let tight_geometry = QuillConfig {
            scribe_shard_budget_bytes: 1,
            delta_budget_bytes: 1,
            tier_fanout: 2,
            ..e55_config()
        };
        let mut content_mutated = documents.clone();
        content_mutated[3] =
            IndexableDocument::new("doc-4", "alpha beta saffron").with_title("reference");

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for seed in SEEDS {
                let batch_size = e63_seeded_flush_batch_size(seed);
                assert_ne!(
                    batch_size,
                    documents.len(),
                    "E6.3 seed {seed:#x} must exercise a real flush-batch transform",
                );
                assert_eq!(
                    batch_size,
                    e63_seeded_flush_batch_size(seed),
                    "E6.3 seed {seed:#x} must replay its batch schedule byte-identically",
                );
                let baseline = e63_observations_with_config_and_batch_size(
                    &cx,
                    &documents,
                    &queries,
                    seed,
                    "e6.3-flush-batch-schedule-v1",
                    tight_geometry.clone(),
                    documents.len(),
                )
                .await;
                let transformed = e63_observations_with_config_and_batch_size(
                    &cx,
                    &documents,
                    &queries,
                    seed,
                    "e6.3-flush-batch-schedule-v1",
                    tight_geometry.clone(),
                    batch_size,
                )
                .await;
                let invalid = e63_observations_with_config_and_batch_size(
                    &cx,
                    &content_mutated,
                    &queries,
                    seed,
                    "e6.3-flush-batch-schedule-v1",
                    tight_geometry.clone(),
                    batch_size,
                )
                .await;

                for (
                    (baseline_id, baseline_quill, baseline_tantivy),
                    (transformed_id, transformed_quill, transformed_tantivy),
                ) in baseline.iter().zip(&transformed)
                {
                    assert_eq!(
                        baseline_id, transformed_id,
                        "E6.3 seed {seed:#x} flush-batch case identity drifted"
                    );
                    for (engine, before, after) in [
                        ("Quill", baseline_quill, transformed_quill),
                        ("Tantivy", baseline_tantivy, transformed_tantivy),
                    ] {
                        let comparison = compare_observations(
                            before.clone(),
                            after.clone(),
                            ComparatorConfig::default(),
                        )
                        .unwrap_or_else(|error| {
                            panic!(
                                "E6.3 {engine} seed {seed:#x} {baseline_id} flush-batch comparison failed: {error}"
                            )
                        });
                        assert!(
                            matches!(
                                comparison.rank_class,
                                RankClass::RankExact | RankClass::TieOrder
                            ) && comparison
                                .divergences
                                .iter()
                                .all(|divergence| divergence.class == DivergenceClass::TieOrder),
                            "E6.3 {engine} seed {seed:#x} {baseline_id} produced a non-tie divergence under flush batching: {:?}",
                            comparison.divergences,
                        );
                    }
                }

                let baseline_sentinel = baseline
                    .iter()
                    .find(|(case_id, _, _)| case_id == "negative-sentinel")
                    .expect("E6.3 baseline flush-batch negative fixture");
                let invalid_sentinel = invalid
                    .iter()
                    .find(|(case_id, _, _)| case_id == "negative-sentinel")
                    .expect("E6.3 invalid flush-batch negative fixture");
                for (engine, before, after) in [
                    ("Quill", &baseline_sentinel.1, &invalid_sentinel.1),
                    ("Tantivy", &baseline_sentinel.2, &invalid_sentinel.2),
                ] {
                    let comparison = compare_observations(
                        before.clone(),
                        after.clone(),
                        ComparatorConfig::default(),
                    )
                    .unwrap_or_else(|error| {
                        panic!(
                            "E6.3 {engine} seed {seed:#x} invalid flush-batch comparison failed: {error}"
                        )
                    });
                    assert_eq!(
                        comparison.status,
                        ComparisonStatus::Failed,
                        "E6.3 {engine} seed {seed:#x} incorrectly accepted a content mutation as flush batching",
                    );
                }
            }
        });
    }

    /// E6.3 law: the declared scalar G1A analyzer makes free-text case and
    /// surrounding ASCII whitespace non-observable. The projection is the
    /// normal cross-engine differential observation, and this intentionally
    /// excludes fielded and syntax-bearing queries whose parser spelling is
    /// semantically significant. Replacing the analyzed term itself is the
    /// invalid fixture and must not be accepted as normalization.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_query_case_and_whitespace_normalization_is_exact_and_term_mutation_is_not() {
        use frankensearch_core::IndexableDocument;

        const SEED: u64 = 0xe63_c453_0001_5eed;
        let documents = vec![
            IndexableDocument::new("doc-1", "alpha beta beta").with_title("guide"),
            IndexableDocument::new("doc-2", "alpha gamma").with_title("alpha overview"),
            IndexableDocument::new("doc-3", "beta gamma gamma gamma").with_title("alpha"),
            IndexableDocument::new("doc-4", "alpha beta gamma delta").with_title("reference"),
            IndexableDocument::new("doc-5", "delta epsilon").with_title("quiet"),
        ];
        let canonical = [("free-text-alpha", "alpha")];
        let normalized = [("free-text-alpha", " \tAlPhA\n")];
        let term_mutated = [("free-text-alpha", "saffron")];

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let baseline = e63_observations(
                &cx,
                &documents,
                &canonical,
                SEED,
                "e6.3-query-normalization-v1",
            )
            .await;
            let transformed = e63_observations(
                &cx,
                &documents,
                &normalized,
                SEED,
                "e6.3-query-normalization-v1",
            )
            .await;
            let invalid = e63_observations(
                &cx,
                &documents,
                &term_mutated,
                SEED,
                "e6.3-query-normalization-v1",
            )
            .await;

            for (
                (baseline_id, baseline_quill, baseline_tantivy),
                (normalized_id, normalized_quill, normalized_tantivy),
            ) in baseline.iter().zip(&transformed)
            {
                assert_eq!(
                    baseline_id, normalized_id,
                    "E6.3 query-normalization case identity drifted"
                );
                for (engine, before, after) in [
                    ("Quill", baseline_quill, normalized_quill),
                    ("Tantivy", baseline_tantivy, normalized_tantivy),
                ] {
                    let comparison = compare_observations(
                        before.clone(),
                        after.clone(),
                        ComparatorConfig::default(),
                    )
                    .unwrap_or_else(|error| {
                        panic!("E6.3 {engine} {baseline_id} query-normalization comparison failed: {error}")
                    });
                    assert!(
                        matches!(
                            comparison.rank_class,
                            RankClass::RankExact | RankClass::TieOrder
                        ) && comparison
                            .divergences
                            .iter()
                            .all(|divergence| divergence.class == DivergenceClass::TieOrder),
                        "E6.3 {engine} {baseline_id} produced a non-tie divergence under analyzer-declared query normalization: {:?}",
                        comparison.divergences,
                    );
                }
            }

            let baseline_case = baseline
                .first()
                .expect("E6.3 baseline normalization fixture");
            let invalid_case = invalid.first().expect("E6.3 invalid normalization fixture");
            for (engine, before, after) in [
                ("Quill", &baseline_case.1, &invalid_case.1),
                ("Tantivy", &baseline_case.2, &invalid_case.2),
            ] {
                let comparison = compare_observations(
                    before.clone(),
                    after.clone(),
                    ComparatorConfig::default(),
                )
                .unwrap_or_else(|error| {
                    panic!("E6.3 {engine} invalid query-normalization comparison failed: {error}")
                });
                assert_eq!(
                    comparison.status,
                    ComparisonStatus::Failed,
                    "E6.3 {engine} incorrectly accepted a changed analyzed term as normalization",
                );
            }
        });
    }

    /// E6.3 seeded property campaign for analyzer-declared ASCII free-text
    /// normalization. The generator chooses one bounded whitespace/case form
    /// per seed and every form must replay through the normal live
    /// cross-engine observation. The paired single-fixture law keeps the
    /// intentionally invalid analyzed-term mutation.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_query_normalization_seed_matrix_replays_live_observations() {
        use frankensearch_core::IndexableDocument;

        const SEEDS: [u64; 3] = [
            0xe63_c453_0001_5eed,
            0xe63_c453_0001_5eee,
            0xe63_c453_0001_5eef,
        ];
        let documents = vec![
            IndexableDocument::new("doc-1", "alpha beta beta").with_title("guide"),
            IndexableDocument::new("doc-2", "alpha gamma").with_title("alpha overview"),
            IndexableDocument::new("doc-3", "beta gamma gamma gamma").with_title("alpha"),
            IndexableDocument::new("doc-4", "alpha beta gamma delta").with_title("reference"),
            IndexableDocument::new("doc-5", "delta epsilon").with_title("quiet"),
        ];

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (term, seed) in [("alpha", SEEDS[0]), ("beta", SEEDS[1]), ("gamma", SEEDS[2])] {
                let normalized = e63_seeded_ascii_query_normalization(term, seed);
                assert_ne!(
                    normalized, term,
                    "E6.3 seed {seed:#x} must transform its query"
                );
                assert_eq!(
                    normalized,
                    e63_seeded_ascii_query_normalization(term, seed),
                    "E6.3 seed {seed:#x} must replay byte-identically",
                );
                let canonical_cases = [("free-text", term)];
                let normalized_cases = [("free-text", normalized.as_str())];
                let baseline = e63_observations(
                    &cx,
                    &documents,
                    &canonical_cases,
                    seed,
                    "e6.3-query-normalization-v1",
                )
                .await;
                let transformed = e63_observations(
                    &cx,
                    &documents,
                    &normalized_cases,
                    seed,
                    "e6.3-query-normalization-v1",
                )
                .await;
                let baseline_case = baseline
                    .first()
                    .expect("E6.3 baseline normalization fixture");
                let transformed_case = transformed
                    .first()
                    .expect("E6.3 transformed normalization fixture");
                for (engine, before, after) in [
                    ("Quill", &baseline_case.1, &transformed_case.1),
                    ("Tantivy", &baseline_case.2, &transformed_case.2),
                ] {
                    let comparison = compare_observations(
                        before.clone(),
                        after.clone(),
                        ComparatorConfig::default(),
                    )
                    .unwrap_or_else(|error| {
                        panic!(
                            "E6.3 {engine} seed {seed:#x} normalization comparison failed: {error}"
                        )
                    });
                    assert!(
                        matches!(
                            comparison.rank_class,
                            RankClass::RankExact | RankClass::TieOrder
                        ) && comparison
                            .divergences
                            .iter()
                            .all(|divergence| divergence.class == DivergenceClass::TieOrder),
                        "E6.3 {engine} seed {seed:#x} produced a non-tie normalization divergence: {:?}",
                        comparison.divergences,
                    );
                }
            }
        });
    }

    /// E6.3 law: the two distinct, unboosted positive operands of one scalar
    /// `AND` are commutative. This deliberately excludes three-or-more clause
    /// association, boosts, and mixed Boolean occurrences because their score
    /// accumulation or parser shape is observable. The normal differential
    /// observation is the projection; changing the operator to `OR` is the
    /// intentionally invalid counterexample.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_two_term_and_commutes_but_or_is_not_equivalent() {
        use frankensearch_core::IndexableDocument;

        const SEED: u64 = 0xe63_a11d_c0aa_5eed;
        let documents = vec![
            IndexableDocument::new("doc-1", "alpha beta beta").with_title("guide"),
            IndexableDocument::new("doc-2", "alpha gamma").with_title("alpha overview"),
            IndexableDocument::new("doc-3", "beta gamma gamma gamma").with_title("alpha"),
            IndexableDocument::new("doc-4", "alpha beta gamma delta").with_title("reference"),
            IndexableDocument::new("doc-5", "delta epsilon").with_title("quiet"),
        ];
        let canonical = [("two-term-and", "alpha AND beta")];
        let commuted = [("two-term-and", "beta AND alpha")];
        let operator_mutated = [("two-term-and", "alpha OR beta")];

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let baseline = e63_observations(
                &cx,
                &documents,
                &canonical,
                SEED,
                "e6.3-two-term-and-commutativity-v1",
            )
            .await;
            let transformed = e63_observations(
                &cx,
                &documents,
                &commuted,
                SEED,
                "e6.3-two-term-and-commutativity-v1",
            )
            .await;
            let invalid = e63_observations(
                &cx,
                &documents,
                &operator_mutated,
                SEED,
                "e6.3-two-term-and-commutativity-v1",
            )
            .await;

            let baseline_case = baseline.first().expect("E6.3 baseline AND fixture");
            let commuted_case = transformed.first().expect("E6.3 commuted AND fixture");
            assert_eq!(
                baseline_case.0, commuted_case.0,
                "E6.3 two-term AND case identity drifted"
            );
            for (engine, before, after) in [
                ("Quill", &baseline_case.1, &commuted_case.1),
                ("Tantivy", &baseline_case.2, &commuted_case.2),
            ] {
                let comparison = compare_observations(
                    before.clone(),
                    after.clone(),
                    ComparatorConfig::default(),
                )
                .unwrap_or_else(|error| {
                    panic!("E6.3 {engine} two-term AND comparison failed: {error}")
                });
                assert!(
                    matches!(
                        comparison.rank_class,
                        RankClass::RankExact | RankClass::TieOrder
                    ) && comparison
                        .divergences
                        .iter()
                        .all(|divergence| divergence.class == DivergenceClass::TieOrder),
                    "E6.3 {engine} produced a non-tie divergence under two-term AND commutation: {:?}",
                    comparison.divergences,
                );
            }

            let invalid_case = invalid.first().expect("E6.3 invalid OR fixture");
            for (engine, before, after) in [
                ("Quill", &baseline_case.1, &invalid_case.1),
                ("Tantivy", &baseline_case.2, &invalid_case.2),
            ] {
                let comparison = compare_observations(
                    before.clone(),
                    after.clone(),
                    ComparatorConfig::default(),
                )
                .unwrap_or_else(|error| {
                    panic!("E6.3 {engine} invalid AND-to-OR comparison failed: {error}")
                });
                assert_eq!(
                    comparison.status,
                    ComparisonStatus::Failed,
                    "E6.3 {engine} incorrectly accepted an AND-to-OR mutation as commutation",
                );
            }
        });
    }

    /// E6.3 bounded replay campaign for the qualified two-distinct-term,
    /// unboosted positive `AND` commutativity law. The paired one-fixture law
    /// keeps the intentionally invalid `OR` operator mutation.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_two_term_and_seed_matrix_replays_live_observations() {
        use frankensearch_core::IndexableDocument;

        const SEEDS: [u64; 3] = [
            0xe63_a11d_c0aa_5eed,
            0xe63_a11d_c0aa_5eee,
            0xe63_a11d_c0aa_5eef,
        ];
        let documents = vec![
            IndexableDocument::new("doc-1", "alpha beta beta").with_title("guide"),
            IndexableDocument::new("doc-2", "alpha gamma").with_title("alpha overview"),
            IndexableDocument::new("doc-3", "beta gamma gamma gamma").with_title("alpha"),
            IndexableDocument::new("doc-4", "alpha beta gamma delta").with_title("reference"),
            IndexableDocument::new("doc-5", "delta epsilon").with_title("quiet"),
        ];

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for ((left, right), seed) in [
                (("alpha", "beta"), SEEDS[0]),
                (("alpha", "gamma"), SEEDS[1]),
                (("beta", "gamma"), SEEDS[2]),
            ] {
                let canonical = format!("{left} AND {right}");
                let commuted = format!("{right} AND {left}");
                assert_ne!(
                    canonical, commuted,
                    "E6.3 seed {seed:#x} must exercise a real operand-order transform"
                );
                assert_eq!(
                    commuted,
                    format!("{right} AND {left}"),
                    "E6.3 seed {seed:#x} must replay its operand-order transform byte-identically",
                );
                let canonical_cases = [("two-term-and", canonical.as_str())];
                let commuted_cases = [("two-term-and", commuted.as_str())];
                let baseline = e63_observations(
                    &cx,
                    &documents,
                    &canonical_cases,
                    seed,
                    "e6.3-two-term-and-commutativity-v1",
                )
                .await;
                let transformed = e63_observations(
                    &cx,
                    &documents,
                    &commuted_cases,
                    seed,
                    "e6.3-two-term-and-commutativity-v1",
                )
                .await;
                let baseline_case = baseline.first().expect("E6.3 seed baseline AND fixture");
                let commuted_case = transformed.first().expect("E6.3 seed commuted AND fixture");
                assert_eq!(
                    baseline_case.0, commuted_case.0,
                    "E6.3 seed {seed:#x} AND case identity drifted"
                );
                for (engine, before, after) in [
                    ("Quill", &baseline_case.1, &commuted_case.1),
                    ("Tantivy", &baseline_case.2, &commuted_case.2),
                ] {
                    let comparison = compare_observations(
                        before.clone(),
                        after.clone(),
                        ComparatorConfig::default(),
                    )
                    .unwrap_or_else(|error| {
                        panic!("E6.3 {engine} seed {seed:#x} AND comparison failed: {error}")
                    });
                    assert!(
                        matches!(
                            comparison.rank_class,
                            RankClass::RankExact | RankClass::TieOrder
                        ) && comparison
                            .divergences
                            .iter()
                            .all(|divergence| divergence.class == DivergenceClass::TieOrder),
                        "E6.3 {engine} seed {seed:#x} produced a non-tie AND commutation divergence: {:?}",
                        comparison.divergences,
                    );
                }
            }
        });
    }

    /// E6.3 law: the two distinct, unboosted optional operands of one scalar
    /// `OR` are commutative. As with the `AND` law, this excludes association,
    /// boosts, and mixed occurrences because those shapes can expose parser or
    /// score-accumulation behavior. Changing the operator to `AND` is the
    /// intentionally invalid counterexample.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_two_term_or_commutes_but_and_is_not_equivalent() {
        use frankensearch_core::IndexableDocument;

        const SEED: u64 = 0xe63_0f00_c0aa_5eed;
        let documents = vec![
            IndexableDocument::new("doc-1", "alpha beta beta").with_title("guide"),
            IndexableDocument::new("doc-2", "alpha gamma").with_title("alpha overview"),
            IndexableDocument::new("doc-3", "beta gamma gamma gamma").with_title("alpha"),
            IndexableDocument::new("doc-4", "alpha beta gamma delta").with_title("reference"),
            IndexableDocument::new("doc-5", "delta epsilon").with_title("quiet"),
        ];
        let canonical = [("two-term-or", "alpha OR beta")];
        let commuted = [("two-term-or", "beta OR alpha")];
        let operator_mutated = [("two-term-or", "alpha AND beta")];

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let baseline = e63_observations(
                &cx,
                &documents,
                &canonical,
                SEED,
                "e6.3-two-term-or-commutativity-v1",
            )
            .await;
            let transformed = e63_observations(
                &cx,
                &documents,
                &commuted,
                SEED,
                "e6.3-two-term-or-commutativity-v1",
            )
            .await;
            let invalid = e63_observations(
                &cx,
                &documents,
                &operator_mutated,
                SEED,
                "e6.3-two-term-or-commutativity-v1",
            )
            .await;

            let baseline_case = baseline.first().expect("E6.3 baseline OR fixture");
            let commuted_case = transformed.first().expect("E6.3 commuted OR fixture");
            assert_eq!(
                baseline_case.0, commuted_case.0,
                "E6.3 two-term OR case identity drifted"
            );
            for (engine, before, after) in [
                ("Quill", &baseline_case.1, &commuted_case.1),
                ("Tantivy", &baseline_case.2, &commuted_case.2),
            ] {
                let comparison = compare_observations(
                    before.clone(),
                    after.clone(),
                    ComparatorConfig::default(),
                )
                .unwrap_or_else(|error| {
                    panic!("E6.3 {engine} two-term OR comparison failed: {error}")
                });
                assert!(
                    matches!(
                        comparison.rank_class,
                        RankClass::RankExact | RankClass::TieOrder
                    ) && comparison
                        .divergences
                        .iter()
                        .all(|divergence| divergence.class == DivergenceClass::TieOrder),
                    "E6.3 {engine} produced a non-tie divergence under two-term OR commutation: {:?}",
                    comparison.divergences,
                );
            }

            let invalid_case = invalid.first().expect("E6.3 invalid AND fixture");
            for (engine, before, after) in [
                ("Quill", &baseline_case.1, &invalid_case.1),
                ("Tantivy", &baseline_case.2, &invalid_case.2),
            ] {
                let comparison = compare_observations(
                    before.clone(),
                    after.clone(),
                    ComparatorConfig::default(),
                )
                .unwrap_or_else(|error| {
                    panic!("E6.3 {engine} invalid OR-to-AND comparison failed: {error}")
                });
                assert_eq!(
                    comparison.status,
                    ComparisonStatus::Failed,
                    "E6.3 {engine} incorrectly accepted an OR-to-AND mutation as commutation",
                );
            }
        });
    }

    /// E6.3 bounded replay campaign for the qualified two-distinct-term,
    /// unboosted optional `OR` commutativity law. The paired one-fixture law
    /// keeps the intentionally invalid `AND` operator mutation.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_two_term_or_seed_matrix_replays_live_observations() {
        use frankensearch_core::IndexableDocument;

        const SEEDS: [u64; 3] = [
            0xe63_0f00_c0aa_5eed,
            0xe63_0f00_c0aa_5eee,
            0xe63_0f00_c0aa_5eef,
        ];
        let documents = vec![
            IndexableDocument::new("doc-1", "alpha beta beta").with_title("guide"),
            IndexableDocument::new("doc-2", "alpha gamma").with_title("alpha overview"),
            IndexableDocument::new("doc-3", "beta gamma gamma gamma").with_title("alpha"),
            IndexableDocument::new("doc-4", "alpha beta gamma delta").with_title("reference"),
            IndexableDocument::new("doc-5", "delta epsilon").with_title("quiet"),
        ];

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for ((left, right), seed) in [
                (("alpha", "beta"), SEEDS[0]),
                (("alpha", "gamma"), SEEDS[1]),
                (("beta", "gamma"), SEEDS[2]),
            ] {
                let canonical = format!("{left} OR {right}");
                let commuted = format!("{right} OR {left}");
                assert_ne!(
                    canonical, commuted,
                    "E6.3 seed {seed:#x} must exercise a real operand-order transform"
                );
                assert_eq!(
                    commuted,
                    format!("{right} OR {left}"),
                    "E6.3 seed {seed:#x} must replay its operand-order transform byte-identically",
                );
                let canonical_cases = [("two-term-or", canonical.as_str())];
                let commuted_cases = [("two-term-or", commuted.as_str())];
                let baseline = e63_observations(
                    &cx,
                    &documents,
                    &canonical_cases,
                    seed,
                    "e6.3-two-term-or-commutativity-v1",
                )
                .await;
                let transformed = e63_observations(
                    &cx,
                    &documents,
                    &commuted_cases,
                    seed,
                    "e6.3-two-term-or-commutativity-v1",
                )
                .await;
                let baseline_case = baseline.first().expect("E6.3 seed baseline OR fixture");
                let commuted_case = transformed.first().expect("E6.3 seed commuted OR fixture");
                assert_eq!(
                    baseline_case.0, commuted_case.0,
                    "E6.3 seed {seed:#x} OR case identity drifted"
                );
                for (engine, before, after) in [
                    ("Quill", &baseline_case.1, &commuted_case.1),
                    ("Tantivy", &baseline_case.2, &commuted_case.2),
                ] {
                    let comparison = compare_observations(
                        before.clone(),
                        after.clone(),
                        ComparatorConfig::default(),
                    )
                    .unwrap_or_else(|error| {
                        panic!("E6.3 {engine} seed {seed:#x} OR comparison failed: {error}")
                    });
                    assert!(
                        matches!(
                            comparison.rank_class,
                            RankClass::RankExact | RankClass::TieOrder
                        ) && comparison
                            .divergences
                            .iter()
                            .all(|divergence| divergence.class == DivergenceClass::TieOrder),
                        "E6.3 {engine} seed {seed:#x} produced a non-tie OR commutation divergence: {:?}",
                        comparison.divergences,
                    );
                }
            }
        });
    }

    /// E6.3 law: on the position-capable scalar fixture, a quoted single term
    /// reduces to the same term query. This does not assert phrase equivalence
    /// generally: the multi-term phrase is the intentionally invalid control,
    /// because it reads adjacency positions and narrows the matching corpus.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_single_term_quote_matches_bare_term_but_multi_term_phrase_does_not() {
        use frankensearch_core::IndexableDocument;

        const SEED: u64 = 0xe63_907e_5eed_0001;
        let documents = vec![
            IndexableDocument::new("doc-1", "alpha beta beta").with_title("guide"),
            IndexableDocument::new("doc-2", "alpha gamma").with_title("alpha overview"),
            IndexableDocument::new("doc-3", "beta gamma gamma gamma").with_title("alpha"),
            IndexableDocument::new("doc-4", "alpha beta gamma delta").with_title("reference"),
            IndexableDocument::new("doc-5", "delta epsilon").with_title("quiet"),
        ];
        let bare_term = [("single-term-phrase", "alpha")];
        let quoted_term = [("single-term-phrase", "\"alpha\"")];
        let multi_term_phrase = [("single-term-phrase", "\"alpha beta\"")];

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let baseline = e63_observations(
                &cx,
                &documents,
                &bare_term,
                SEED,
                "e6.3-single-term-quote-v1",
            )
            .await;
            let transformed = e63_observations(
                &cx,
                &documents,
                &quoted_term,
                SEED,
                "e6.3-single-term-quote-v1",
            )
            .await;
            let invalid = e63_observations(
                &cx,
                &documents,
                &multi_term_phrase,
                SEED,
                "e6.3-single-term-quote-v1",
            )
            .await;

            let baseline_case = baseline.first().expect("E6.3 bare-term fixture");
            let quoted_case = transformed.first().expect("E6.3 quoted-term fixture");
            assert_eq!(
                baseline_case.0, quoted_case.0,
                "E6.3 single-term quote case identity drifted"
            );
            for (engine, before, after) in [
                ("Quill", &baseline_case.1, &quoted_case.1),
                ("Tantivy", &baseline_case.2, &quoted_case.2),
            ] {
                let comparison = compare_observations(
                    before.clone(),
                    after.clone(),
                    ComparatorConfig::default(),
                )
                .unwrap_or_else(|error| {
                    panic!("E6.3 {engine} single-term quote comparison failed: {error}")
                });
                assert!(
                    matches!(
                        comparison.rank_class,
                        RankClass::RankExact | RankClass::TieOrder
                    ) && comparison
                        .divergences
                        .iter()
                        .all(|divergence| divergence.class == DivergenceClass::TieOrder),
                    "E6.3 {engine} produced a non-tie divergence under single-term quote equivalence: {:?}",
                    comparison.divergences,
                );
            }

            let invalid_case = invalid.first().expect("E6.3 multi-term phrase fixture");
            for (engine, before, after) in [
                ("Quill", &baseline_case.1, &invalid_case.1),
                ("Tantivy", &baseline_case.2, &invalid_case.2),
            ] {
                let comparison = compare_observations(
                    before.clone(),
                    after.clone(),
                    ComparatorConfig::default(),
                )
                .unwrap_or_else(|error| {
                    panic!("E6.3 {engine} invalid phrase comparison failed: {error}")
                });
                assert_eq!(
                    comparison.status,
                    ComparisonStatus::Failed,
                    "E6.3 {engine} incorrectly accepted a multi-term phrase as bare-term equivalence",
                );
            }
        });
    }

    /// E6.3 bounded replay campaign for the qualified position-capable
    /// single-term quote law. Each deterministic seed selects one scalar term
    /// and must preserve the full live observation; the paired one-fixture law
    /// above retains the intentionally invalid multi-term phrase control.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_single_term_quote_seed_matrix_replays_live_observations() {
        use frankensearch_core::IndexableDocument;

        const SEEDS: [u64; 3] = [
            0xe63_907e_5eed_0001,
            0xe63_907e_5eed_0002,
            0xe63_907e_5eed_0003,
        ];
        let documents = vec![
            IndexableDocument::new("doc-1", "alpha beta beta").with_title("guide"),
            IndexableDocument::new("doc-2", "alpha gamma").with_title("alpha overview"),
            IndexableDocument::new("doc-3", "beta gamma gamma gamma").with_title("alpha"),
            IndexableDocument::new("doc-4", "alpha beta gamma delta").with_title("reference"),
            IndexableDocument::new("doc-5", "delta epsilon").with_title("quiet"),
        ];

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (term, seed) in [("alpha", SEEDS[0]), ("beta", SEEDS[1]), ("gamma", SEEDS[2])] {
                let quoted = format!("\"{term}\"");
                assert_eq!(
                    quoted,
                    format!("\"{term}\""),
                    "E6.3 seed {seed:#x} must replay its quote transform byte-identically",
                );
                let bare_cases = [("single-term-quote", term)];
                let quoted_cases = [("single-term-quote", quoted.as_str())];
                let baseline = e63_observations(
                    &cx,
                    &documents,
                    &bare_cases,
                    seed,
                    "e6.3-single-term-quote-v1",
                )
                .await;
                let transformed = e63_observations(
                    &cx,
                    &documents,
                    &quoted_cases,
                    seed,
                    "e6.3-single-term-quote-v1",
                )
                .await;
                let baseline_case = baseline.first().expect("E6.3 seed baseline quote fixture");
                let quoted_case = transformed.first().expect("E6.3 seed quoted quote fixture");
                assert_eq!(
                    baseline_case.0, quoted_case.0,
                    "E6.3 seed {seed:#x} quote case identity drifted"
                );
                for (engine, before, after) in [
                    ("Quill", &baseline_case.1, &quoted_case.1),
                    ("Tantivy", &baseline_case.2, &quoted_case.2),
                ] {
                    let comparison = compare_observations(
                        before.clone(),
                        after.clone(),
                        ComparatorConfig::default(),
                    )
                    .unwrap_or_else(|error| {
                        panic!("E6.3 {engine} seed {seed:#x} quote comparison failed: {error}")
                    });
                    assert!(
                        matches!(
                            comparison.rank_class,
                            RankClass::RankExact | RankClass::TieOrder
                        ) && comparison
                            .divergences
                            .iter()
                            .all(|divergence| divergence.class == DivergenceClass::TieOrder),
                        "E6.3 {engine} seed {seed:#x} produced a non-tie single-term quote divergence: {:?}",
                        comparison.divergences,
                    );
                }
            }
        });
    }

    /// E6.3 capability law: a positionless schema still serves a single-term
    /// quote because it reduces to a term query, while a multi-term phrase
    /// must fail at the typed position-capability boundary. This is not a
    /// cross-engine equivalence claim: the expected Quill refusal is itself
    /// the observable, with its schema, field, operator, and capability named.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_positionless_multi_term_phrase_is_typed_error_but_single_term_quote_is_servable() {
        use frankensearch_core::IndexableDocument;
        use frankensearch_quill::index::QuillIndexError;
        use frankensearch_quill::query::{IndexCapability, QueryCapabilityError, QueryExplanation};

        const SEED: u64 = 0xe63_b051_5eed_0001;
        let documents = vec![
            IndexableDocument::new("doc-1", "alpha beta beta").with_title("guide"),
            IndexableDocument::new("doc-2", "alpha gamma").with_title("alpha overview"),
        ];
        let mut single_term =
            DifferentialCase::new("e63-positionless-single-term", "\"alpha\"", 16);
        single_term.snippet_max_chars = None;
        single_term.tie_expansion_limit = 64;
        single_term.metadata.generator_id = Some("e6.3-positionless-capability-v1".to_owned());
        single_term.metadata.generator_seed = Some(SEED);
        let mut multi_term =
            DifferentialCase::new("e63-positionless-multi-term", "\"alpha beta\"", 16);
        multi_term.snippet_max_chars = None;
        multi_term.tie_expansion_limit = 64;
        multi_term.metadata.generator_id = Some("e6.3-positionless-capability-v1".to_owned());
        multi_term.metadata.generator_seed = Some(SEED);

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let mut subject = qg_position_mode_subject(false);
            subject
                .claim_fresh_campaign()
                .expect("E6.3 claim positionless Quill campaign");
            subject
                .index_mut()
                .expect("E6.3 open positionless Quill campaign")
                .index_documents(&cx, &documents)
                .await
                .expect("E6.3 index positionless Quill fixture");
            subject
                .index_mut()
                .expect("E6.3 open positionless Quill campaign")
                .commit(&cx)
                .await
                .expect("E6.3 commit positionless Quill fixture");
            subject
                .mark_committed()
                .expect("E6.3 publish positionless Quill campaign");

            let single_observation = subject
                .observe(&cx, &single_term)
                .await
                .expect("E6.3 single-term quote must remain servable without positions");
            assert!(
                !single_observation.hits.is_empty(),
                "E6.3 positive positionless fixture"
            );

            let error = subject
                .observe(&cx, &multi_term)
                .await
                .expect_err("E6.3 multi-term phrase must require positions");
            assert!(matches!(
                error,
                GauntletError::Quill(QuillIndexError::QueryCapability(
                    QueryCapabilityError::PositionsRequired {
                        schema: "frankensearch-default-no-positions-v1",
                        ref field,
                        operator: QueryExplanation::Phrase,
                        capability: IndexCapability::Positions,
                    }
                )) if field == "content"
            ));
        });
    }

    /// E6.3 lifecycle law: the scalar campaign deliberately rejects a second
    /// live external ID instead of silently adopting the lexical upsert
    /// contract. The rejection must leave the already-published original as
    /// the only observable row; accepting the replacement or partially
    /// mutating the snapshot is an invalid lifecycle transition.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_duplicate_live_id_is_typed_rejection_and_preserves_published_original() {
        use frankensearch_core::IndexableDocument;
        use frankensearch_quill::index::QuillIndexError;

        const SEED: u64 = 0xe63_d0e5_5eed_0001;
        let original = IndexableDocument::new("duplicate-id", "alpha original");
        let replacement = IndexableDocument::new("duplicate-id", "beta replacement");
        let mut original_case = DifferentialCase::new("e63-duplicate-original", "alpha", 16);
        original_case.snippet_max_chars = None;
        original_case.tie_expansion_limit = 64;
        original_case.metadata.generator_id = Some("e6.3-duplicate-id-reject-v1".to_owned());
        original_case.metadata.generator_seed = Some(SEED);
        let mut replacement_case = DifferentialCase::new("e63-duplicate-replacement", "beta", 16);
        replacement_case.snippet_max_chars = None;
        replacement_case.tie_expansion_limit = 64;
        replacement_case.metadata.generator_id = Some("e6.3-duplicate-id-reject-v1".to_owned());
        replacement_case.metadata.generator_seed = Some(SEED);

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let mut in_batch_subject = qg_position_mode_subject(true);
            in_batch_subject
                .claim_fresh_campaign()
                .expect("E6.3 claim in-batch duplicate-ID Quill campaign");
            let in_batch_error = in_batch_subject
                .index_mut()
                .expect("E6.3 open in-batch duplicate-ID Quill campaign")
                .index_documents(&cx, &[original.clone(), replacement.clone()])
                .await
                .expect_err("E6.3 duplicate IDs in one batch must be rejected atomically");
            assert!(matches!(
                in_batch_error,
                QuillIndexError::InvalidState { ref detail }
                    if detail.contains("duplicate live document id")
            ));
            in_batch_subject
                .mark_committed()
                .expect("E6.3 publish empty rejected in-batch duplicate-ID campaign");
            let in_batch_observation = in_batch_subject
                .observe(&cx, &original_case)
                .await
                .expect("E6.3 observe atomic in-batch rejection");
            assert_eq!(in_batch_observation.doc_count, 0);
            assert!(
                in_batch_observation.hits.is_empty(),
                "E6.3 in-batch duplicate rejection must not partially admit the original"
            );

            let mut subject = qg_position_mode_subject(true);
            subject
                .claim_fresh_campaign()
                .expect("E6.3 claim duplicate-ID Quill campaign");
            subject
                .index_mut()
                .expect("E6.3 open duplicate-ID Quill campaign")
                .index_documents(&cx, std::slice::from_ref(&original))
                .await
                .expect("E6.3 index original duplicate-ID fixture");
            subject
                .index_mut()
                .expect("E6.3 open duplicate-ID Quill campaign")
                .commit(&cx)
                .await
                .expect("E6.3 publish original duplicate-ID fixture");

            let error = subject
                .index_mut()
                .expect("E6.3 reopen duplicate-ID Quill campaign")
                .index_documents(&cx, std::slice::from_ref(&replacement))
                .await
                .expect_err("E6.3 duplicate live ID must be rejected");
            assert!(matches!(
                error,
                QuillIndexError::InvalidState { ref detail }
                    if detail.contains("duplicate live document id")
            ));

            subject
                .mark_committed()
                .expect("E6.3 preserve original duplicate-ID publication");
            let original_observation = subject
                .observe(&cx, &original_case)
                .await
                .expect("E6.3 original remains observable after rejection");
            assert_eq!(original_observation.doc_count, 1);
            assert_eq!(original_observation.hits.len(), 1);
            assert_eq!(original_observation.hits[0].doc_id, "duplicate-id");

            let replacement_observation = subject
                .observe(&cx, &replacement_case)
                .await
                .expect("E6.3 rejected replacement query remains observable");
            assert_eq!(replacement_observation.doc_count, 1);
            assert!(
                replacement_observation.hits.is_empty(),
                "E6.3 rejected duplicate must not partially publish replacement content"
            );
        });
    }

    /// E6.3 lifecycle law: a duplicate-ID batch is rejected before either row
    /// publishes. Deleting that ID afterwards must therefore be the same
    /// lifecycle as deleting an ID that was never added: it returns `false`
    /// and leaves the total lexical observation empty. The paired invalid
    /// fixture admits one unique row first; its delete must return `true` so
    /// equal empty terminal search results cannot mask an incorrect relation.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_duplicate_then_delete_seed_matrix_replays_never_added_lifecycle() {
        use frankensearch_core::IndexableDocument;
        use frankensearch_quill::index::QuillIndexError;

        const SEEDS: [u64; 3] = [
            0xe63_ded1_0000_0001,
            0xe63_ded1_0000_0002,
            0xe63_ded1_0000_0003,
        ];

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for seed in SEEDS {
                let document_id = format!("e63-duplicate-then-delete-{seed:016x}");
                let original = IndexableDocument::new(
                    document_id.clone(),
                    format!("alpha original {seed:016x}"),
                );
                let duplicate = IndexableDocument::new(
                    document_id.clone(),
                    format!("beta replacement {seed:016x}"),
                );
                let mut case = DifferentialCase::new(
                    format!("e63-duplicate-then-delete-{seed:016x}"),
                    "alpha",
                    16,
                );
                case.snippet_max_chars = None;
                case.tie_expansion_limit = 64;
                case.metadata.generator_id = Some("e6.3-duplicate-then-delete-v1".to_owned());
                case.metadata.generator_seed = Some(seed);

                let mut rejected = qg_position_mode_subject(true);
                rejected
                    .claim_fresh_campaign()
                    .expect("E6.3 claim rejected duplicate lifecycle campaign");
                let duplicate_error = rejected
                    .index_mut()
                    .expect("E6.3 open rejected duplicate lifecycle campaign")
                    .index_documents(&cx, &[original, duplicate])
                    .await
                    .expect_err("E6.3 duplicate batch must reject atomically");
                assert!(matches!(
                    duplicate_error,
                    QuillIndexError::InvalidState { ref detail }
                        if detail.contains("duplicate live document id")
                ));
                let rejected_delete = rejected
                    .index_mut()
                    .expect("E6.3 rejected duplicate campaign remains open")
                    .delete_document(&cx, &document_id)
                    .await
                    .expect("E6.3 delete after rejected duplicate batch");
                assert!(
                    !rejected_delete,
                    "E6.3 seed {seed:#x} rejected duplicate ID must remain never-added"
                );
                rejected
                    .mark_committed()
                    .expect("E6.3 publish rejected duplicate lifecycle campaign");
                let rejected_observation = rejected
                    .observe(&cx, &case)
                    .await
                    .expect("E6.3 observe rejected duplicate lifecycle campaign");

                let mut never_added = qg_position_mode_subject(true);
                never_added
                    .claim_fresh_campaign()
                    .expect("E6.3 claim never-added lifecycle campaign");
                let never_added_delete = never_added
                    .index_mut()
                    .expect("E6.3 open never-added lifecycle campaign")
                    .delete_document(&cx, &document_id)
                    .await
                    .expect("E6.3 delete never-added ID");
                assert!(
                    !never_added_delete,
                    "E6.3 seed {seed:#x} never-added ID must report absent"
                );
                never_added
                    .mark_committed()
                    .expect("E6.3 publish never-added lifecycle campaign");
                let never_added_observation = never_added
                    .observe(&cx, &case)
                    .await
                    .expect("E6.3 observe never-added lifecycle campaign");
                let comparison = compare_observations(
                    rejected_observation,
                    never_added_observation,
                    ComparatorConfig::default(),
                )
                .unwrap_or_else(|error| {
                    panic!("E6.3 seed {seed:#x} lifecycle comparison failed: {error}")
                });
                assert_eq!(comparison.status, ComparisonStatus::Exact);
                assert_eq!(comparison.rank_class, RankClass::RankExact);

                let unique = IndexableDocument::new(
                    document_id.clone(),
                    format!("alpha uniquely-admitted {seed:016x}"),
                );
                let mut invalid = qg_position_mode_subject(true);
                invalid
                    .claim_fresh_campaign()
                    .expect("E6.3 claim invalid unique-admission lifecycle campaign");
                invalid
                    .index_mut()
                    .expect("E6.3 open invalid unique-admission lifecycle campaign")
                    .index_documents(&cx, std::slice::from_ref(&unique))
                    .await
                    .expect("E6.3 admit invalid unique fixture");
                invalid
                    .index_mut()
                    .expect("E6.3 commit invalid unique-admission lifecycle campaign")
                    .commit(&cx)
                    .await
                    .expect("E6.3 publish invalid unique fixture");
                let invalid_delete = invalid
                    .index_mut()
                    .expect("E6.3 reopen invalid unique-admission lifecycle campaign")
                    .delete_document(&cx, &document_id)
                    .await
                    .expect("E6.3 delete invalid unique fixture");
                assert!(
                    invalid_delete,
                    "E6.3 seed {seed:#x} planted unique admission must not satisfy never-added relation"
                );
            }
        });
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn e410_controlled_public_search_semantics_match_oracle() {
        let (revision, dirty) = test_producer_source();
        let mut subject =
            QuillSubject::in_memory_with_source(e55_config(), revision.clone(), dirty)
                .expect("E4.10 Quill subject");
        let mut oracle = TantivyOracle::in_memory_scalar_g1a_with_source(&revision, dirty)
            .expect("E4.10 Tantivy oracle");
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
        let (revision, dirty) = test_producer_source();
        let mut subject =
            QuillSubject::in_memory_with_source(e55_config(), revision.clone(), dirty)
                .expect("E4.10 limits Quill subject");
        let mut oracle = TantivyOracle::in_memory_scalar_g1a_with_source(&revision, dirty)
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

    /// E4.10 deferred-fusion-candidate envelope, ported from the incumbent's
    /// `deferred_fusion_candidates_restore_exact_metadata` and
    /// `search_results_have_lexical_source` and run engine-parameterized over
    /// the Quill subject and the Tantivy oracle (bd-quill-e4-argus-3ycz.10).
    ///
    /// `LexicalSearch` — not the internal collector — is what hybrid fusion
    /// consumes, so the whole envelope has to agree: the cheap candidate path
    /// must be bit-identical to the full path except for deferred metadata,
    /// hydration must restore exactly the metadata the full path would have
    /// returned (including when only a strict subset of candidates survives
    /// fusion), hydration must not perturb order or scores, and winners that
    /// never came from the lexical pool must be left untouched.
    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn e410_deferred_fusion_candidate_envelope_matches_on_both_engines() {
        use frankensearch_core::{LexicalRead, ScoreSource, ScoredResult};

        const E410_FUSION_QUERY: &str = "rust";
        const E410_FUSION_LIMIT: usize = 10;

        /// Assert the whole deferred-candidate contract on one engine and
        /// return its hydrated candidates for cross-engine comparison.
        async fn assert_deferred_envelope(
            cx: &Cx,
            engine: &dyn LexicalRead,
            label: &str,
        ) -> Vec<ScoredResult> {
            let full = engine
                .search(cx, E410_FUSION_QUERY, E410_FUSION_LIMIT)
                .await
                .unwrap_or_else(|error| panic!("{label}: full search failed: {error}"));
            assert!(
                full.len() >= 2,
                "{label}: fixture must produce a multi-hit ranking, got {}",
                full.len(),
            );

            let batch = engine
                .search_candidates(cx, E410_FUSION_QUERY, E410_FUSION_LIMIT)
                .await
                .unwrap_or_else(|error| panic!("{label}: fusion candidates failed: {error}"));
            assert!(
                batch.is_deferred(),
                "{label}: the candidate path must advertise deferred metadata",
            );
            let (mut candidates, pin) = batch.into_parts();
            assert_eq!(
                candidates.len(),
                full.len(),
                "{label}: candidate arity must match the full path",
            );
            assert!(
                candidates.iter().all(|result| result.metadata.is_none()),
                "{label}: deferred candidates must carry no stored metadata",
            );

            for (rank, (candidate, expected)) in candidates.iter().zip(&full).enumerate() {
                assert_eq!(
                    candidate.doc_id, expected.doc_id,
                    "{label}: rank {rank} document identity",
                );
                assert_eq!(
                    candidate.score.to_bits(),
                    expected.score.to_bits(),
                    "{label}: rank {rank} score bits",
                );
                assert_eq!(
                    candidate.lexical_score.map(f32::to_bits),
                    expected.lexical_score.map(f32::to_bits),
                    "{label}: rank {rank} lexical score bits",
                );
                for (path, result) in [("candidate", candidate), ("full", expected)] {
                    assert_eq!(
                        result.source,
                        ScoreSource::Lexical,
                        "{label}/{path}: rank {rank} score source",
                    );
                    assert!(
                        result.lexical_score.is_some_and(|score| score > 0.0),
                        "{label}/{path}: rank {rank} must carry a positive lexical score",
                    );
                    assert!(
                        result.index.is_none()
                            && result.fast_score.is_none()
                            && result.quality_score.is_none()
                            && result.rerank_score.is_none()
                            && result.explanation.is_none(),
                        "{label}/{path}: rank {rank} must populate only the lexical channel",
                    );
                }
            }

            // Fusion hydrates the winners, which is generally a strict subset
            // of the candidate pool: hydrating one winner must restore exactly
            // what the full path returned for that document.
            let mut winner = candidates[..1].to_vec();
            engine
                .hydrate_candidates(cx, pin.as_ref(), &mut winner)
                .await
                .unwrap_or_else(|error| panic!("{label}: winner-subset hydration failed: {error}"));
            assert_eq!(
                winner[0].metadata, full[0].metadata,
                "{label}: subset hydration must restore the winner's metadata",
            );

            engine
                .hydrate_candidates(cx, pin.as_ref(), &mut candidates)
                .await
                .unwrap_or_else(|error| panic!("{label}: hydration failed: {error}"));
            for (rank, (candidate, expected)) in candidates.iter().zip(&full).enumerate() {
                assert_eq!(
                    candidate.metadata, expected.metadata,
                    "{label}: rank {rank} hydrated metadata must equal the full path",
                );
                assert_eq!(
                    candidate.doc_id, expected.doc_id,
                    "{label}: rank {rank} hydration must not reorder candidates",
                );
                assert_eq!(
                    candidate.score.to_bits(),
                    expected.score.to_bits(),
                    "{label}: rank {rank} hydration must not perturb scores",
                );
            }

            // A winner that reached fusion from the semantic arm alone carries
            // no lexical score; hydration must ignore it rather than invent
            // metadata for a document it never retrieved.
            let mut semantic_only = vec![candidates[0].clone()];
            semantic_only[0].metadata = None;
            semantic_only[0].lexical_score = None;
            semantic_only[0].source = ScoreSource::SemanticFast;
            engine
                .hydrate_candidates(cx, pin.as_ref(), &mut semantic_only)
                .await
                .unwrap_or_else(|error| panic!("{label}: semantic-only hydration failed: {error}"));
            assert!(
                semantic_only[0].metadata.is_none(),
                "{label}: hydration must ignore winners with no lexical score",
            );

            candidates
        }

        let revision = oracle_version_contract()
            .expect("oracle version contract")
            .lexical_contract_audit_revision;
        let mut subject =
            QuillSubject::in_memory_with_source(e55_config(), "e410-fusion-subject", false)
                .expect("E4.10 fusion Quill subject");
        let mut oracle = TantivyOracle::in_memory_scalar_g1a_with_source(&revision, false)
            .expect("E4.10 fusion Tantivy oracle");
        // Three documents match `rust` at distinct (tf, |d|) pairs so the
        // ranking is total, and each carries different stored metadata so a
        // mis-keyed hydration cannot pass by accident.
        let documents = vec![
            frankensearch_core::IndexableDocument::new("doc-1", "rust ownership and borrowing")
                .with_title("rust guide")
                .with_metadata("lang", "rust")
                .with_metadata("kind", "guide"),
            frankensearch_core::IndexableDocument::new(
                "doc-2",
                "rust rust rust async runtimes and executors",
            )
            .with_title("concurrency")
            .with_metadata("lang", "rust"),
            frankensearch_core::IndexableDocument::new("doc-3", "python data pipelines")
                .with_title("etl")
                .with_metadata("lang", "python"),
            frankensearch_core::IndexableDocument::new("doc-4", "rust embedded systems")
                .with_title("no_std"),
        ];

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            subject
                .claim_fresh_campaign()
                .expect("claim E4.10 fusion subject campaign");
            subject
                .index_mut()
                .expect("E4.10 fusion subject index")
                .index_documents(&cx, &documents)
                .await
                .expect("index E4.10 fusion subject corpus");
            subject
                .index_mut()
                .expect("E4.10 fusion subject index")
                .commit(&cx)
                .await
                .expect("commit E4.10 fusion subject corpus");
            subject
                .mark_committed()
                .expect("publish E4.10 fusion subject campaign");

            oracle
                .claim_fresh_campaign()
                .expect("claim E4.10 fusion oracle campaign");
            oracle
                .index_documents(&cx, &documents)
                .await
                .expect("index E4.10 fusion oracle corpus");
            oracle
                .mark_committed()
                .expect("publish E4.10 fusion oracle campaign");

            let quill_candidates = assert_deferred_envelope(
                &cx,
                subject.index().expect("E4.10 fusion subject index"),
                "quill",
            )
            .await;
            let oracle_candidates =
                assert_deferred_envelope(&cx, oracle.index(), "tantivy-oracle").await;

            assert_eq!(
                quill_candidates
                    .iter()
                    .map(|result| result.doc_id.as_str())
                    .collect::<Vec<_>>(),
                oracle_candidates
                    .iter()
                    .map(|result| result.doc_id.as_str())
                    .collect::<Vec<_>>(),
                "deferred candidate ranking must agree across engines",
            );
            for (rank, (quill, tantivy)) in
                quill_candidates.iter().zip(&oracle_candidates).enumerate()
            {
                assert_eq!(
                    quill.metadata, tantivy.metadata,
                    "rank {rank} ({}): hydrated metadata must agree across engines",
                    quill.doc_id,
                );
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
    fn oracle_constructor_records_dirty_diagnostics_but_rejects_malformed_source() {
        let revision = oracle_version_contract()
            .expect("version contract")
            .lexical_contract_audit_revision;
        let dirty = TantivyOracle::in_memory_with_source(&revision, true)
            .expect("dirty producer identity remains recordable diagnostic provenance");
        assert!(dirty.descriptor().source_dirty);
        let unavailable = TantivyOracle::in_memory_with_source("unavailable", true)
            .expect("conservative unavailable producer identity remains recordable");
        assert_eq!(unavailable.descriptor().source_revision, "unavailable");
        assert!(TantivyOracle::in_memory_with_source("unavailable", false).is_err());
        assert!(TantivyOracle::in_memory_with_source(&"0".repeat(39), false).is_err());
        assert!(TantivyOracle::in_memory_with_source(&"A".repeat(40), false).is_err());
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn oracle_descriptor_keeps_compiled_producer_separate_from_dependency_contract() {
        let contract = oracle_version_contract().expect("version contract");
        let producer_revision = "f".repeat(40);
        assert_ne!(producer_revision, contract.lexical_contract_audit_revision);

        let oracle =
            TantivyOracle::in_memory_with_source(&producer_revision, false).expect("clean oracle");
        assert_eq!(oracle.descriptor().source_revision, producer_revision);
        assert_eq!(
            oracle_version_contract()
                .expect("independent version contract")
                .lexical_contract_audit_revision,
            contract.lexical_contract_audit_revision,
        );
    }

    #[test]
    fn quill_descriptor_uses_the_linked_quill_package_version() {
        let subject = QuillSubject::in_memory_with_source(e55_config(), "a".repeat(40), false)
            .expect("Quill subject");
        assert_eq!(
            subject.descriptor().crate_version,
            frankensearch_quill::FRANKENSEARCH_QUILL_CRATE_VERSION
        );
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn oracle_rejects_oversized_case_before_query_execution() {
        let revision = oracle_version_contract()
            .expect("version contract")
            .lexical_contract_audit_revision;
        let oracle = TantivyOracle::in_memory_with_source(&revision, false).expect("oracle");
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
