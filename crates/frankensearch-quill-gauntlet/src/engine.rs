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
#[cfg(feature = "fuzz-harness")]
use frankensearch_quill::{
    DEFAULT_SCHEMA, DefaultQueryParser, ParsedQuery, Query, QueryDiagnosticKind,
};
use frankensearch_quill::{QuillConfig, QuillIndex, QuillSearchResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
#[cfg(any(test, feature = "tantivy-oracle"))]
use xxhash_rust::xxh3::xxh3_64;

use crate::GauntletError;
use crate::artifact::GauntletProducerBuildIdentity;
#[cfg(feature = "fuzz-harness")]
use crate::artifact::{PinnedDirectory, PinnedRegularFile};
use crate::comparator::{
    ComparatorConfig, ComparisonReport, CountState, EngineObservation, NativeTieKey,
    OracleBugControlObservation, RankedHit, compare_observations,
};
#[cfg(feature = "fuzz-harness")]
use crate::comparator::{ComparisonStatus, Divergence, DivergenceClass, RankClass};
#[cfg(feature = "tantivy-oracle")]
use crate::generator::GeneratedDocument;
use crate::generator::MAX_DOCUMENT_ID_BYTES;
#[cfg(feature = "fuzz-harness")]
use crate::generator::{CorpusManifest, SyntheticCorpus, SyntheticCorpusSpec, ZipfExponent};
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
const BUILT_IN_PROFILE_V2_QUILL_CRATE_VERSION: &str = "0.2.1";
const BUILT_IN_PROFILE_V2_LEXICAL_CRATE_VERSION: &str = "0.2.1";
const BUILT_IN_PROFILE_V2_DEFAULT_ANALYZER_HASH: &str =
    "7425c0f2d0a909ca4103bd20f439b6282d3ce00ab3c9f6784ec7333398197041";
const BUILT_IN_PROFILE_V2_DEFAULT_SCHEMA_HASH: &str =
    "afe3ad4998181c98ee26de5c47905e3c9e0623e2e144643a02e19ce697b42c0a";
const BUILT_IN_PROFILE_V2_SCALAR_G1A_SCHEMA_HASH: &str =
    "ed82305678b4145b83bd48dc605bf3e9c65736ba3c74983f2268f0f8dbf11e59";
const BUILT_IN_PROFILE_V2_CASS_ANALYZER_HASH: &str =
    "8db8c441617927a16604df40ff17f57a5478996eaa2b0c7b4018dfac1340edcf";
const BUILT_IN_PROFILE_V2_CASS_SCHEMA_HASH: &str =
    "11057d81013ddadc6499674acb23a8b6842d589f4344fa88b3e70fa744fc4ee9";
const BUILT_IN_PROFILE_V2_SCALAR_ORACLE_CONFIG_HASH: &str = "shipping-schema-and-parser-v1";
const BUILT_IN_PROFILE_V2_CASS_ORACLE_CONFIG_HASH: &str = "cass-schema-and-parser-v1";
// v3 (gh#416): the semantic contract is IDENTICAL to v2; only the lexical
// wrapper version moved (0.2.1 -> 0.2.2) when the oracle switched from the
// git tantivy 0.27-dev pin to the registry 0.26.1 dependency universe
// (oracle dependency contract v4). Historical v2 receipts stay archive-valid
// and can no longer create runs, exactly like v1 before them.
const BUILT_IN_PROFILE_V3_QUILL_CRATE_VERSION: &str = "0.2.1";
const BUILT_IN_PROFILE_V3_LEXICAL_CRATE_VERSION: &str = "0.2.2";
// v4 (gh-39 facade release, b31fa58f): the semantic contract is still
// IDENTICAL to v2; both crate versions moved (quill 0.2.1 -> 0.2.2, lexical
// 0.2.2 -> 0.2.3) under the same registry tantivy 0.26.1 oracle (oracle
// dependency contract v5). Historical v3 receipts stay archive-valid and can
// no longer create runs, exactly like v1 and v2 before them.
const BUILT_IN_PROFILE_V4_QUILL_CRATE_VERSION: &str = "0.2.2";
const BUILT_IN_PROFILE_V4_LEXICAL_CRATE_VERSION: &str = "0.2.3";

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
    const V2_SCHEMA_VERSION: u32 = 2;
    const V3_SCHEMA_VERSION: u32 = 3;
    const V4_SCHEMA_VERSION: u32 = 4;
    #[cfg_attr(
        not(any(test, feature = "tantivy-oracle")),
        expect(
            dead_code,
            reason = "typed built-in receipts are constructed only by oracle-backed or test lanes"
        )
    )]
    const CURRENT_SCHEMA_VERSION: u32 = Self::V4_SCHEMA_VERSION;

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
            2 => self.validate_stored_v2(engines),
            3 => self.validate_stored_v3(engines),
            4 => self.validate_stored_v4(engines),
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

    fn stored_semantic_contract_v2(&self) -> crate::runner::SemanticContract {
        let (analyzer_contract_hash, schema_contract_hash) = match self.profile {
            BuiltInEngineProfile::ScalarShipping => (
                BUILT_IN_PROFILE_V2_DEFAULT_ANALYZER_HASH,
                BUILT_IN_PROFILE_V2_DEFAULT_SCHEMA_HASH,
            ),
            BuiltInEngineProfile::ScalarG1a => (
                BUILT_IN_PROFILE_V2_DEFAULT_ANALYZER_HASH,
                BUILT_IN_PROFILE_V2_SCALAR_G1A_SCHEMA_HASH,
            ),
            BuiltInEngineProfile::Cass => (
                BUILT_IN_PROFILE_V2_CASS_ANALYZER_HASH,
                BUILT_IN_PROFILE_V2_CASS_SCHEMA_HASH,
            ),
        };
        crate::runner::SemanticContract {
            analyzer_contract_hash: analyzer_contract_hash.to_owned(),
            schema_contract_hash: schema_contract_hash.to_owned(),
        }
    }

    fn validate_stored_v2(&self, engines: &EnginePairIdentity) -> Result<(), GauntletError> {
        self.subject_config.validate_stored_v1()?;
        let (subject_implementation, subject_hash, oracle_hash) = match self.profile {
            BuiltInEngineProfile::ScalarShipping | BuiltInEngineProfile::ScalarG1a => (
                "frankensearch-quill/scalar-index",
                self.subject_config.descriptor_hash_v1(),
                BUILT_IN_PROFILE_V2_SCALAR_ORACLE_CONFIG_HASH,
            ),
            BuiltInEngineProfile::Cass => (
                "frankensearch-quill/cass-index",
                format!(
                    "cass-semantic-v1:{}",
                    self.subject_config.descriptor_hash_v1()
                ),
                BUILT_IN_PROFILE_V2_CASS_ORACLE_CONFIG_HASH,
            ),
        };
        if self.schema_version != Self::V2_SCHEMA_VERSION
            || engines.comparison_mode != ComparisonMode::CrossEngine
            || engines.subject.implementation != subject_implementation
            || engines.subject.crate_version != BUILT_IN_PROFILE_V2_QUILL_CRATE_VERSION
            || engines.subject.config_hash != subject_hash
            || engines.oracle.implementation != "frankensearch-lexical/tantivy-index"
            || engines.oracle.crate_version != BUILT_IN_PROFILE_V2_LEXICAL_CRATE_VERSION
            || engines.oracle.config_hash != oracle_hash
            || engines.semantic_contract.as_ref() != Some(&self.stored_semantic_contract_v2())
            || engines.subject.source_revision != engines.oracle.source_revision
            || engines.subject.source_dirty != engines.oracle.source_dirty
        {
            return Err(GauntletError::InvalidContract {
                reason: "built-in engine profile receipt v2 does not match its stored adapter identities and semantic contract"
                    .to_owned(),
            });
        }
        validate_recorded_producer_source(
            &engines.subject.source_revision,
            engines.subject.source_dirty,
        )?;
        Ok(())
    }

    /// v3 differs from v2 only in the pinned lexical wrapper version
    /// (0.2.2, the registry-oracle line of gh#416); the analyzer/schema
    /// semantic contract is byte-identical, so v2's is reused.
    fn validate_stored_v3(&self, engines: &EnginePairIdentity) -> Result<(), GauntletError> {
        self.subject_config.validate_stored_v1()?;
        let (subject_implementation, subject_hash, oracle_hash) = match self.profile {
            BuiltInEngineProfile::ScalarShipping | BuiltInEngineProfile::ScalarG1a => (
                "frankensearch-quill/scalar-index",
                self.subject_config.descriptor_hash_v1(),
                BUILT_IN_PROFILE_V2_SCALAR_ORACLE_CONFIG_HASH,
            ),
            BuiltInEngineProfile::Cass => (
                "frankensearch-quill/cass-index",
                format!(
                    "cass-semantic-v1:{}",
                    self.subject_config.descriptor_hash_v1()
                ),
                BUILT_IN_PROFILE_V2_CASS_ORACLE_CONFIG_HASH,
            ),
        };
        if self.schema_version != Self::V3_SCHEMA_VERSION
            || engines.comparison_mode != ComparisonMode::CrossEngine
            || engines.subject.implementation != subject_implementation
            || engines.subject.crate_version != BUILT_IN_PROFILE_V3_QUILL_CRATE_VERSION
            || engines.subject.config_hash != subject_hash
            || engines.oracle.implementation != "frankensearch-lexical/tantivy-index"
            || engines.oracle.crate_version != BUILT_IN_PROFILE_V3_LEXICAL_CRATE_VERSION
            || engines.oracle.config_hash != oracle_hash
            || engines.semantic_contract.as_ref() != Some(&self.stored_semantic_contract_v2())
            || engines.subject.source_revision != engines.oracle.source_revision
            || engines.subject.source_dirty != engines.oracle.source_dirty
        {
            return Err(GauntletError::InvalidContract {
                reason: "built-in engine profile receipt v3 does not match its stored adapter identities and semantic contract"
                    .to_owned(),
            });
        }
        validate_recorded_producer_source(
            &engines.subject.source_revision,
            engines.subject.source_dirty,
        )?;
        Ok(())
    }

    /// v4 differs from v3 only in the pinned crate versions (quill 0.2.2,
    /// lexical 0.2.3 — the gh-39 facade release line); the analyzer/schema
    /// semantic contract is still byte-identical to v2, so v2's is reused.
    fn validate_stored_v4(&self, engines: &EnginePairIdentity) -> Result<(), GauntletError> {
        self.subject_config.validate_stored_v1()?;
        let (subject_implementation, subject_hash, oracle_hash) = match self.profile {
            BuiltInEngineProfile::ScalarShipping | BuiltInEngineProfile::ScalarG1a => (
                "frankensearch-quill/scalar-index",
                self.subject_config.descriptor_hash_v1(),
                BUILT_IN_PROFILE_V2_SCALAR_ORACLE_CONFIG_HASH,
            ),
            BuiltInEngineProfile::Cass => (
                "frankensearch-quill/cass-index",
                format!(
                    "cass-semantic-v1:{}",
                    self.subject_config.descriptor_hash_v1()
                ),
                BUILT_IN_PROFILE_V2_CASS_ORACLE_CONFIG_HASH,
            ),
        };
        if self.schema_version != Self::V4_SCHEMA_VERSION
            || engines.comparison_mode != ComparisonMode::CrossEngine
            || engines.subject.implementation != subject_implementation
            || engines.subject.crate_version != BUILT_IN_PROFILE_V4_QUILL_CRATE_VERSION
            || engines.subject.config_hash != subject_hash
            || engines.oracle.implementation != "frankensearch-lexical/tantivy-index"
            || engines.oracle.crate_version != BUILT_IN_PROFILE_V4_LEXICAL_CRATE_VERSION
            || engines.oracle.config_hash != oracle_hash
            || engines.semantic_contract.as_ref() != Some(&self.stored_semantic_contract_v2())
            || engines.subject.source_revision != engines.oracle.source_revision
            || engines.subject.source_dirty != engines.oracle.source_dirty
        {
            return Err(GauntletError::InvalidContract {
                reason: "built-in engine profile receipt v4 does not match its stored adapter identities and semantic contract"
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

        // bd-pjvl1: a present certificate must be a valid closed proof for
        // exactly this case's page, and its exact total must be the count
        // this observation reports (when it reports one). Absence is no
        // claim; presence is authority and is checked here.
        if let Some(certificate) = &observation.cutoff_certificate {
            certificate
                .validate()
                .map_err(|error| GauntletError::InvalidObservation {
                    reason: format!("{label} cutoff certificate is invalid: {error}"),
                })?;
            let page_len = certificate.page.end.saturating_sub(certificate.page.start);
            if certificate.offset != self.offset
                || certificate.limit != self.limit
                || page_len != hit_count
            {
                return Err(GauntletError::InvalidObservation {
                    reason: format!(
                        "{label} cutoff certificate does not describe the requested page"
                    ),
                });
            }
            if let CountState::Value(match_count) = observation.match_count
                && certificate.exact_total != match_count
            {
                return Err(GauntletError::InvalidObservation {
                    reason: format!(
                        "{label} cutoff certificate exact total disagrees with the exact count"
                    ),
                });
            }
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oracle_bug_control: Option<OracleBugControlObservation>,
    pub comparison: ComparisonReport,
}

/// Current serialized harness-run schema.
pub const HARNESS_RUN_SCHEMA_VERSION: u32 = 3;

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
        // bd-bxya1: one attribution seam for both lanes. The harness holds the
        // query, so a differential run earns the same reviewed oracle-blame
        // class the campaign lane does, from the same gate, and stores the
        // configuration that re-derives it. Before this, the only producer of
        // `OracleBug` was a probe mutating a comparison after the fact.
        let (comparison, comparator_config) = crate::runner::attribute_case_comparison(
            self.comparator_config,
            &case.query,
            comparison,
        )?;
        Ok(HarnessRun {
            schema_version: HARNESS_RUN_SCHEMA_VERSION,
            producer_build_identity,
            engines,
            case: case.clone(),
            comparator_config,
            oracle_bug_control: None,
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

    /// Wrap an index this crate already opened, instead of creating one.
    ///
    /// The E6.3 index-maintenance laws (`bd-quill-e6-gauntlet-scale-rm3q.3`)
    /// need this for exactly one reason: a durable close/reopen cycle. Every
    /// other construction path here is in-memory, and an in-memory index cannot
    /// express recovery — so without this seam the reopen-recovery law could
    /// only be "tested" by approximating a reopen with a fresh open, which
    /// tests something other than what it claims.
    ///
    /// The subject starts `Fresh`, so the caller still walks the normal
    /// claim/commit/publish campaign states; this changes where the index came
    /// from and nothing else. Test-only, and only under `perf-harness`.
    #[cfg(all(test, feature = "perf-harness"))]
    pub(crate) fn from_open_index(
        index: QuillIndex,
        config: QuillConfig,
    ) -> Result<Self, GauntletError> {
        let producer = GauntletProducerBuildIdentity::compiled()?;
        let descriptor = EngineDescriptor {
            family: EngineFamily::Quill,
            implementation: "frankensearch-quill/scalar-index".to_owned(),
            crate_version: frankensearch_quill::FRANKENSEARCH_QUILL_CRATE_VERSION.to_owned(),
            source_revision: producer.source_git_revision,
            source_dirty: producer.source_git_dirty,
            config_hash: quill_config_hash(&config),
        };
        descriptor.validate()?;
        Ok(Self {
            index: Some(index),
            config,
            descriptor,
            state: QuillCampaignState::Fresh,
        })
    }

    /// Take the open index out of this subject, leaving it unusable.
    ///
    /// Used by the reopen-recovery law to CLOSE an index for real: the returned
    /// value is dropped by the caller before the directory is reopened, so the
    /// reopen observes durable state rather than a still-live writer.
    #[cfg(all(test, feature = "perf-harness"))]
    pub(crate) fn take_index(&mut self) -> Result<QuillIndex, GauntletError> {
        self.index
            .take()
            .ok_or_else(|| GauntletError::SubjectUnavailable {
                reason: "Quill campaign subject has no open index to close".to_owned(),
            })
    }

    /// Reinstall an index after a close/reopen cycle, preserving campaign state.
    #[cfg(all(test, feature = "perf-harness"))]
    pub(crate) fn restore_index(&mut self, index: QuillIndex) {
        self.index = Some(index);
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
    // All three run against ONE loaded publication (bd-pjvl1): a refresh
    // between separate calls could otherwise make the page, its expanded
    // evidence, and the exact count describe different snapshots.
    let bundle = index.search_paginated_pinned(cx, &case.query, limit, offset, fetch_limit)?;
    quill_native_observation_from_results(
        &bundle.observed,
        &bundle.expanded,
        &bundle.count,
        limit,
        offset,
        case.count_requested,
        Some(bundle.snapshot_sha256),
    )
}

fn quill_native_observation_from_results(
    observed: &QuillSearchResult,
    evidence: &QuillSearchResult,
    count_evidence: &QuillSearchResult,
    limit: usize,
    offset: usize,
    count_requested: bool,
    snapshot_sha256: Option<[u8; 32]>,
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
        snapshot_sha256,
    )
}

#[cfg(any(feature = "tantivy-oracle", test))]
fn quill_observation_from_results(
    observed: &QuillSearchResult,
    evidence: &QuillSearchResult,
    limit: usize,
    offset: usize,
    count_requested: bool,
    snapshot_sha256: Option<[u8; 32]>,
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
        snapshot_sha256,
    )
}

fn quill_observation_from_validated_results(
    observed: &QuillSearchResult,
    evidence: &QuillSearchResult,
    total_count: u64,
    match_count: CountState,
    limit: usize,
    offset: usize,
    snapshot_sha256: Option<[u8; 32]>,
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
    // The certificate exists only when the three observations were taken on
    // one pinned publication (the native adapter); the CASS preparsed path
    // passes no snapshot digest and therefore makes no claim.
    let cutoff_certificate = match snapshot_sha256 {
        None => None,
        Some(snapshot_sha256) => {
            let rows = |hits: &[frankensearch_quill::QuillHit]| {
                observation_digest(hits.iter().map(|hit| {
                    (
                        hit.document_id.as_str(),
                        hit.score.to_bits(),
                        [hit.global_docid, 0],
                    )
                }))
            };
            let provenance = certificate_provenance(
                snapshot_sha256,
                "frankensearch-quill/scalar-index",
                rows(&observed.hits),
                rows(&evidence.hits),
                total_count,
            );
            let expanded_bits = evidence
                .hits
                .iter()
                .map(|hit| hit.score.to_bits())
                .collect::<Vec<_>>();
            derive_cutoff_certificate(
                total_count,
                u64::try_from(offset).unwrap_or(u64::MAX),
                u64::try_from(limit).unwrap_or(u64::MAX),
                &expanded_bits,
                provenance,
            )?
        }
    };
    Ok(EngineObservation {
        hits,
        cutoff_tie_group,
        cutoff_tie_complete,
        offset_tie_group,
        offset_tie_complete,
        snippets: BTreeMap::new(),
        match_count,
        doc_count: observed.doc_count,
        cutoff_certificate,
        // Subject-side only. The oracle builders below keep an empty vector
        // because the pinned Tantivy parser emits no structured diagnostics to
        // project; that asymmetry is stated verbatim inside each emitted
        // difference rather than inferred from an empty oracle vector.
        // `observed` and `evidence` were already required to agree on
        // diagnostics above, so either source is the same value.
        ast_differences: crate::comparator::ast_differences_from_quill_diagnostics(
            &observed.diagnostics,
        ),
    })
}

/// Digest of one native observation: document identity, score bits, and the
/// native tie key of every row, in order. Ties the certificate to exactly the
/// rows it was derived from.
fn observation_digest<'a>(rows: impl Iterator<Item = (&'a str, u32, [u32; 2])>) -> [u8; 32] {
    use sha2::Digest as _;

    let mut hasher = sha2::Sha256::new();
    hasher.update(b"frankensearch/quill-gauntlet/native-observation/v1\0");
    for (doc_id, score_bits, tie) in rows {
        hasher.update(
            u64::try_from(doc_id.len())
                .unwrap_or(u64::MAX)
                .to_be_bytes(),
        );
        hasher.update(doc_id.as_bytes());
        hasher.update(score_bits.to_be_bytes());
        hasher.update(tie[0].to_be_bytes());
        hasher.update(tie[1].to_be_bytes());
    }
    hasher.finalize().into()
}

/// Bind a certificate's provenance: physical snapshot, arm, both native
/// observations, and the one-shot same-snapshot authority — the latter is a
/// digest over the snapshot, both observations, and the exact count, so the
/// three facts can only be certified together.
fn certificate_provenance(
    snapshot_sha256: [u8; 32],
    arm: &str,
    ranked_observation_sha256: [u8; 32],
    expanded_observation_sha256: [u8; 32],
    exact_total: u64,
) -> crate::cutoff_certificate::CertificateProvenanceV1 {
    use sha2::Digest as _;

    let mut arm_hasher = sha2::Sha256::new();
    arm_hasher.update(b"frankensearch/quill-gauntlet/certificate-arm/v1\0");
    arm_hasher.update(arm.as_bytes());
    let arm_sha256: [u8; 32] = arm_hasher.finalize().into();

    let mut authority = sha2::Sha256::new();
    authority.update(b"frankensearch/quill-gauntlet/same-snapshot-authority/v1\0");
    authority.update(snapshot_sha256);
    authority.update(ranked_observation_sha256);
    authority.update(expanded_observation_sha256);
    authority.update(exact_total.to_be_bytes());
    let authority: [u8; 32] = authority.finalize().into();
    let mut same_snapshot_authority = [0_u8; 16];
    same_snapshot_authority.copy_from_slice(&authority[..16]);

    crate::cutoff_certificate::CertificateProvenanceV1 {
        snapshot_sha256,
        arm_sha256,
        ranked_observation_sha256,
        expanded_observation_sha256,
        same_snapshot_authority,
    }
}

/// Derive the same-snapshot certificate, or `None` when the native prefix
/// does not reach past the trailing score group (no claim is made). Any
/// other derivation failure means the observation itself is malformed.
fn derive_cutoff_certificate(
    exact_total: u64,
    offset: u64,
    limit: u64,
    expanded_score_bits: &[u32],
    provenance: crate::cutoff_certificate::CertificateProvenanceV1,
) -> Result<Option<crate::cutoff_certificate::CutoffCertificateV1>, GauntletError> {
    use crate::cutoff_certificate::{CutoffCertificateV1, CutoffDerivationError};

    match CutoffCertificateV1::from_native_prefix(
        exact_total,
        offset,
        limit,
        expanded_score_bits,
        provenance,
    ) {
        Ok(certificate) => Ok(Some(certificate)),
        Err(CutoffDerivationError::TrailingGroupTruncated { .. }) => Ok(None),
        Err(error) => Err(GauntletError::InvalidObservation {
            reason: format!("native evidence cannot certify the cutoff: {error}"),
        }),
    }
}

/// Certificate for one Tantivy oracle observation taken on one pinned searcher.
#[cfg(feature = "tantivy-oracle")]
fn oracle_cutoff_certificate(
    observation: &frankensearch_lexical::OracleQueryObservation,
    case: &DifferentialCase,
) -> Result<Option<crate::cutoff_certificate::CutoffCertificateV1>, GauntletError> {
    let rows = |hits: &[frankensearch_lexical::OracleRankedHit]| {
        observation_digest(hits.iter().map(|hit| {
            (
                hit.doc_id.as_str(),
                hit.score_bits,
                [hit.segment_ord, hit.segment_doc_id],
            )
        }))
    };
    let provenance = certificate_provenance(
        observation.searcher_sha256,
        "frankensearch-lexical/tantivy-index",
        rows(&observation.hits),
        rows(&observation.expanded_hits),
        u64::try_from(observation.total_count).unwrap_or(u64::MAX),
    );
    let expanded_bits = observation
        .expanded_hits
        .iter()
        .map(|hit| hit.score_bits)
        .collect::<Vec<_>>();
    derive_cutoff_certificate(
        u64::try_from(observation.total_count).unwrap_or(u64::MAX),
        case.offset,
        case.limit,
        &expanded_bits,
        provenance,
    )
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
        // The preparsed CASS path issues two separate searches; without a
        // pinned bundle it makes no same-snapshot completeness claim.
        quill_observation_from_results(
            &observed,
            &evidence,
            limit,
            offset,
            case.count_requested,
            None,
        )
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
    // The write-side trait, not `LexicalRead`: this constructor only indexes
    // and commits. The original import named the read trait, which is why this
    // `fuzz-harness`-gated function had never compiled (bd-jt7b2).
    use frankensearch_core::LexicalWrite;

    let config = QuillConfig {
        deterministic_ingest: true,
        ..QuillConfig::default()
    };
    // Capture the producer source once, then bind both engines to that exact
    // identity. This keeps every fresh external-fuzz pair in one provenance
    // universe even when a dirty checkout changes while a long fuzz job runs.
    let producer = GauntletProducerBuildIdentity::compiled()?;
    let mut subject = QuillSubject::in_memory_with_source(
        config,
        &producer.source_git_revision,
        producer.source_git_dirty,
    )?;
    let mut oracle = TantivyOracle::in_memory_scalar_g1a_with_source(
        &producer.source_git_revision,
        producer.source_git_dirty,
    )?;
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

/// Version of the typed-query fuzz input grammar and replay payload.
///
/// Version three binds the original corpus seed and manifest to the minimized
/// AST bytes.  Earlier payloads could render the minimized query against a
/// different regenerated corpus and therefore were not replayable evidence.
#[cfg(feature = "fuzz-harness")]
pub const TYPED_QUERY_FUZZ_REPLAY_SCHEMA_VERSION: u32 = 3;
/// Stable generator identity for the version-three typed-query grammar.
#[cfg(feature = "fuzz-harness")]
pub const TYPED_QUERY_FUZZ_GENERATOR_ID: &str = "typed-query-tree-fuzz-v3";
/// Maximum libFuzzer input length admitted by the typed-query target.
#[cfg(feature = "fuzz-harness")]
pub const TYPED_QUERY_FUZZ_MAX_INPUT_BYTES: usize = 64;
/// Bounded number of structural candidates considered by the typed shrinker.
#[cfg(feature = "fuzz-harness")]
pub const TYPED_QUERY_FUZZ_SHRINK_FUEL: usize = 64;

#[cfg(feature = "fuzz-harness")]
const TYPED_QUERY_FUZZ_DOCUMENT_COUNT: u64 = 16;
#[cfg(feature = "fuzz-harness")]
const TYPED_QUERY_FUZZ_VOCABULARY_SIZE: u32 = 32;
#[cfg(feature = "fuzz-harness")]
const TYPED_QUERY_FUZZ_DOCUMENT_BYTES: u32 = 256;
#[cfg(feature = "fuzz-harness")]
const TYPED_QUERY_FUZZ_OVERSIZED_TOKEN_BYTES: usize = 65_531;
#[cfg(feature = "fuzz-harness")]
const TYPED_QUERY_FUZZ_SEED_BASIS: u64 = 0x6273_6a77_0002_f29b;
#[cfg(feature = "fuzz-harness")]
const TYPED_QUERY_FUZZ_SEED_MULTIPLIER: u64 = 0x1000_0000_01b3;

/// Closed AST grammar consumed by the typed-query fuzz target.
///
/// This stays in the runnable gauntlet crate rather than the `test = false`
/// fuzz target, so artifact replay and hostile tests exercise the same byte
/// interpretation as libFuzzer.
#[cfg(feature = "fuzz-harness")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypedQueryTree {
    Empty,
    Term(u8),
    Phrase(u8, u8),
    NegatedTerm(u8),
    Boolean(u8, u8, u8),
    NestedBoolean(u8, u8, u8),
    Fielded(u8),
    BoostedTerm(u8),
    Slop(u8, u8),
    PhrasePrefix(u8, u8),
    UnterminatedPhrase(u8, u8),
    EscapedTerm(u8, u8),
    TrailingBoolean(u8),
    NonFiniteBoost(u8),
    OversizedToken(u8),
    MalformedNestedOperator(u8, u8, u8),
    MalformedOperator(u8, u8),
    MalformedField(u8),
    MalformedEscape(u8),
    MalformedBoost(u8),
    MalformedSlop(u8, u8),
    OovOnly(u8),
    MixedHitMiss(u8, u8),
}

#[cfg(feature = "fuzz-harness")]
impl TypedQueryTree {
    /// Decode the closed grammar from a bounded libFuzzer byte slice.
    #[must_use]
    pub fn from_input(input: &[u8]) -> Self {
        let byte = |index| input.get(index).copied().unwrap_or(0);
        let first = byte(1);
        let second = byte(2);
        match byte(0) % 23 {
            0 => Self::Empty,
            1 => Self::Term(first),
            2 => Self::Phrase(first, second),
            3 => Self::NegatedTerm(first),
            4 => Self::Boolean(first, second, byte(3)),
            5 => Self::NestedBoolean(first, second, byte(3)),
            6 => Self::Fielded(first),
            7 => Self::BoostedTerm(first),
            8 => Self::Slop(first, second),
            9 => Self::PhrasePrefix(first, second),
            10 => Self::UnterminatedPhrase(first, second),
            11 => Self::EscapedTerm(first, second),
            12 => Self::TrailingBoolean(first),
            13 => Self::NonFiniteBoost(first),
            14 => Self::OversizedToken(first),
            15 => Self::MalformedNestedOperator(first, second, byte(3)),
            16 => Self::MalformedOperator(first, second),
            17 => Self::MalformedField(first),
            18 => Self::MalformedEscape(first),
            19 => Self::MalformedBoost(first),
            20 => Self::MalformedSlop(first, second),
            21 => Self::OovOnly(first),
            _ => Self::MixedHitMiss(first, second),
        }
    }

    /// Canonical minimal byte representation of this tree.
    #[must_use]
    pub fn canonical_input(self) -> Vec<u8> {
        match self {
            Self::Empty => vec![0],
            Self::Term(first) => vec![1, first],
            Self::Phrase(first, second) => vec![2, first, second],
            Self::NegatedTerm(first) => vec![3, first],
            Self::Boolean(first, second, connective) => vec![4, first, second, connective],
            Self::NestedBoolean(first, second, third) => vec![5, first, second, third],
            Self::Fielded(first) => vec![6, first],
            Self::BoostedTerm(first) => vec![7, first],
            Self::Slop(first, second) => vec![8, first, second],
            Self::PhrasePrefix(first, second) => vec![9, first, second],
            Self::UnterminatedPhrase(first, second) => vec![10, first, second],
            Self::EscapedTerm(first, second) => vec![11, first, second],
            Self::TrailingBoolean(first) => vec![12, first],
            Self::NonFiniteBoost(first) => vec![13, first],
            Self::OversizedToken(first) => vec![14, first],
            Self::MalformedNestedOperator(first, second, third) => vec![15, first, second, third],
            Self::MalformedOperator(first, second) => vec![16, first, second],
            Self::MalformedField(first) => vec![17, first],
            Self::MalformedEscape(first) => vec![18, first],
            Self::MalformedBoost(first) => vec![19, first],
            Self::MalformedSlop(first, second) => vec![20, first, second],
            Self::OovOnly(first) => vec![21, first],
            Self::MixedHitMiss(first, second) => vec![22, first, second],
        }
    }

    /// Render the AST against the exact persisted corpus vocabulary.
    #[must_use]
    pub fn render(self, vocabulary: &[String]) -> String {
        assert!(
            !vocabulary.is_empty(),
            "typed-query rendering requires the replayed corpus vocabulary"
        );
        let word = |index: u8| vocabulary[usize::from(index) % vocabulary.len()].as_str();
        match self {
            Self::Empty => String::new(),
            Self::Term(term) => word(term).to_owned(),
            Self::Phrase(first, second) => format!("\"{} {}\"", word(first), word(second)),
            Self::NegatedTerm(term) => format!("-{}", word(term)),
            Self::Boolean(first, second, connective) => {
                let operator = if connective.is_multiple_of(2) {
                    "AND"
                } else {
                    "OR"
                };
                format!("{} {operator} {}", word(first), word(second))
            }
            Self::NestedBoolean(first, second, third) => {
                format!("({} OR {}) AND {}", word(first), word(second), word(third))
            }
            Self::Fielded(term) => format!("content:{}", word(term)),
            Self::BoostedTerm(term) => format!("{}^2", word(term)),
            Self::Slop(first, second) => format!("\"{} {}\"~1", word(first), word(second)),
            Self::PhrasePrefix(first, second) => format!("\"{} {}\"*", word(first), word(second)),
            Self::UnterminatedPhrase(first, second) => {
                format!("\"{} {}", word(first), word(second))
            }
            Self::EscapedTerm(first, second) => format!(r"{}\:{}", word(first), word(second)),
            Self::TrailingBoolean(term) => format!("{} OR", word(term)),
            Self::NonFiniteBoost(term) => {
                format!("{} {}^{}", word(term), word(term), "9".repeat(400))
            }
            Self::OversizedToken(term) => {
                let suffix_len =
                    TYPED_QUERY_FUZZ_OVERSIZED_TOKEN_BYTES.saturating_sub(word(term).len());
                format!("{}{}", word(term), "x".repeat(suffix_len))
            }
            Self::MalformedNestedOperator(first, second, third) => {
                format!(
                    "({} AND ({} OR )) AND {}",
                    word(first),
                    word(second),
                    word(third)
                )
            }
            Self::MalformedOperator(first, second) => {
                format!("{} AND OR {}", word(first), word(second))
            }
            Self::MalformedField(term) => format!(":{}", word(term)),
            Self::MalformedEscape(term) => format!(r"{}\", word(term)),
            Self::MalformedBoost(term) => format!("{}^not-a-number", word(term)),
            Self::MalformedSlop(first, second) => {
                format!("\"{} {}\"~not-a-number", word(first), word(second))
            }
            Self::OovOnly(miss) => format!("oovterm{miss}"),
            Self::MixedHitMiss(hit, miss) => format!("{} OR oovterm{miss}", word(hit)),
        }
    }

    /// Quill capability refusal asserted independently of parser recovery.
    #[must_use]
    pub const fn exact_refusal_detail(self) -> Option<&'static str> {
        match self {
            Self::Slop(..) => Some("phrase slop=1 prefix=false"),
            Self::PhrasePrefix(..) => Some("phrase slop=0 prefix=true"),
            _ => None,
        }
    }

    #[must_use]
    pub const fn is_nonfinite_boost(self) -> bool {
        matches!(self, Self::NonFiniteBoost(..))
    }

    #[must_use]
    pub const fn is_reviewed_oversized_lowering(self) -> bool {
        matches!(self, Self::OversizedToken(..))
    }

    /// Grammar that Quill deliberately repairs with `parse_lenient`.
    #[must_use]
    pub const fn is_malformed(self) -> bool {
        matches!(
            self,
            Self::UnterminatedPhrase(..)
                | Self::TrailingBoolean(..)
                | Self::MalformedNestedOperator(..)
                | Self::MalformedOperator(..)
                | Self::MalformedField(..)
                | Self::MalformedEscape(..)
                | Self::MalformedBoost(..)
                | Self::MalformedSlop(..)
        )
    }

    /// Strictly smaller candidates for structural failure shrinking.
    #[must_use]
    pub fn shrink_candidates(self) -> Vec<Self> {
        let mut candidates = match self {
            Self::Empty => Vec::new(),
            Self::Term(term) => vec![(term != 0).then_some(Self::Term(0))],
            Self::Phrase(first, second) => vec![
                Some(Self::Term(first)),
                Some(Self::Term(second)),
                (first != 0 || second != 0).then_some(Self::Phrase(0, 0)),
            ],
            Self::NegatedTerm(term) => vec![
                Some(Self::Term(term)),
                (term != 0).then_some(Self::NegatedTerm(0)),
            ],
            Self::Boolean(first, second, _) => vec![
                Some(Self::Term(first)),
                Some(Self::Term(second)),
                (first != 0 || second != 0).then_some(Self::Boolean(0, 0, 0)),
            ],
            Self::NestedBoolean(first, second, third) => vec![
                Some(Self::Boolean(first, second, 1)),
                Some(Self::Boolean(second, third, 0)),
                Some(Self::Term(third)),
                (first != 0 || second != 0 || third != 0).then_some(Self::NestedBoolean(0, 0, 0)),
            ],
            Self::Fielded(term) => vec![
                Some(Self::Term(term)),
                (term != 0).then_some(Self::Fielded(0)),
            ],
            Self::BoostedTerm(term) => vec![
                Some(Self::Term(term)),
                (term != 0).then_some(Self::BoostedTerm(0)),
            ],
            Self::Slop(first, second) => {
                vec![(first != 0 || second != 0).then_some(Self::Slop(0, 0))]
            }
            Self::PhrasePrefix(first, second) => {
                vec![(first != 0 || second != 0).then_some(Self::PhrasePrefix(0, 0))]
            }
            Self::UnterminatedPhrase(first, second) => {
                vec![(first != 0 || second != 0).then_some(Self::UnterminatedPhrase(0, 0))]
            }
            Self::EscapedTerm(first, second) => {
                vec![(first != 0 || second != 0).then_some(Self::EscapedTerm(0, 0))]
            }
            Self::TrailingBoolean(term) => vec![(term != 0).then_some(Self::TrailingBoolean(0))],
            Self::NonFiniteBoost(term) => vec![(term != 0).then_some(Self::NonFiniteBoost(0))],
            Self::OversizedToken(term) => vec![(term != 0).then_some(Self::OversizedToken(0))],
            Self::MalformedNestedOperator(first, second, third) => vec![
                Some(Self::MalformedOperator(first, second)),
                (first != 0 || second != 0 || third != 0)
                    .then_some(Self::MalformedNestedOperator(0, 0, 0)),
            ],
            Self::MalformedOperator(first, second) => {
                vec![(first != 0 || second != 0).then_some(Self::MalformedOperator(0, 0))]
            }
            Self::MalformedField(term) => vec![(term != 0).then_some(Self::MalformedField(0))],
            Self::MalformedEscape(term) => vec![(term != 0).then_some(Self::MalformedEscape(0))],
            Self::MalformedBoost(term) => vec![(term != 0).then_some(Self::MalformedBoost(0))],
            Self::MalformedSlop(first, second) => {
                vec![(first != 0 || second != 0).then_some(Self::MalformedSlop(0, 0))]
            }
            Self::OovOnly(miss) => vec![(miss != 0).then_some(Self::OovOnly(0))],
            Self::MixedHitMiss(hit, miss) => vec![
                Some(Self::OovOnly(miss)),
                (hit != 0 || miss != 0).then_some(Self::MixedHitMiss(0, 0)),
            ],
        }
        .into_iter()
        .flatten()
        .filter(|candidate| *candidate != self)
        .collect::<Vec<_>>();
        candidates.sort_unstable_by_key(|candidate| format!("{candidate:?}"));
        candidates.dedup();
        candidates
    }
}

/// Exact comparison signature that a minimized fuzz replay must retain.
#[cfg(feature = "fuzz-harness")]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedQueryFailureFingerprint {
    pub status: ComparisonStatus,
    pub rank_class: RankClass,
    pub first_divergence: Option<String>,
    pub divergences: Vec<Divergence>,
}

#[cfg(feature = "fuzz-harness")]
impl TypedQueryFailureFingerprint {
    /// Capture the entire semantic failure signature, not just its first class.
    #[must_use]
    pub fn from_report(report: &ComparisonReport) -> Self {
        Self {
            status: report.status,
            rank_class: report.rank_class,
            first_divergence: report.first_divergence.clone(),
            divergences: report.divergences.clone(),
        }
    }
}

/// Return the fix-required class carried by a failed report.
///
/// Score epsilon and tie order are accepted automatic classes and cannot name
/// an emitted minimized divergence artifact by themselves.
#[cfg(feature = "fuzz-harness")]
pub fn typed_query_failure_divergence_class(
    report: &ComparisonReport,
) -> Result<DivergenceClass, GauntletError> {
    report
        .divergences
        .iter()
        .find(|divergence| {
            !matches!(
                divergence.class,
                DivergenceClass::ScoreEpsilon | DivergenceClass::TieOrder
            )
        })
        .map(|divergence| divergence.class)
        .ok_or_else(|| GauntletError::InvalidObservation {
            reason: "failed typed-query fuzz comparison lacks a non-automatic divergence class"
                .to_owned(),
        })
}

/// Fully materialized deterministic input to one typed-query differential run.
#[cfg(feature = "fuzz-harness")]
#[derive(Clone, Debug)]
pub struct TypedQueryFuzzWorkload {
    pub original_input: Vec<u8>,
    pub ast: TypedQueryTree,
    pub seed: u64,
    pub corpus_spec: SyntheticCorpusSpec,
    pub corpus_manifest: CorpusManifest,
    pub corpus_manifest_hash: String,
    pub documents: Vec<GeneratedDocument>,
    pub vocabulary: Vec<String>,
    pub case: DifferentialCase,
}

#[cfg(feature = "fuzz-harness")]
impl TypedQueryFuzzWorkload {
    /// Build a case for a shrunk AST against this exact original corpus.
    #[must_use]
    pub fn case_for_ast(&self, ast: TypedQueryTree) -> DifferentialCase {
        typed_query_differential_case(
            &self.original_input,
            self.seed,
            ast,
            ast.render(&self.vocabulary),
            &self.corpus_manifest_hash,
        )
    }

    /// Stable provenance text used only for diagnostic failures in the target.
    #[must_use]
    pub fn provenance_for_ast(&self, ast: TypedQueryTree) -> String {
        typed_query_provenance(
            &self.original_input,
            self.seed,
            ast,
            &self.corpus_manifest_hash,
        )
    }
}

/// Construct the deterministic corpus, vocabulary, and differential case for
/// a bounded libFuzzer input.
///
/// The input bytes are retained even when several byte strings decode to one
/// AST: their seed determines the corpus and is thus part of replay identity.
#[cfg(feature = "fuzz-harness")]
pub fn materialize_typed_query_fuzz_workload(
    input: &[u8],
) -> Result<TypedQueryFuzzWorkload, GauntletError> {
    if input.len() > TYPED_QUERY_FUZZ_MAX_INPUT_BYTES {
        return Err(GauntletError::InvalidGenerator {
            reason: format!(
                "typed-query fuzz input exceeds {} bytes",
                TYPED_QUERY_FUZZ_MAX_INPUT_BYTES
            ),
        });
    }
    let seed = typed_query_fuzz_seed(input);
    materialize_typed_query_fuzz_workload_with_recipe(
        input,
        seed,
        typed_query_fuzz_corpus_spec(seed),
    )
}

#[cfg(feature = "fuzz-harness")]
fn materialize_typed_query_fuzz_workload_with_recipe(
    original_input: &[u8],
    seed: u64,
    corpus_spec: SyntheticCorpusSpec,
) -> Result<TypedQueryFuzzWorkload, GauntletError> {
    if original_input.len() > TYPED_QUERY_FUZZ_MAX_INPUT_BYTES {
        return Err(GauntletError::InvalidGenerator {
            reason: "stored typed-query fuzz input exceeds the schema-v3 byte bound".to_owned(),
        });
    }
    if seed != typed_query_fuzz_seed(original_input) {
        return Err(GauntletError::ManifestMismatch {
            reason: "stored typed-query fuzz seed does not match the original input bytes"
                .to_owned(),
        });
    }
    if corpus_spec != typed_query_fuzz_corpus_spec(seed) {
        return Err(GauntletError::ManifestMismatch {
            reason: "stored typed-query fuzz corpus recipe is not the schema-v3 fixed recipe"
                .to_owned(),
        });
    }
    let corpus = SyntheticCorpus::new(corpus_spec.clone())?;
    let corpus_manifest = corpus.manifest()?;
    corpus.verify_manifest(&corpus_manifest)?;
    let corpus_manifest_hash = corpus_manifest.manifest_hash()?;
    let documents = corpus.iter().collect::<Vec<_>>();
    let vocabulary = typed_query_fuzz_vocabulary(&documents)?;
    let ast = TypedQueryTree::from_input(original_input);
    let case = typed_query_differential_case(
        original_input,
        seed,
        ast,
        ast.render(&vocabulary),
        &corpus_manifest_hash,
    );
    Ok(TypedQueryFuzzWorkload {
        original_input: original_input.to_vec(),
        ast,
        seed,
        corpus_spec,
        corpus_manifest,
        corpus_manifest_hash,
        documents,
        vocabulary,
        case,
    })
}

/// Fixed corpus recipe that accompanies every schema-v3 byte stream.
#[cfg(feature = "fuzz-harness")]
#[must_use]
pub const fn typed_query_fuzz_corpus_spec(seed: u64) -> SyntheticCorpusSpec {
    SyntheticCorpusSpec {
        seed,
        document_count: TYPED_QUERY_FUZZ_DOCUMENT_COUNT,
        vocabulary_size: TYPED_QUERY_FUZZ_VOCABULARY_SIZE,
        zipf_exponent: ZipfExponent::S11,
        max_document_bytes: TYPED_QUERY_FUZZ_DOCUMENT_BYTES,
    }
}

/// Deterministic 64-bit seed for the full raw libFuzzer input.
#[cfg(feature = "fuzz-harness")]
#[must_use]
pub fn typed_query_fuzz_seed(input: &[u8]) -> u64 {
    input
        .iter()
        .fold(TYPED_QUERY_FUZZ_SEED_BASIS, |state, byte| {
            state
                .rotate_left(5)
                .wrapping_mul(TYPED_QUERY_FUZZ_SEED_MULTIPLIER)
                ^ u64::from(*byte)
        })
}

/// Derive the exact regular-corpus vocabulary used to render a fuzz AST.
#[cfg(feature = "fuzz-harness")]
pub fn typed_query_fuzz_vocabulary(
    documents: &[GeneratedDocument],
) -> Result<Vec<String>, GauntletError> {
    let mut vocabulary = documents
        .iter()
        .filter(|document| document.pathology.is_none())
        .flat_map(|document| document.content.split_whitespace())
        .filter(|word| {
            word.strip_prefix("term").is_some_and(|suffix| {
                !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
            })
        })
        .map(str::to_owned)
        .collect::<Vec<_>>();
    vocabulary.sort_unstable();
    vocabulary.dedup();
    if vocabulary.is_empty() {
        return Err(GauntletError::ManifestMismatch {
            reason: "typed-query fuzz corpus has no searchable regular terms".to_owned(),
        });
    }
    Ok(vocabulary)
}

#[cfg(feature = "fuzz-harness")]
fn typed_query_differential_case(
    original_input: &[u8],
    seed: u64,
    ast: TypedQueryTree,
    query: String,
    corpus_manifest_hash: &str,
) -> DifferentialCase {
    let input_hex = typed_query_hex(original_input);
    DifferentialCase {
        fixture_id: format!("typed-query-tree-v3-{input_hex}-{ast:?}"),
        query,
        limit: 20,
        offset: 0,
        tie_expansion_limit: 256,
        count_requested: true,
        snippet_max_chars: None,
        metadata: DifferentialCaseMetadata {
            generator_id: Some(typed_query_provenance(
                original_input,
                seed,
                ast,
                corpus_manifest_hash,
            )),
            generator_seed: Some(seed),
            corpus_hash: Some(corpus_manifest_hash.to_owned()),
        },
    }
}

#[cfg(feature = "fuzz-harness")]
fn typed_query_provenance(
    original_input: &[u8],
    seed: u64,
    ast: TypedQueryTree,
    corpus_manifest_hash: &str,
) -> String {
    format!(
        "generator={TYPED_QUERY_FUZZ_GENERATOR_ID};schema={TYPED_QUERY_FUZZ_REPLAY_SCHEMA_VERSION};input={};corpus_seed={seed:016x};docs={TYPED_QUERY_FUZZ_DOCUMENT_COUNT};vocab={TYPED_QUERY_FUZZ_VOCABULARY_SIZE};zipf=s11;bytes={TYPED_QUERY_FUZZ_DOCUMENT_BYTES};ast={ast:?};manifest={corpus_manifest_hash}",
        typed_query_hex(original_input),
    )
}

#[cfg(feature = "fuzz-harness")]
fn typed_query_hex(input: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(input.len().saturating_mul(2));
    for byte in input {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

/// Exact oracle result intentionally paired with Quill's lenient syntax repair.
#[cfg(feature = "fuzz-harness")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TypedQueryOracleBehavior {
    /// The pinned Tantivy oracle accepted the query and exposed no lowering
    /// record in its normalized observation surface.
    AcceptedWithoutAstDifferences,
}

/// Typed classification of a malformed grammar input, rather than an assumed
/// shared parser error.  The recovered AST and all Quill diagnostic kinds are
/// retained so the fuzz lane asserts the production `parse_lenient` contract.
#[cfg(feature = "fuzz-harness")]
#[derive(Clone, Debug, PartialEq)]
pub struct TypedQueryLenientAsymmetry {
    pub recovered_quill_ast: Query,
    pub quill_diagnostic_kinds: Vec<QueryDiagnosticKind>,
    pub oracle_behavior: TypedQueryOracleBehavior,
}

#[cfg(feature = "fuzz-harness")]
impl QuillSubject {
    /// Parse with the same default schema as the scalar G1a subject without
    /// turning malformed syntax into a harness failure.
    ///
    /// This API is deliberately fuzz-harness-only: regular observations return
    /// normalized result evidence, while this corrective lane must also assert
    /// the raw recovered Quill AST and diagnostics.
    pub fn parse_typed_query_lenient(&self, query: &str) -> Result<ParsedQuery, GauntletError> {
        self.require_committed()?;
        let parser = DefaultQueryParser::new(DEFAULT_SCHEMA).map_err(|error| {
            GauntletError::InvalidContract {
                reason: format!("cannot bind the scalar G1a lenient parser: {error}"),
            }
        })?;
        Ok(parser.parse_lenient(query))
    }

    /// Classify the expected malformed-syntax asymmetry after both live engine
    /// observations completed successfully.
    pub fn classify_typed_query_lenient_asymmetry(
        &self,
        ast: TypedQueryTree,
        query: &str,
        oracle_observation: &EngineObservation,
    ) -> Result<TypedQueryLenientAsymmetry, GauntletError> {
        if !ast.is_malformed() {
            return Err(GauntletError::InvalidCase {
                reason: "lenient syntax classification requires an explicit malformed typed AST"
                    .to_owned(),
            });
        }
        let parsed = self.parse_typed_query_lenient(query)?;
        let quill_diagnostic_kinds = parsed
            .diagnostics
            .iter()
            .map(|diagnostic| diagnostic.kind)
            .collect::<Vec<_>>();
        if quill_diagnostic_kinds.is_empty() {
            return Err(GauntletError::InvalidObservation {
                reason: "Quill parse_lenient returned no recovery diagnostics for an explicit malformed typed AST"
                    .to_owned(),
            });
        }
        if !oracle_observation.ast_differences.is_empty() {
            return Err(GauntletError::InvalidObservation {
                reason: "pinned Tantivy oracle must accept malformed typed-query grammar without normalized AST differences"
                    .to_owned(),
            });
        }
        Ok(TypedQueryLenientAsymmetry {
            recovered_quill_ast: parsed.query,
            quill_diagnostic_kinds,
            oracle_behavior: TypedQueryOracleBehavior::AcceptedWithoutAstDifferences,
        })
    }
}

/// Durable minimized replay payload for one unclassified typed-query failure.
///
/// The original input is not discarded: it fixes the corpus seed.  The
/// minimized bytes are stored separately and must reconstruct the minimized
/// AST and query against that exact regenerated corpus.
#[cfg(feature = "fuzz-harness")]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedQueryFuzzReplay {
    pub schema_version: u32,
    pub generator_id: String,
    pub original_input: Vec<u8>,
    pub original_seed: u64,
    pub corpus_spec: SyntheticCorpusSpec,
    pub corpus_manifest: CorpusManifest,
    pub corpus_manifest_hash: String,
    pub vocabulary: Vec<String>,
    pub minimized_input: Vec<u8>,
    pub minimized_ast: TypedQueryTree,
    pub minimized_query: String,
    pub fingerprint: TypedQueryFailureFingerprint,
    pub divergence_class: DivergenceClass,
}

/// An owned, descriptor-bound typed-query replay.
///
/// This capability deliberately exposes replay reconstruction rather than an
/// authenticated filesystem path. Its held sidecar-directory and regular-file
/// descriptors remain the authority when a consumer asks to reconstruct the
/// minimized workload.
#[cfg(feature = "fuzz-harness")]
pub struct TypedQueryFuzzReplayArtifact {
    replay: TypedQueryFuzzReplay,
    replay_directory: PinnedDirectory,
    filename: std::ffi::OsString,
    file: PinnedRegularFile,
}

#[cfg(feature = "fuzz-harness")]
const TYPED_QUERY_FUZZ_REPLAY_EXTENSION: &str = "json";
#[cfg(feature = "fuzz-harness")]
const TYPED_QUERY_FUZZ_REPLAY_DIRECTORY: &str = "typed_query_tree";
/// A replay contains a 64-byte fuzz input, a fixed sixteen-document manifest,
/// and one typed AST.  This bounded reader fails closed before allocating for
/// an untrusted sidecar file.
#[cfg(feature = "fuzz-harness")]
const MAX_TYPED_QUERY_FUZZ_REPLAY_BYTES: u64 = 1024 * 1024;

#[cfg(all(test, feature = "fuzz-harness"))]
thread_local! {
    static TYPED_QUERY_FUZZ_REPLAY_FINAL_BINDING_HOOK:
        std::cell::RefCell<Option<Box<dyn FnOnce(&std::path::Path)>>> =
            std::cell::RefCell::new(None);
    static TYPED_QUERY_FUZZ_REPLAY_POST_DISPLAY_VERIFICATION_HOOK:
        std::cell::RefCell<Option<Box<dyn FnOnce(&std::path::Path)>>> =
            std::cell::RefCell::new(None);
}

/// Test-only interlock for a deterministic mutation after replay I/O and
/// before the final descriptor-to-dirent binding.  Production performs no
/// callback at this point.
#[cfg(feature = "fuzz-harness")]
fn typed_query_fuzz_replay_before_final_binding(path: &std::path::Path) {
    #[cfg(all(test, feature = "fuzz-harness"))]
    TYPED_QUERY_FUZZ_REPLAY_FINAL_BINDING_HOOK.with(|slot| {
        if let Some(hook) = slot.borrow_mut().take() {
            hook(path);
        }
    });
    #[cfg(not(test))]
    let _ = path;
}

/// Test-only interlock after the ambient display-path check and before the
/// final descriptor-relative entry authentication. Production performs no
/// callback at this point.
#[cfg(feature = "fuzz-harness")]
fn typed_query_fuzz_replay_after_display_verification(path: &std::path::Path) {
    #[cfg(all(test, feature = "fuzz-harness"))]
    TYPED_QUERY_FUZZ_REPLAY_POST_DISPLAY_VERIFICATION_HOOK.with(|slot| {
        if let Some(hook) = slot.borrow_mut().take() {
            hook(path);
        }
    });
    #[cfg(not(test))]
    let _ = path;
}

#[cfg(all(test, feature = "fuzz-harness"))]
fn install_typed_query_fuzz_replay_final_binding_hook(
    hook: impl FnOnce(&std::path::Path) + 'static,
) {
    TYPED_QUERY_FUZZ_REPLAY_FINAL_BINDING_HOOK.with(|slot| {
        assert!(
            slot.borrow().is_none(),
            "typed-query replay binding hook must be consumed before replacement"
        );
        *slot.borrow_mut() = Some(Box::new(hook));
    });
}

#[cfg(all(test, feature = "fuzz-harness"))]
fn install_typed_query_fuzz_replay_post_display_verification_hook(
    hook: impl FnOnce(&std::path::Path) + 'static,
) {
    TYPED_QUERY_FUZZ_REPLAY_POST_DISPLAY_VERIFICATION_HOOK.with(|slot| {
        assert!(
            slot.borrow().is_none(),
            "typed-query replay post-display hook must be consumed before replacement"
        );
        *slot.borrow_mut() = Some(Box::new(hook));
    });
}

#[cfg(feature = "fuzz-harness")]
impl TypedQueryFuzzReplay {
    /// Create and immediately validate a minimized artifact from a live run.
    pub fn from_failure(
        workload: &TypedQueryFuzzWorkload,
        minimized_ast: TypedQueryTree,
        minimized_report: &ComparisonReport,
    ) -> Result<Self, GauntletError> {
        if minimized_report.status != ComparisonStatus::Failed {
            return Err(GauntletError::InvalidObservation {
                reason: "only a failed comparison may become a typed-query minimized replay"
                    .to_owned(),
            });
        }
        let fingerprint = TypedQueryFailureFingerprint::from_report(minimized_report);
        let divergence_class = typed_query_failure_divergence_class(minimized_report)?;
        let replay = Self {
            schema_version: TYPED_QUERY_FUZZ_REPLAY_SCHEMA_VERSION,
            generator_id: TYPED_QUERY_FUZZ_GENERATOR_ID.to_owned(),
            original_input: workload.original_input.clone(),
            original_seed: workload.seed,
            corpus_spec: workload.corpus_spec.clone(),
            corpus_manifest: workload.corpus_manifest.clone(),
            corpus_manifest_hash: workload.corpus_manifest_hash.clone(),
            vocabulary: workload.vocabulary.clone(),
            minimized_input: minimized_ast.canonical_input(),
            minimized_ast,
            minimized_query: minimized_ast.render(&workload.vocabulary),
            fingerprint,
            divergence_class,
        };
        replay.replay_workload()?;
        Ok(replay)
    }

    /// Deterministically rebuild the corpus and minimized case from durable
    /// replay bytes, refusing any seed, corpus, AST, query, or fingerprint
    /// mutation before an engine is invoked.
    pub fn replay_workload(&self) -> Result<TypedQueryFuzzWorkload, GauntletError> {
        if self.schema_version != TYPED_QUERY_FUZZ_REPLAY_SCHEMA_VERSION
            || self.generator_id != TYPED_QUERY_FUZZ_GENERATOR_ID
        {
            return Err(GauntletError::ManifestMismatch {
                reason: "unsupported typed-query fuzz replay schema or generator".to_owned(),
            });
        }
        if self.original_seed != typed_query_fuzz_seed(&self.original_input) {
            return Err(GauntletError::ManifestMismatch {
                reason: "typed-query replay original seed does not match original input".to_owned(),
            });
        }
        if TypedQueryTree::from_input(&self.minimized_input) != self.minimized_ast {
            return Err(GauntletError::ManifestMismatch {
                reason: "typed-query replay minimized bytes do not reconstruct the minimized AST"
                    .to_owned(),
            });
        }
        if self.fingerprint.status != ComparisonStatus::Failed
            || self.divergence_class
                != typed_query_failure_divergence_class_from_fingerprint(&self.fingerprint)?
        {
            return Err(GauntletError::ManifestMismatch {
                reason: "typed-query replay fingerprint and divergence class are inconsistent"
                    .to_owned(),
            });
        }
        let mut workload = materialize_typed_query_fuzz_workload_with_recipe(
            &self.original_input,
            self.original_seed,
            self.corpus_spec.clone(),
        )?;
        if workload.corpus_manifest != self.corpus_manifest
            || workload.corpus_manifest_hash != self.corpus_manifest_hash
        {
            return Err(GauntletError::ManifestMismatch {
                reason: "typed-query replay regenerated corpus does not match its stored manifest identity"
                    .to_owned(),
            });
        }
        if workload.vocabulary != self.vocabulary {
            return Err(GauntletError::ManifestMismatch {
                reason:
                    "typed-query replay regenerated vocabulary does not match stored vocabulary"
                        .to_owned(),
            });
        }
        let expected_query = self.minimized_ast.render(&workload.vocabulary);
        if expected_query != self.minimized_query {
            return Err(GauntletError::ManifestMismatch {
                reason:
                    "typed-query replay minimized query does not match minimized AST and vocabulary"
                        .to_owned(),
            });
        }
        workload.ast = self.minimized_ast;
        workload.case = workload.case_for_ast(self.minimized_ast);
        if workload.case.query != self.minimized_query {
            return Err(GauntletError::ManifestMismatch {
                reason: "typed-query replay rebuilt case query differs from stored minimized query"
                    .to_owned(),
            });
        }
        Ok(workload)
    }

    /// Collision-resistant artifact key that visibly binds corpus and exact
    /// failure signature as well as the full replay payload.
    pub fn artifact_key(&self) -> Result<String, GauntletError> {
        let canonical_bytes = self.canonical_bytes()?;
        self.artifact_key_from_canonical_bytes(&canonical_bytes)
    }

    fn artifact_key_from_canonical_bytes(
        &self,
        canonical_bytes: &[u8],
    ) -> Result<String, GauntletError> {
        let fingerprint_bytes = serde_json::to_vec(&self.fingerprint)?;
        Ok(format!(
            "{}-{}-{}",
            self.corpus_manifest_hash,
            typed_query_sha256(b"fingerprint-v1\0", &fingerprint_bytes),
            typed_query_sha256(b"payload-v3\0", canonical_bytes),
        ))
    }

    /// Canonical JSON bytes validated by the real replay entrypoint.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, GauntletError> {
        self.replay_workload()?;
        Ok(serde_json::to_vec(self)?)
    }
}

#[cfg(feature = "fuzz-harness")]
impl TypedQueryFuzzReplayArtifact {
    /// Reconstruct the minimized workload through the still-owned descriptor
    /// binding, rejecting a later replacement of the original sidecar entry.
    pub fn replay_workload(&self) -> Result<TypedQueryFuzzWorkload, GauntletError> {
        self.replay_directory
            .authenticate_regular_child(&self.filename, &self.file)?;
        self.replay.replay_workload()
    }

    /// Return the content-addressed key for diagnostics without exposing an
    /// ambient path as an authenticated replay handle.
    pub fn artifact_key(&self) -> Result<String, GauntletError> {
        self.replay.artifact_key()
    }
}

/// Persist a replay under a key that includes corpus and failure identity.
///
/// The returned capability owns the canonical no-follow directory and regular
/// file descriptors. Consumers must use its replay method rather than treating
/// an ambient path as authenticated after this call returns.
#[cfg(feature = "fuzz-harness")]
pub fn persist_typed_query_fuzz_replay(
    root: &std::path::Path,
    replay: &TypedQueryFuzzReplay,
) -> Result<TypedQueryFuzzReplayArtifact, GauntletError> {
    let bytes = replay.canonical_bytes()?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_TYPED_QUERY_FUZZ_REPLAY_BYTES {
        return Err(GauntletError::InvalidCase {
            reason: "typed-query replay canonical payload exceeds its bounded sidecar budget"
                .to_owned(),
        });
    }
    let key = replay.artifact_key_from_canonical_bytes(&bytes)?;
    let filename = typed_query_fuzz_replay_filename(&key);
    let root_directory = PinnedDirectory::ensure_path(root)?;
    let replay_directory = root_directory
        .ensure_child_directory(std::ffi::OsStr::new(TYPED_QUERY_FUZZ_REPLAY_DIRECTORY))?;
    let path = root.join(TYPED_QUERY_FUZZ_REPLAY_DIRECTORY).join(&filename);
    let file = match replay_directory
        .write_regular_create_new_and_read_back(std::ffi::OsStr::new(&filename), &bytes)?
    {
        Some((read_back, file)) => {
            if read_back != bytes {
                return Err(GauntletError::UnsafeStorePath { path: path.clone() });
            }
            file
        }
        None => {
            let (existing, file) = replay_directory.read_regular_bounded_open(
                std::ffi::OsStr::new(&filename),
                MAX_TYPED_QUERY_FUZZ_REPLAY_BYTES,
            )?;
            if existing != bytes {
                return Err(GauntletError::ArtifactCollision { path });
            }
            file
        }
    };
    bind_typed_query_fuzz_replay_final_entry(
        &replay_directory,
        std::ffi::OsStr::new(&filename),
        &file,
        &path,
    )?;
    // A no-replace loser has authenticated the winner's complete bytes, but
    // its final directory entry remains crash-vulnerable until this held
    // descriptor is synced.  Keep the sync after the final binding for both
    // outcomes so neither caller can report a durable replay prematurely.
    replay_directory.sync_directory()?;
    Ok(TypedQueryFuzzReplayArtifact {
        replay: replay.clone(),
        replay_directory,
        filename: filename.into(),
        file,
    })
}

/// Load and validate a minimized replay into an owned descriptor-bound
/// capability before returning it to a runner.
#[cfg(feature = "fuzz-harness")]
pub fn load_typed_query_fuzz_replay(
    path: &std::path::Path,
) -> Result<TypedQueryFuzzReplayArtifact, GauntletError> {
    let filename = path
        .file_name()
        .ok_or_else(|| GauntletError::ManifestMismatch {
            reason: "typed-query replay path has no final filename component".to_owned(),
        })?;
    if path.extension() != Some(std::ffi::OsStr::new(TYPED_QUERY_FUZZ_REPLAY_EXTENSION)) {
        return Err(GauntletError::ManifestMismatch {
            reason: "typed-query replay path uses an unknown extension".to_owned(),
        });
    }
    let parent = path
        .parent()
        .ok_or_else(|| GauntletError::ManifestMismatch {
            reason: "typed-query replay path has no parent directory".to_owned(),
        })?;
    if parent.file_name() != Some(std::ffi::OsStr::new(TYPED_QUERY_FUZZ_REPLAY_DIRECTORY)) {
        return Err(GauntletError::ManifestMismatch {
            reason: "typed-query replay must reside directly in typed_query_tree".to_owned(),
        });
    }
    let replay_directory = PinnedDirectory::open_path(parent)?;
    load_typed_query_fuzz_replay_from_directory(replay_directory, filename, path)
}

/// Authenticate a replay through an already-owned sidecar-directory capability.
/// The public path entrypoint opens that capability with a no-follow component
/// walk before reaching this shared validation path.
#[cfg(feature = "fuzz-harness")]
fn load_typed_query_fuzz_replay_from_directory(
    replay_directory: PinnedDirectory,
    filename: &std::ffi::OsStr,
    path: &std::path::Path,
) -> Result<TypedQueryFuzzReplayArtifact, GauntletError> {
    let (stored_bytes, file) =
        replay_directory.read_regular_bounded_open(filename, MAX_TYPED_QUERY_FUZZ_REPLAY_BYTES)?;
    let replay = serde_json::from_slice::<TypedQueryFuzzReplay>(&stored_bytes)?;
    let canonical_bytes = replay.canonical_bytes()?;
    if stored_bytes != canonical_bytes {
        return Err(GauntletError::ManifestMismatch {
            reason: "typed-query replay bytes are not canonical".to_owned(),
        });
    }
    let key = replay.artifact_key_from_canonical_bytes(&canonical_bytes)?;
    let expected_filename = typed_query_fuzz_replay_filename(&key);
    let actual_filename = filename
        .to_str()
        .ok_or_else(|| GauntletError::ManifestMismatch {
            reason: "typed-query replay path has no UTF-8 final filename component".to_owned(),
        })?;
    if actual_filename != expected_filename {
        return Err(GauntletError::ManifestMismatch {
            reason: format!(
                "typed-query replay filename must be {expected_filename}, found {actual_filename}"
            ),
        });
    }
    bind_typed_query_fuzz_replay_final_entry(&replay_directory, filename, &file, path)?;
    Ok(TypedQueryFuzzReplayArtifact {
        replay,
        replay_directory,
        filename: filename.to_owned(),
        file,
    })
}

/// Bind the just-read or just-created replay FD to the exact final sidecar
/// name.  The final repeated authentication follows the display-path identity
/// check so this routine reports either a parent-directory substitution or a
/// final-entry replacement before the public operation succeeds.
#[cfg(feature = "fuzz-harness")]
fn bind_typed_query_fuzz_replay_final_entry(
    replay_directory: &PinnedDirectory,
    filename: &std::ffi::OsStr,
    file: &PinnedRegularFile,
    path: &std::path::Path,
) -> Result<(), GauntletError> {
    typed_query_fuzz_replay_before_final_binding(path);
    replay_directory.authenticate_regular_child(filename, file)?;
    replay_directory.verify_display_path_identity()?;
    typed_query_fuzz_replay_after_display_verification(path);
    replay_directory.authenticate_regular_child(filename, file)
}

#[cfg(feature = "fuzz-harness")]
fn typed_query_fuzz_replay_filename(key: &str) -> String {
    format!("{key}.{TYPED_QUERY_FUZZ_REPLAY_EXTENSION}")
}

#[cfg(feature = "fuzz-harness")]
fn typed_query_failure_divergence_class_from_fingerprint(
    fingerprint: &TypedQueryFailureFingerprint,
) -> Result<DivergenceClass, GauntletError> {
    fingerprint
        .divergences
        .iter()
        .find(|divergence| {
            !matches!(
                divergence.class,
                DivergenceClass::ScoreEpsilon | DivergenceClass::TieOrder
            )
        })
        .map(|divergence| divergence.class)
        .ok_or_else(|| GauntletError::ManifestMismatch {
            reason: "typed-query replay fingerprint lacks a non-automatic divergence class"
                .to_owned(),
        })
}

#[cfg(feature = "fuzz-harness")]
fn typed_query_sha256(domain: &[u8], bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut output = String::with_capacity(digest.len().saturating_mul(2));
    for byte in digest {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[cfg(feature = "fuzz-harness")]
impl QuillSubject {
    /// Build fresh scalar-G1a adapters for one external fuzz shrink attempt.
    ///
    /// An external `cargo-fuzz` target cannot safely reproduce the private
    /// claim/index/commit lifecycle itself.  These factories deliberately
    /// return fresh, uncommitted engines for every [`crate::ShrinkDriver`]
    /// candidate, leaving the shrinker to own the complete campaign
    /// lifecycle.  This is distinct from [`scalar_g1a_fuzz_pair`], which
    /// returns one already-committed pair for the initial observation.
    #[must_use]
    pub fn scalar_g1a_fuzz_shrink_factories()
    -> (crate::ShrinkEngineFactory, crate::ShrinkEngineFactory) {
        use crate::DifferentialCampaignEngine;

        let config = QuillConfig {
            deterministic_ingest: true,
            ..QuillConfig::default()
        };
        let producer = GauntletProducerBuildIdentity::compiled()
            .expect("capture the source identity for fuzz shrink factories");
        let subject_revision = producer.source_git_revision.clone();
        let oracle_revision = producer.source_git_revision;
        let source_dirty = producer.source_git_dirty;
        let make_subject = Box::new(move || {
            Ok(Box::new(Self::in_memory_with_source(
                config.clone(),
                subject_revision.clone(),
                source_dirty,
            )?) as Box<dyn DifferentialCampaignEngine>)
        });
        let make_oracle = Box::new(move || {
            Ok(Box::new(TantivyOracle::in_memory_scalar_g1a_with_source(
                &oracle_revision,
                source_dirty,
            )?) as Box<dyn DifferentialCampaignEngine>)
        });
        (make_subject, make_oracle)
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
            let cutoff_certificate = oracle_cutoff_certificate(&observation, case)?;
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
                cutoff_certificate,
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
    let cutoff_certificate = oracle_cutoff_certificate(&observation, case)?;
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
        cutoff_certificate,
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
    #[cfg(feature = "fuzz-harness")]
    use crate::comparator::Divergence;
    #[cfg(any(feature = "perf-harness", feature = "fuzz-harness"))]
    use crate::comparator::DivergenceClass;
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
        schema_version: u32,
    ) -> EnginePairIdentity {
        let receipt = match schema_version {
            BuiltInEngineProfileReceipt::V1_SCHEMA_VERSION
            | BuiltInEngineProfileReceipt::V2_SCHEMA_VERSION
            | BuiltInEngineProfileReceipt::V3_SCHEMA_VERSION => BuiltInEngineProfileReceipt {
                schema_version,
                profile,
                subject_config: QuillConfigReceipt::from_config(config),
            },
            BuiltInEngineProfileReceipt::V4_SCHEMA_VERSION => {
                BuiltInEngineProfileReceipt::new(profile, config)
            }
            _ => panic!("unsupported test profile schema {schema_version}"),
        };
        let semantic_contract = match schema_version {
            BuiltInEngineProfileReceipt::V1_SCHEMA_VERSION => receipt.stored_semantic_contract_v1(),
            // v3 and v4 kept v2's semantic contract byte-identical.
            BuiltInEngineProfileReceipt::V2_SCHEMA_VERSION
            | BuiltInEngineProfileReceipt::V3_SCHEMA_VERSION
            | BuiltInEngineProfileReceipt::V4_SCHEMA_VERSION => {
                receipt.stored_semantic_contract_v2()
            }
            _ => unreachable!("validated above"),
        };
        let (quill_crate_version, lexical_crate_version) = match schema_version {
            BuiltInEngineProfileReceipt::V3_SCHEMA_VERSION => (
                BUILT_IN_PROFILE_V3_QUILL_CRATE_VERSION,
                BUILT_IN_PROFILE_V3_LEXICAL_CRATE_VERSION,
            ),
            BuiltInEngineProfileReceipt::V4_SCHEMA_VERSION => (
                BUILT_IN_PROFILE_V4_QUILL_CRATE_VERSION,
                BUILT_IN_PROFILE_V4_LEXICAL_CRATE_VERSION,
            ),
            _ => (
                BUILT_IN_PROFILE_V1_QUILL_CRATE_VERSION,
                BUILT_IN_PROFILE_V1_LEXICAL_CRATE_VERSION,
            ),
        };
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
                crate_version: quill_crate_version.to_owned(),
                source_revision: producer_revision.clone(),
                source_dirty: false,
                config_hash: subject_config_hash,
            },
            EngineDescriptor {
                family: EngineFamily::Tantivy,
                implementation: "frankensearch-lexical/tantivy-index".to_owned(),
                crate_version: lexical_crate_version.to_owned(),
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

    /// E6.3 executable precondition for the Boolean associativity laws.
    ///
    /// A re-association fixture is only meaningful when the two spellings
    /// differ *solely* in grouping: same operands, same order, same operator,
    /// different parenthesisation. Asserting that here keeps a fixture that
    /// silently stopped exercising re-association -- or that mutated a term
    /// while nobody was looking -- from passing vacuously.
    #[cfg(feature = "perf-harness")]
    fn e63_assert_regrouping_precondition(left: &str, right: &str, operands: &[&str]) {
        assert_ne!(
            left, right,
            "E6.3 associativity fixture must exercise a real re-association"
        );
        let distinct = operands.iter().collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            distinct.len(),
            operands.len(),
            "E6.3 associativity operands must be distinct: {operands:?}"
        );
        let strip = |query: &str| query.replace(['(', ')'], "");
        assert_eq!(
            strip(left),
            strip(right),
            "E6.3 associativity spellings must differ only in grouping"
        );
        for operand in operands {
            assert!(
                left.contains(operand) && right.contains(operand),
                "E6.3 associativity operand {operand} missing from a spelling"
            );
        }
    }

    /// The declared projection for WITHIN-ENGINE Boolean re-association.
    ///
    /// Re-association reorders finite additions, so one engine can return the
    /// same document at the same rank with a score one ULP apart — measured on
    /// Quill at operands `(alpha, gamma, delta)`, `doc-4@3fdc09b7` versus
    /// `doc-4@3fdc09b6`. That is a real effect and the law must not pretend
    /// otherwise.
    ///
    /// It also must not be classified as `DIV-007`. That register entry is
    /// recorded as CROSS-ENGINE (Quill's fused scorer against the pinned
    /// oracle) and scoped to composite shapes, and re-association within a
    /// single engine at three conjunctive leaves is outside it on both axes.
    /// Citing it here would be proof-class inflation, so this law declares a
    /// narrowed projection instead of borrowing a tolerance class:
    ///
    ///   PROJECTED:  ranked document-ID sequence, live doc count, match count,
    ///               and snippets — the full observation minus score bits.
    ///   EXCLUDED:   score bits.
    ///
    /// The exclusion is bounded rather than open: the maximum score-bit
    /// distance is returned so the caller can witness it, and a caller that
    /// finds it growing beyond the measured one ULP is looking at a different
    /// effect than the one this projection was declared for.
    ///
    /// `e6.3-three-term-or-associativity-v1` also uses it in a second role —
    /// to MEASURE the cross-engine cell it excludes, rather than to apply a
    /// law across engines. Measuring an excluded cell keeps the exclusion
    /// earned; it does not make the projection a cross-engine relation.
    #[cfg(feature = "perf-harness")]
    fn e63_reassociation_projection(
        before: &EngineObservation,
        after: &EngineObservation,
    ) -> (bool, u32) {
        let ranked_ids = |observation: &EngineObservation| {
            observation
                .hits
                .iter()
                .map(|hit| hit.doc_id.clone())
                .collect::<Vec<_>>()
        };
        let equivalent = ranked_ids(before) == ranked_ids(after)
            && before.doc_count == after.doc_count
            && before.match_count == after.match_count
            && before.snippets == after.snippets;
        let max_score_bit_distance = before
            .hits
            .iter()
            .zip(after.hits.iter())
            .map(|(before_hit, after_hit)| before_hit.score_bits.abs_diff(after_hit.score_bits))
            .max()
            .unwrap_or(0);
        (equivalent, max_score_bit_distance)
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
    ///
    /// Cross-engine exactness stays asserted here, byte for byte, for every
    /// existing caller. A law that needs to MEASURE a cross-engine divergence
    /// rather than die on it calls [`e63_runs_with_config_and_batch_size`]
    /// directly and states in its own descriptor why.
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
        let runs = e63_runs_with_config_and_batch_size(
            cx,
            documents,
            cases,
            seed,
            generator_id,
            subject_config,
            batch_size,
        )
        .await;
        runs.into_iter()
            .map(|(case_id, comparison)| {
                assert_eq!(
                    comparison.status,
                    ComparisonStatus::Exact,
                    "E6.3 cross-engine case {case_id}: {:?}",
                    comparison.divergences,
                );
                assert_eq!(
                    comparison.rank_class,
                    RankClass::RankExact,
                    "E6.3 cross-engine case {case_id}: {:?}",
                    comparison.divergences,
                );
                (case_id, comparison.subject, comparison.oracle)
            })
            .collect()
    }

    /// The same E6.3 campaign WITHOUT the shared cross-engine exactness
    /// assertion, returning each case's full comparison report.
    ///
    /// The assertion in [`e63_observations_with_config_and_batch_size`] is
    /// strictly stronger than the divergence envelope this project has already
    /// reviewed, so any law whose fixture happens to reach a legitimate
    /// cross-engine difference dies inside the shared harness before its own
    /// equivalence relation is ever evaluated. That is the correct default —
    /// twelve laws depend on it and it is unchanged — but it makes one class
    /// of finding impossible to record: a divergence the law exists to
    /// measure.
    ///
    /// This seam is per-case opt-in, which is what the DIV-007 ruling
    /// prescribes ("the comparator's default config REMAINS zero-tolerance;
    /// campaign lanes ... opt in"). A caller taking it MUST state the
    /// exclusion in its registry descriptor; silently routing a law here to
    /// dodge a red comparison would be gate self-weakening.
    #[cfg(feature = "perf-harness")]
    async fn e63_runs_with_config_and_batch_size(
        cx: &Cx,
        documents: &[frankensearch_core::IndexableDocument],
        cases: &[(&str, &str)],
        seed: u64,
        generator_id: &str,
        subject_config: QuillConfig,
        batch_size: usize,
    ) -> Vec<(String, ComparisonReport)> {
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
            observations.push((case_id.to_owned(), run.comparison));
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
                    cutoff_certificate: None,
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
                input: E55QueryInput::Preparsed(Query::boost(
                    Query::range(
                        E55_I64_FIELD,
                        Bound::Included(QueryValue::I64(-7)),
                        Bound::Excluded(QueryValue::I64(8)),
                    ),
                    2.5,
                )),
            },
            E55QueryCase {
                id: "range-str",
                input: E55QueryInput::Preparsed(Query::range(
                    E55_TAG_FIELD,
                    Bound::Included(QueryValue::Str("blue".to_owned())),
                    Bound::Included(QueryValue::Str("green".to_owned())),
                )),
            },
            E55QueryCase {
                id: "range-i64",
                input: E55QueryInput::Preparsed(Query::range(
                    E55_I64_FIELD,
                    Bound::Included(QueryValue::I64(-7)),
                    Bound::Excluded(QueryValue::I64(8)),
                )),
            },
            E55QueryCase {
                id: "range-u64",
                input: E55QueryInput::Preparsed(Query::range(
                    E55_U64_FIELD,
                    Bound::Included(QueryValue::U64(2)),
                    Bound::Included(QueryValue::U64(8)),
                )),
            },
            E55QueryCase {
                id: "set-str",
                input: E55QueryInput::Preparsed(Query::set(
                    E55_TAG_FIELD,
                    vec![
                        QueryValue::Str("blue".to_owned()),
                        QueryValue::Str("red".to_owned()),
                    ],
                )),
            },
            E55QueryCase {
                id: "set-i64",
                input: E55QueryInput::Preparsed(Query::set(
                    E55_I64_FIELD,
                    vec![QueryValue::I64(-7), QueryValue::I64(9)],
                )),
            },
            E55QueryCase {
                id: "set-u64",
                input: E55QueryInput::Preparsed(Query::set(
                    E55_U64_FIELD,
                    vec![QueryValue::U64(2), QueryValue::U64(13)],
                )),
            },
            E55QueryCase {
                id: "glob",
                input: E55QueryInput::Preparsed(Query::glob(
                    vec![E55_CONTENT_FIELD, E55_TITLE_FIELD],
                    "*lpha*".to_owned(),
                )),
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
        let snapshot = index
            .search_snapshot()
            .expect("E5.5 published snapshot is authoritative");
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
            None,
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
        let snapshot = index
            .search_snapshot()
            .expect("E5.5 published snapshot is authoritative");
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
                cutoff_certificate: None,
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
        let generation = index
            .search_snapshot()
            .expect("historical E5.5 snapshot is authoritative")
            .keeper_generation();
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
                .expect("historical E5.5 snapshot is authoritative")
                .materialize_document_id(historical_docid)
                .as_deref(),
            Some(E55_HISTORICAL_ID),
            "the sealed upsert source remains live until its replacement is staged"
        );
        let baseline_history_segments = index
            .snapshot()
            .expect("historical E5.5 snapshot is authoritative")
            .segments()
            .len();
        (index, baseline_history_segments, historical_docid)
    }

    fn e55_tombstone_sealed_upsert_source(index: &QuillIndex, historical_docid: u32) -> QuillIndex {
        let committed = index
            .snapshot()
            .expect("sealed upsert source snapshot is authoritative")
            .clone();
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
            .expect("E5.5 live corpus snapshot is authoritative")
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
        let generation = index
            .search_snapshot()
            .expect("E5.5 published snapshot is authoritative")
            .keeper_generation();
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
                .expect("E5.5 published snapshot is authoritative")
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
                .expect("E5.5 published snapshot is authoritative")
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
        let snapshot = index
            .search_snapshot()
            .expect("E5.5 published snapshot is authoritative");
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
        let snapshot = index
            .search_snapshot()
            .expect("E5.5 published snapshot is authoritative");
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
        let snapshot = index
            .search_snapshot()
            .expect("E5.5 published snapshot is authoritative");
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
        let replacement_query = Query::term(
            vec![QueryField::new(E55_ID_FIELD, 1.0)],
            E55_HISTORICAL_ID.to_owned(),
        );
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
        let snapshot = index
            .search_snapshot()
            .expect("E5.5 published snapshot is authoritative");
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
        let snapshot = index
            .search_snapshot()
            .expect("E4.10 published snapshot is authoritative");
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
        let generation = index
            .search_snapshot()
            .expect("strict E4.10 snapshot is authoritative")
            .keeper_generation();
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
        let successor_generation = index
            .search_snapshot()
            .expect("sealed-upsert successor snapshot is authoritative")
            .keeper_generation();
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
            .expect("mixed-residency snapshot is authoritative")
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
    fn quill_native_observation_carries_a_same_snapshot_certificate() {
        use crate::cutoff_certificate::{BoundaryNotApplicableV1, BoundaryWitnessV1};

        let mut subject =
            QuillSubject::in_memory_with_source(QuillConfig::default(), "a".repeat(40), false)
                .expect("live Quill subject");
        let documents = vec![
            frankensearch_core::IndexableDocument::new("a", "shared token"),
            frankensearch_core::IndexableDocument::new("b", "shared token"),
            frankensearch_core::IndexableDocument::new("c", "shared token"),
        ];
        let mut case = DifferentialCase::new("quill-certificate", "shared", 2);
        case.tie_expansion_limit = 8;
        case.snippet_max_chars = None;
        case.count_requested = false;
        let mut cut_case = case.clone();
        cut_case.fixture_id = "quill-certificate-cut".to_owned();
        cut_case.tie_expansion_limit = 0;
        let mut counted_case = case.clone();
        counted_case.fixture_id = "quill-certificate-counted".to_owned();
        counted_case.count_requested = true;

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            subject
                .claim_fresh_campaign()
                .expect("claim certificate campaign");
            subject
                .index_mut()
                .expect("open certificate campaign")
                .index_documents(&cx, &documents)
                .await
                .expect("index subject corpus");
            subject
                .index_mut()
                .expect("open certificate campaign")
                .commit(&cx)
                .await
                .expect("commit subject corpus");
            subject
                .mark_committed()
                .expect("publish certificate campaign");

            let observation = subject.observe(&cx, &case).await.expect("observe");
            let certificate = observation
                .cutoff_certificate
                .as_ref()
                .expect("pinned bundle prefix reaches the exact total");
            assert_eq!(certificate.exact_total, 3);
            assert_eq!((certificate.page.start, certificate.page.end), (0, 2));
            assert_eq!(certificate.expanded.len(), 3);
            assert!(certificate.is_exhausted());
            assert_eq!(
                certificate.leading_boundary,
                BoundaryWitnessV1::NotApplicable {
                    reason: BoundaryNotApplicableV1::AtStart
                }
            );
            assert_ne!(certificate.provenance.snapshot_sha256, [0; 32]);
            assert_ne!(certificate.provenance.same_snapshot_authority, [0; 16]);
            // The count-free case still certifies: exact_total comes from
            // the separate count invocation on the same snapshot, never from
            // the ranked page.
            assert_eq!(observation.match_count, CountState::NotRequested);

            // A fetch that cuts the trailing group: no claim.
            let cut = subject.observe(&cx, &cut_case).await.expect("observe cut");
            assert!(cut.cutoff_certificate.is_none());
            assert!(!cut.cutoff_tie_complete);

            // Requesting the count changes nothing about the certificate.
            let counted = subject
                .observe(&cx, &counted_case)
                .await
                .expect("observe counted");
            assert_eq!(counted.match_count, CountState::Value(3));
            let counted_certificate = counted
                .cutoff_certificate
                .as_ref()
                .expect("counted case certifiable");
            assert_eq!(counted_certificate.expanded, certificate.expanded);
            assert_eq!(
                counted_certificate.provenance.snapshot_sha256,
                certificate.provenance.snapshot_sha256,
                "same publication, same physical digest"
            );
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
    fn built_in_profile_v4_is_current_while_v1_v2_and_v3_remain_archive_only() {
        for profile in [
            BuiltInEngineProfile::ScalarShipping,
            BuiltInEngineProfile::ScalarG1a,
            BuiltInEngineProfile::Cass,
        ] {
            for archived_schema in [
                BuiltInEngineProfileReceipt::V1_SCHEMA_VERSION,
                BuiltInEngineProfileReceipt::V2_SCHEMA_VERSION,
                BuiltInEngineProfileReceipt::V3_SCHEMA_VERSION,
            ] {
                let archived =
                    stored_profile_pair(profile, &QuillConfig::default(), archived_schema);
                archived
                    .validate_stored_contract()
                    .expect("frozen receipt remains archive-valid");
                assert!(
                    archived.validate_builtin_contract().is_err(),
                    "schema v{archived_schema} cannot create a run under the gh-39 facade \
                     dependency contract (quill 0.2.2, lexical 0.2.3)"
                );
            }

            let current = stored_profile_pair(
                profile,
                &QuillConfig::default(),
                BuiltInEngineProfileReceipt::V4_SCHEMA_VERSION,
            );
            current
                .validate_stored_contract()
                .expect("v4 receipt remains independently replay-valid");
            current
                .validate_builtin_contract()
                .expect("v4 receipt must match the current adapters");
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
        let baseline = stored_profile_pair(
            BuiltInEngineProfile::ScalarG1a,
            &QuillConfig::default(),
            BuiltInEngineProfileReceipt::V1_SCHEMA_VERSION,
        );
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
            pair.subject.crate_version = "0.0.0-mutated".to_owned();
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
        assert!(
            pair.validate_builtin_contract().is_err(),
            "literal v1 bytes cannot authorize a new Tantivy 0.27 execution"
        );
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
            }]
            .into(),
            total_count: Some(1),
            doc_count: 1,
            diagnostics: Vec::new(),
        };
        let observed = QuillSearchResult {
            total_count: None,
            ..evidence.clone()
        };
        let expected_reason = "Quill observed and expanded collector pages differ";

        let with_mutated_hit = |mutate: fn(&mut frankensearch_quill::QuillHit)| {
            let mut mismatch = observed.clone();
            let mut hits = mismatch.hits.to_vec();
            mutate(&mut hits[0]);
            mismatch.hits = hits.into();
            mismatch
        };
        let wrong_external_id = with_mutated_hit(|hit| hit.document_id = "other".to_owned());
        let wrong_native_tie_key = with_mutated_hit(|hit| hit.global_docid = 8);
        let wrong_score_bits =
            with_mutated_hit(|hit| hit.score = f32::from_bits(3.5_f32.to_bits() + 1));

        for mismatch in [wrong_external_id, wrong_native_tie_key, wrong_score_bits] {
            assert!(matches!(
                quill_observation_from_results(&mismatch, &evidence, 1, 0, false, None),
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
            }]
            .into(),
            total_count: None,
            doc_count: 120,
            diagnostics: Vec::new(),
        };
        let count_evidence = QuillSearchResult {
            hits: Arc::from([]),
            total_count: Some(110),
            doc_count: 120,
            diagnostics: Vec::new(),
        };

        let counted = quill_native_observation_from_results(
            &ranked,
            &ranked,
            &count_evidence,
            1,
            0,
            true,
            None,
        )
        .expect("native ranking plus independent count evidence");
        assert_eq!(counted.hits[0].score_bits, native_score.to_bits());
        assert_eq!(counted.match_count, CountState::Value(110));

        let count_free = quill_native_observation_from_results(
            &ranked,
            &ranked,
            &count_evidence,
            1,
            0,
            false,
            None,
        )
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
            }]
            .into(),
            total_count: None,
            doc_count: 3,
            diagnostics: Vec::new(),
        };
        let count_evidence = QuillSearchResult {
            hits: Arc::from([]),
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
                None,
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
                None,
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
                None,
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
                None,
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
                None,
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
            cutoff_certificate: None,
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

        // bd-pjvl1: a present certificate must describe exactly this case.
        {
            use crate::cutoff_certificate::{CertificateProvenanceV1, CutoffCertificateV1};
            let provenance = CertificateProvenanceV1 {
                snapshot_sha256: [0x11; 32],
                arm_sha256: [0x22; 32],
                ranked_observation_sha256: [0x33; 32],
                expanded_observation_sha256: [0x44; 32],
                same_snapshot_authority: [0x55; 16],
            };
            let prefix = [4.0_f32.to_bits(), 3.0_f32.to_bits(), 1.0_f32.to_bits()];
            let certificate = CutoffCertificateV1::from_native_prefix(3, 0, 2, &prefix, provenance)
                .expect("certifiable");
            let hit = |doc_id: &str, score: f32, doc| RankedHit {
                doc_id: doc_id.to_owned(),
                score_bits: score.to_bits(),
                native_tie_key: NativeTieKey::QuillDocId { doc_id: doc },
            };
            let mut certified_case = DifferentialCase::new("certified", "query", 2);
            certified_case.count_requested = false;
            let certified = EngineObservation {
                cutoff_certificate: Some(certificate.clone()),
                hits: vec![hit("a", 4.0, 1), hit("b", 3.0, 2)],
                cutoff_tie_group: Vec::new(),
                cutoff_tie_complete: true,
                offset_tie_group: Vec::new(),
                offset_tie_complete: false,
                snippets: BTreeMap::new(),
                match_count: CountState::NotRequested,
                doc_count: 3,
                ast_differences: Vec::new(),
            };
            certified_case
                .validate_observation_shape("subject", &certified)
                .expect("a certificate for exactly this page is admissible");

            // Requested page differs from the certified one.
            let mut other_case = certified_case.clone();
            other_case.limit = 3;
            assert!(matches!(
                other_case.validate_observation_shape("subject", &certified),
                Err(GauntletError::InvalidObservation { reason })
                    if reason.contains("does not describe the requested page")
            ));
            // A tampered certificate is refused by its own validator.
            let mut tampered = certified.clone();
            tampered
                .cutoff_certificate
                .as_mut()
                .expect("certificate")
                .expanded_start = 1;
            assert!(matches!(
                certified_case.validate_observation_shape("subject", &tampered),
                Err(GauntletError::InvalidObservation { reason })
                    if reason.contains("cutoff certificate is invalid")
            ));
            // The reported exact count must be the certified exact total.
            let mut miscounted = certified;
            miscounted.match_count = CountState::Value(4);
            let mut counted_case = certified_case;
            counted_case.count_requested = true;
            assert!(matches!(
                counted_case.validate_observation_shape("subject", &miscounted),
                Err(GauntletError::InvalidObservation { reason })
                    if reason.contains("disagrees with the exact count")
                        || reason.contains("inconsistent with its exact count")
            ));
        }

        let quill_hit = RankedHit {
            doc_id: "one".to_owned(),
            score_bits: 1.0_f32.to_bits(),
            native_tie_key: NativeTieKey::QuillDocId { doc_id: 1 },
        };
        let subject_overfilled = EngineObservation {
            cutoff_certificate: None,
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
            cutoff_certificate: None,
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
            cutoff_certificate: None,
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

            // bd-pjvl1: the same observations now carry a same-searcher
            // certificate derived from the native expanded prefix.
            use crate::cutoff_certificate::{BoundaryNotApplicableV1, BoundaryWitnessV1};
            let certificate = observation
                .cutoff_certificate
                .as_ref()
                .expect("prefix reaches the exact total: certifiable");
            assert_eq!(certificate.exact_total, 3);
            assert_eq!((certificate.page.start, certificate.page.end), (0, 2));
            assert_eq!(certificate.expanded_start, 0);
            assert_eq!(certificate.expanded.len(), 3, "the whole tie group");
            assert!(certificate.is_exhausted());
            assert_eq!(
                certificate.trailing_boundary,
                BoundaryWitnessV1::NotApplicable {
                    reason: BoundaryNotApplicableV1::Exhausted
                }
            );
            assert_ne!(certificate.provenance.snapshot_sha256, [0; 32]);
            assert_ne!(certificate.provenance.same_snapshot_authority, [0; 16]);
            // A fetch that cuts the group makes NO claim — not "incomplete".
            assert!(exhausted.cutoff_certificate.is_none());
            // Zero limit: certified with no score boundary.
            let zero = zero_limit
                .cutoff_certificate
                .as_ref()
                .expect("zero limit is certifiable");
            assert!(zero.page_is_empty());
            assert_eq!(
                zero.leading_boundary,
                BoundaryWitnessV1::NotApplicable {
                    reason: BoundaryNotApplicableV1::ZeroLimit
                }
            );
            // Offset inside the tie: the leading group is expanded back to 0.
            let inside = paginated
                .cutoff_certificate
                .as_ref()
                .expect("offset page inside the tie is certifiable");
            assert_eq!((inside.page.start, inside.page.end), (1, 2));
            assert_eq!(inside.expanded_start, 0);
            assert!(inside.is_exhausted());
            assert_ne!(
                inside.digest_sha256().expect("digest"),
                certificate.digest_sha256().expect("digest")
            );
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

    /// E6.3 law: three distinct, unboosted positive scalar `AND` operands
    /// re-associate. `(A AND B) AND C` and `A AND (B AND C)` select the same
    /// conjunction, so the ranked document sequence must agree — under the
    /// declared score-insensitive projection of
    /// [`e63_reassociation_projection`], never under a borrowed tolerance
    /// class.
    ///
    /// This is the law the two-term commutativity pair explicitly excluded
    /// ("excludes association ... because those shapes can expose parser or
    /// score-accumulation behavior"). The exclusion was a deferral, not a
    /// proof, and the measurement vindicated the warning: at operands
    /// `(alpha, gamma, delta)` Quill returns the same document at the same
    /// rank with a score one ULP apart, because re-association reorders finite
    /// additions.
    ///
    /// The law originally bound that shift to `DIV-007`
    /// (`ScoreEpsilonReason::SummationAssociation`). That classification is
    /// RETRACTED: the register records DIV-007 as cross-engine and scoped to
    /// composite shapes, while this shift is within one engine at three
    /// conjunctive leaves — outside the reviewed envelope on both axes.
    /// Widening an owner-ruled tolerance class to cover a shape it was not
    /// reviewed for is proof-class inflation, so the law declares its own
    /// narrowed projection and reports the excluded distance instead. The
    /// measurement itself stands and is routed to the Divergence Register
    /// (`bd-quill-e6-gauntlet-scale-rm3q.8.1`) beside the cross-engine
    /// three-clause OR case.
    ///
    /// Preconditions are executable, not decorative: the operands must be
    /// three distinct analyzed terms and both spellings must differ only in
    /// grouping, so a fixture that accidentally stopped exercising
    /// re-association fails loudly instead of passing vacuously.
    ///
    /// The intentionally invalid control re-groups across MIXED operators
    /// (`(A AND B) OR C` versus `A AND (B OR C)`), which is genuinely not an
    /// associativity transform and must be rejected.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_three_term_and_associates_but_mixed_operator_regrouping_is_not_equivalent() {
        use frankensearch_core::IndexableDocument;

        const SEED: u64 = 0xe63_a55d_c0aa_5eed;
        let documents = vec![
            IndexableDocument::new("doc-1", "alpha beta beta").with_title("guide"),
            IndexableDocument::new("doc-2", "alpha gamma").with_title("alpha overview"),
            IndexableDocument::new("doc-3", "beta gamma gamma gamma").with_title("alpha"),
            IndexableDocument::new("doc-4", "alpha beta gamma delta").with_title("reference"),
            IndexableDocument::new("doc-5", "delta epsilon").with_title("quiet"),
        ];

        let left_grouped = "(alpha AND beta) AND gamma";
        let right_grouped = "alpha AND (beta AND gamma)";
        e63_assert_regrouping_precondition(
            left_grouped,
            right_grouped,
            &["alpha", "beta", "gamma"],
        );

        let canonical = [("three-term-and-assoc", left_grouped)];
        let regrouped = [("three-term-and-assoc", right_grouped)];
        let operator_mutated = [("three-term-and-assoc", "alpha AND (beta OR gamma)")];

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let baseline = e63_observations(
                &cx,
                &documents,
                &canonical,
                SEED,
                "e6.3-three-term-and-associativity-v1",
            )
            .await;
            let transformed = e63_observations(
                &cx,
                &documents,
                &regrouped,
                SEED,
                "e6.3-three-term-and-associativity-v1",
            )
            .await;
            let invalid = e63_observations(
                &cx,
                &documents,
                &operator_mutated,
                SEED,
                "e6.3-three-term-and-associativity-v1",
            )
            .await;

            let baseline_case = baseline.first().expect("E6.3 baseline AND-assoc fixture");
            let regrouped_case = transformed
                .first()
                .expect("E6.3 regrouped AND-assoc fixture");
            assert_eq!(
                baseline_case.0, regrouped_case.0,
                "E6.3 three-term AND associativity case identity drifted"
            );
            for (engine, before, after) in [
                ("Quill", &baseline_case.1, &regrouped_case.1),
                ("Tantivy", &baseline_case.2, &regrouped_case.2),
            ] {
                let (equivalent, score_bit_distance) = e63_reassociation_projection(before, after);
                assert!(
                    equivalent,
                    "E6.3 {engine} three-term AND association changed the projected observation"
                );
                assert!(
                    score_bit_distance <= 1,
                    "E6.3 {engine} excluded a {score_bit_distance}-ULP score shift under AND \
                     association; the projection was declared for the measured one-ULP \
                     re-association effect, so re-measure before widening it"
                );
            }

            let invalid_case = invalid
                .first()
                .expect("E6.3 invalid mixed-operator AND-assoc fixture");
            // The narrowed projection must still REJECT the planted invalid.
            // Excluding scores is only defensible if the remaining projection
            // is what carries the law, so prove it rejects a real
            // non-associativity rather than trusting the zero-tolerance
            // comparison below to do all the work.
            for (engine, before, after) in [
                ("Quill", &baseline_case.1, &invalid_case.1),
                ("Tantivy", &baseline_case.2, &invalid_case.2),
            ] {
                let (equivalent, _) = e63_reassociation_projection(before, after);
                assert!(
                    !equivalent,
                    "E6.3 {engine} score-insensitive projection accepted a mixed-operator \
                     regrouping, so it is too weak to carry this law"
                );
            }
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
                    panic!("E6.3 {engine} invalid mixed-operator comparison failed: {error}")
                });
                assert_eq!(
                    comparison.status,
                    ComparisonStatus::Failed,
                    "E6.3 {engine} incorrectly accepted a mixed-operator regrouping as association",
                );
            }
        });
    }

    /// E6.3 bounded replay campaign for three-term `AND` associativity. The
    /// seed matrix varies which three of the fixture's analyzed terms are
    /// associated, so a law that only held for one operand triple cannot pass.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_three_term_and_associativity_seed_matrix_replays_live_observations() {
        use frankensearch_core::IndexableDocument;

        const SEEDS: [u64; 3] = [
            0xe63_a55d_c0aa_5eed,
            0xe63_a55d_c0aa_5eee,
            0xe63_a55d_c0aa_5eef,
        ];
        let documents = vec![
            IndexableDocument::new("doc-1", "alpha beta beta").with_title("guide"),
            IndexableDocument::new("doc-2", "alpha gamma").with_title("alpha overview"),
            IndexableDocument::new("doc-3", "beta gamma gamma gamma").with_title("alpha"),
            IndexableDocument::new("doc-4", "alpha beta gamma delta").with_title("reference"),
            IndexableDocument::new("doc-5", "delta epsilon").with_title("quiet"),
        ];

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for ((first, second, third), seed) in [
                (("alpha", "beta", "gamma"), SEEDS[0]),
                (("alpha", "gamma", "delta"), SEEDS[1]),
                (("beta", "gamma", "delta"), SEEDS[2]),
            ] {
                let left_grouped = format!("({first} AND {second}) AND {third}");
                let right_grouped = format!("{first} AND ({second} AND {third})");
                e63_assert_regrouping_precondition(
                    &left_grouped,
                    &right_grouped,
                    &[first, second, third],
                );
                assert_eq!(
                    right_grouped,
                    format!("{first} AND ({second} AND {third})"),
                    "E6.3 seed {seed:#x} must replay its association transform byte-identically",
                );

                let canonical_cases = [("three-term-and-assoc", left_grouped.as_str())];
                let regrouped_cases = [("three-term-and-assoc", right_grouped.as_str())];
                let baseline = e63_observations(
                    &cx,
                    &documents,
                    &canonical_cases,
                    seed,
                    "e6.3-three-term-and-associativity-v1",
                )
                .await;
                let transformed = e63_observations(
                    &cx,
                    &documents,
                    &regrouped_cases,
                    seed,
                    "e6.3-three-term-and-associativity-v1",
                )
                .await;
                let baseline_case = baseline
                    .first()
                    .expect("E6.3 seed baseline AND-assoc fixture");
                let regrouped_case = transformed
                    .first()
                    .expect("E6.3 seed regrouped AND-assoc fixture");
                assert_eq!(
                    baseline_case.0, regrouped_case.0,
                    "E6.3 seed {seed:#x} AND-assoc case identity drifted"
                );
                for (engine, before, after) in [
                    ("Quill", &baseline_case.1, &regrouped_case.1),
                    ("Tantivy", &baseline_case.2, &regrouped_case.2),
                ] {
                    let (equivalent, score_bit_distance) =
                        e63_reassociation_projection(before, after);
                    assert!(
                        equivalent,
                        "E6.3 {engine} seed {seed:#x} AND association changed the projected observation"
                    );
                    assert!(
                        score_bit_distance <= 1,
                        "E6.3 {engine} seed {seed:#x} excluded a {score_bit_distance}-ULP score \
                         shift under AND association; re-measure before widening the projection"
                    );
                }
            }
        });
    }

    /// E6.3 law `e6.3-three-term-or-associativity-v1`: three distinct
    /// unboosted optional scalar `OR` operands re-associate WITHIN each
    /// engine, under the same score-insensitive projection the `AND` law
    /// declares. Its cross-engine scope is deliberately excluded, and this
    /// test measures the reason rather than asserting it.
    ///
    /// WHY THE LAW IS NOT CROSS-ENGINE. The BASELINE spelling
    /// `(alpha OR gamma) OR delta` — no transform applied — already
    /// diverges between Quill and the pinned oracle: same document, same rank,
    /// `doc-4@3fdc09b7` versus `3fdc09b6`. The divergence is a property of
    /// that operand triple, not of re-association, and `(alpha OR beta) OR
    /// gamma` compares cleanly.
    ///
    /// It is now DIV-008 in the Divergence Register — the register's first
    /// machine-witnessed entry, ingested from the artifact that observed it,
    /// disposition **blocking** on bead `bd-gx7n4`, with its own executable
    /// regression at
    /// `runner::tests::three_clause_or_diverges_at_one_ulp_without_the_div007_envelope`.
    /// That disposition is exactly why this law's cross-engine cell is
    /// omitted rather than claimed: a raw `RankMismatch` is never accepted,
    /// so no law may assert cross-engine equivalence over it while the
    /// envelope's scope is undecided. DIV-008 also records the same
    /// boundary problem from the other side — the DIV-007 mechanism observed
    /// OUTSIDE its documented qualifiers, on a shape the entry says should be
    /// bit-exact.
    ///
    /// So the law declares `[Quill, Tantivy]` and the cross-engine cell is
    /// omitted rather than claimed. The exclusion is EARNED, not asserted:
    /// this test measures the cross-engine comparison through the opt-in seam
    /// and requires the divergence to still be there and still be one ULP. If
    /// it ever becomes exact, this test fails and the scope must be widened;
    /// if it ever grows, this test fails and it is a different finding.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_three_term_or_associates_within_each_engine_while_cross_engine_stays_registered() {
        use frankensearch_core::IndexableDocument;

        const SEEDS: [u64; 3] = [
            0xe63_055d_c0aa_5eed,
            0xe63_055d_c0aa_5eee,
            0xe63_055d_c0aa_5eef,
        ];
        let documents = vec![
            IndexableDocument::new("doc-1", "alpha beta beta").with_title("guide"),
            IndexableDocument::new("doc-2", "alpha gamma").with_title("alpha overview"),
            IndexableDocument::new("doc-3", "beta gamma gamma gamma").with_title("alpha"),
            IndexableDocument::new("doc-4", "alpha beta gamma delta").with_title("reference"),
            IndexableDocument::new("doc-5", "delta epsilon").with_title("quiet"),
        ];

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let mut cross_engine_divergences = 0_u32;
            for ((first, second, third), seed) in [
                (("alpha", "beta", "gamma"), SEEDS[0]),
                (("alpha", "gamma", "delta"), SEEDS[1]),
                (("beta", "gamma", "delta"), SEEDS[2]),
            ] {
                let left_grouped = format!("({first} OR {second}) OR {third}");
                let right_grouped = format!("{first} OR ({second} OR {third})");
                e63_assert_regrouping_precondition(
                    &left_grouped,
                    &right_grouped,
                    &[first, second, third],
                );

                let baseline = e63_or_associativity_runs(
                    &cx,
                    &documents,
                    &[("three-term-or-assoc", left_grouped.as_str())],
                    seed,
                )
                .await;
                let regrouped = e63_or_associativity_runs(
                    &cx,
                    &documents,
                    &[("three-term-or-assoc", right_grouped.as_str())],
                    seed,
                )
                .await;
                let baseline_case = baseline.first().expect("E6.3 baseline OR-assoc fixture");
                let regrouped_case = regrouped.first().expect("E6.3 regrouped OR-assoc fixture");
                assert_eq!(
                    baseline_case.0, regrouped_case.0,
                    "E6.3 seed {seed:#x} OR-assoc case identity drifted"
                );

                // THE LAW, per engine, under the declared projection.
                for (engine, before, after) in [
                    ("Quill", &baseline_case.1.subject, &regrouped_case.1.subject),
                    ("Tantivy", &baseline_case.1.oracle, &regrouped_case.1.oracle),
                ] {
                    let (equivalent, score_bit_distance) =
                        e63_reassociation_projection(before, after);
                    assert!(
                        equivalent,
                        "E6.3 {engine} seed {seed:#x} OR association changed the projected observation"
                    );
                    assert!(
                        score_bit_distance <= 1,
                        "E6.3 {engine} seed {seed:#x} excluded a {score_bit_distance}-ULP score \
                         shift under OR association; re-measure before widening the projection"
                    );
                }

                // THE EXCLUSION, measured on the UN-transformed spelling so it
                // cannot be blamed on re-association.
                let (cross_equivalent, cross_distance) =
                    e63_reassociation_projection(&baseline_case.1.oracle, &baseline_case.1.subject);
                assert!(
                    cross_equivalent,
                    "E6.3 seed {seed:#x} cross-engine OR baseline diverged in RANKED DOCUMENTS, \
                     not just scores; that is outside the registered finding and must be \
                     investigated rather than excluded"
                );
                assert!(
                    cross_distance <= 1,
                    "E6.3 seed {seed:#x} cross-engine OR baseline score distance grew to \
                     {cross_distance} ULP; the registered rm3q.8.1 finding is one ULP"
                );
                if cross_distance == 1 {
                    cross_engine_divergences += 1;
                }
            }
            assert!(
                cross_engine_divergences >= 1,
                "E6.3 no cross-engine OR divergence reproduced; the registered rm3q.8.1 finding \
                 has gone away, so the cross-engine scope must be re-analysed rather than left \
                 excluded"
            );
        });
    }

    /// E6.3 planted invalid for OR associativity: re-grouping across MIXED
    /// operators is not an associativity transform, and the declared
    /// projection must reject it. Without this the law's projection could be
    /// weakened to something that accepts anything and still look green.
    ///
    /// THE OPERAND TRIPLE IS PART OF THE CONTROL, and the first one tried was
    /// wrong. With `(alpha, beta, gamma)`, `(alpha OR beta) OR gamma` and
    /// `alpha OR (beta AND gamma)` select the SAME four documents on this
    /// fixture, so the projection accepted the mutation and the control failed
    /// — correctly. A negative fixture that the transform cannot distinguish
    /// proves nothing about the projection.
    ///
    /// `(alpha, beta, delta)` distinguishes it: doc-5 ("delta epsilon")
    /// matches the baseline through the third operand alone and cannot match
    /// `beta AND delta`, so the matched set genuinely changes. Choosing
    /// operands so the NEGATIVE can fail is the opposite of fixture-fitting;
    /// fitting would be choosing operands so the positive law passes, and the
    /// positive law's seed matrix above still runs all three triples.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_three_term_or_mixed_operator_regrouping_is_rejected_by_the_projection() {
        use frankensearch_core::IndexableDocument;

        const SEED: u64 = 0xe63_055d_c0aa_5eed;
        let documents = vec![
            IndexableDocument::new("doc-1", "alpha beta beta").with_title("guide"),
            IndexableDocument::new("doc-2", "alpha gamma").with_title("alpha overview"),
            IndexableDocument::new("doc-3", "beta gamma gamma gamma").with_title("alpha"),
            IndexableDocument::new("doc-4", "alpha beta gamma delta").with_title("reference"),
            IndexableDocument::new("doc-5", "delta epsilon").with_title("quiet"),
        ];

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let baseline = e63_or_associativity_runs(
                &cx,
                &documents,
                &[("three-term-or-assoc", "(alpha OR beta) OR delta")],
                SEED,
            )
            .await;
            let mutated = e63_or_associativity_runs(
                &cx,
                &documents,
                &[("three-term-or-assoc", "alpha OR (beta AND delta)")],
                SEED,
            )
            .await;
            let baseline_case = baseline.first().expect("E6.3 baseline OR-assoc fixture");
            let mutated_case = mutated.first().expect("E6.3 mutated OR-assoc fixture");
            for (engine, before, after) in [
                ("Quill", &baseline_case.1.subject, &mutated_case.1.subject),
                ("Tantivy", &baseline_case.1.oracle, &mutated_case.1.oracle),
            ] {
                let (equivalent, _) = e63_reassociation_projection(before, after);
                assert!(
                    !equivalent,
                    "E6.3 {engine} projection accepted a mixed-operator regrouping as OR \
                     association, so it is too weak to carry this law"
                );
            }
        });
    }

    /// The OR-associativity law's opt-in into the non-asserting seam, in ONE
    /// place so the exclusion is auditable rather than scattered.
    #[cfg(feature = "perf-harness")]
    async fn e63_or_associativity_runs(
        cx: &Cx,
        documents: &[frankensearch_core::IndexableDocument],
        cases: &[(&str, &str)],
        seed: u64,
    ) -> Vec<(String, ComparisonReport)> {
        e63_runs_with_config_and_batch_size(
            cx,
            documents,
            cases,
            seed,
            "e6.3-three-term-or-associativity-v1",
            e55_config(),
            documents.len(),
        )
        .await
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

    /// E6.3 `e6.3-duplicate-then-delete-v1`: the law's precondition now HOLDS,
    /// and this test is the measurement that discharges its old skip reason.
    ///
    /// This test previously measured the opposite and earned the law's
    /// `SkipWithReason(RejectedIngestPublishesPartialBatch)`: the serial ingest
    /// route detected a duplicate only after accumulating the batch's earlier
    /// rows, so the rejection left them staged, `delete_document` was refused
    /// while they were staged, and the next `commit()` PUBLISHED them — making
    /// the rejected ID live rather than never-added, so the two lifecycles
    /// diverged at the delete itself.
    ///
    /// `bd-quill-rejected-ingest-publishes-partial-batch-aihri` fixed that:
    /// admission now runs over the whole batch before any of it is
    /// accumulated, on every route, so a rejected batch stages nothing. Every
    /// step of the law's sentence is therefore measurable, and measured here:
    ///
    ///   1. the rejection leaves `has_uncommitted_changes()` FALSE;
    ///   2. `delete_document` is accepted, not refused, and reports `false`;
    ///   3. the next `commit()` publishes nothing — `doc_count` stays 0;
    ///   4. so the rejected-then-delete lifecycle AGREES with the never-added
    ///      control, which is exactly the law's equivalence relation.
    ///
    /// The invalid fixture at the end is the planted negative that keeps this
    /// honest: a UNIQUELY ADMITTED and committed ID reports `true` from the
    /// same delete, so the agreement above is a property of rejection and not
    /// of the assertion being trivially satisfiable.
    ///
    /// THE REGISTRY IS DELIBERATELY NOT FLIPPED HERE, and the assertion at the
    /// end still pins the skip. Two reasons, both of them somebody else's call:
    /// this is the registry's LAST remaining `SkipWithReason`, so the
    /// "a skip is never a pass" plant in
    /// `runner::tests::e63_metamorphic_accounting_excludes_skips_from_passes`
    /// would lose its subject and stop testing anything; and flipping a law to
    /// `Applies` is a coverage claim owned by
    /// `bd-quill-e6-gauntlet-scale-rm3q.3`. The stale skip is tracked, not
    /// forgotten — see the follow-up filed on that bead.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_duplicate_then_delete_precondition_holds_now_that_rejection_stages_nothing() {
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
                    .expect_err("E6.3 duplicate batch must be rejected");
                assert!(matches!(
                    duplicate_error,
                    QuillIndexError::InvalidState { ref detail }
                        if detail.contains("duplicate live document id")
                ));
                // 1. The rejected batch staged nothing.
                assert!(
                    !rejected
                        .index()
                        .expect("E6.3 read rejected duplicate lifecycle campaign")
                        .has_uncommitted_changes(),
                    "E6.3 seed {seed:#x} a rejected batch must leave no staged row behind"
                );

                // 2. So the transform's own operation is now executable, and
                //    reports the ID as absent rather than being refused.
                let rejected_delete = rejected
                    .index_mut()
                    .expect("E6.3 rejected duplicate campaign remains open")
                    .delete_document(&cx, &document_id)
                    .await
                    .expect("E6.3 delete after a rejected duplicate batch must be executable");
                assert!(
                    !rejected_delete,
                    "E6.3 seed {seed:#x} the rejected ID must report absent, not live"
                );

                // 3. And committing after the rejection publishes nothing.
                rejected
                    .index_mut()
                    .expect("E6.3 rejected duplicate campaign remains open")
                    .commit(&cx)
                    .await
                    .expect("E6.3 commit after a rejected duplicate batch");
                assert_eq!(
                    rejected
                        .index()
                        .expect("E6.3 read rejected duplicate lifecycle campaign")
                        .doc_count()
                        .expect("E6.3 rejected campaign count is authoritative"),
                    0,
                    "E6.3 seed {seed:#x} a rejected ingest must not publish part of its batch"
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

                // 4. The two lifecycles AGREE, which is the law's equivalence
                //    relation. They disagreed before the atomic-ingest fix.
                assert_eq!(
                    rejected_delete, never_added_delete,
                    "E6.3 seed {seed:#x} rejected-then-delete must be the same lifecycle as \
                     never-added; if these ever diverge again, the ingest atomicity fix \
                     regressed and the registry skip has to come back"
                );

                // WHY THE PROJECTION MATTERS. Both terminal corpora are empty,
                // so a law projected on the searchable corpus alone compares
                // Exact here and would report this transform as holding. It is
                // the typed delete outcome above that shows it does not. This
                // is the "equal empty terminal results mask an incorrect
                // relation" trap the original law warned about, measured.
                let masked = compare_observations(
                    rejected_observation,
                    never_added_observation,
                    ComparatorConfig::default(),
                )
                .unwrap_or_else(|error| {
                    panic!("E6.3 seed {seed:#x} lifecycle comparison failed: {error}")
                });
                assert_eq!(masked.status, ComparisonStatus::Exact);
                assert_eq!(masked.rank_class, RankClass::RankExact);

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

            // Bind the measurement to the declaration. That change has now
            // happened — the rejected batch leaves nothing staged — so this
            // assertion no longer records a justified skip. It records a STALE
            // one, deliberately, so the descriptor cannot drift out of sight:
            // whoever flips it to `Applies` must also give
            // `e63_metamorphic_accounting_excludes_skips_from_passes` another
            // skip to plant on, since this is the registry's last one.
            let descriptor = crate::runner::MetamorphicLawRegistry::scalar_g1a_v1()
                .laws
                .into_iter()
                .find(|law| law.id == "e6.3-duplicate-then-delete-v1")
                .expect("E6.3 registry declares the duplicate-then-delete law");
            assert_eq!(
                descriptor.applicability,
                crate::runner::MetamorphicLawApplicability::SkipWithReason {
                    reason:
                        crate::runner::MetamorphicSkipReason::RejectedIngestPublishesPartialBatch,
                },
                "this skip is STALE: the measurement above shows its reason is discharged, and \
                 flipping it is tracked on bd-quill-e6-gauntlet-scale-rm3q.3"
            );
        });
    }

    /// Execute one E6.3 replacement route through the public Quill writer
    /// surfaces and return its normal total lexical observation.
    ///
    /// `LexicalWrite::index_documents` is deliberately qualified here. The
    /// scalar inherent method has a different, intentionally strict contract:
    /// it rejects a duplicate live ID. This helper probes the shipping writer
    /// trait's replacement contract instead of accidentally exercising that
    /// scalar rejection path twice.
    #[cfg(feature = "perf-harness")]
    async fn e63_writer_replacement_observation(
        cx: &Cx,
        original: &frankensearch_core::IndexableDocument,
        replacement: &frankensearch_core::IndexableDocument,
        case: &DifferentialCase,
        use_writer_upsert: bool,
    ) -> EngineObservation {
        use frankensearch_core::LexicalWrite;

        let mut subject = qg_position_mode_subject(true);
        subject
            .claim_fresh_campaign()
            .expect("E6.3 claim writer replacement campaign");
        let index = subject
            .index_mut()
            .expect("E6.3 open writer replacement campaign");

        if use_writer_upsert {
            <QuillIndex as LexicalWrite>::index_documents(
                index,
                cx,
                std::slice::from_ref(original),
            )
            .await
            .expect("E6.3 writer upsert admits original document");
            <QuillIndex as LexicalWrite>::commit(index, cx)
                .await
                .expect("E6.3 writer upsert commits original document");
            <QuillIndex as LexicalWrite>::index_documents(
                index,
                cx,
                std::slice::from_ref(replacement),
            )
            .await
            .expect("E6.3 writer upsert replaces live document");
            <QuillIndex as LexicalWrite>::commit(index, cx)
                .await
                .expect("E6.3 writer upsert commits replacement document");
        } else {
            index
                .index_documents(cx, std::slice::from_ref(original))
                .await
                .expect("E6.3 delete/add admits original document");
            index
                .commit(cx)
                .await
                .expect("E6.3 delete/add commits original document");
            assert!(
                index
                    .delete_document(cx, &original.id)
                    .await
                    .expect("E6.3 delete/add deletes original document"),
                "E6.3 delete/add must delete its known live original document"
            );
            index
                .index_documents(cx, std::slice::from_ref(replacement))
                .await
                .expect("E6.3 delete/add admits replacement document");
            index
                .commit(cx)
                .await
                .expect("E6.3 delete/add commits replacement document");
        }

        subject
            .mark_committed()
            .expect("E6.3 publish writer replacement campaign");
        subject
            .observe(cx, case)
            .await
            .expect("E6.3 observe writer replacement campaign")
    }

    /// E6.3 writer lifecycle law: the shipping `LexicalWrite` replacement
    /// operation is observationally equivalent to explicitly deleting the
    /// old live ID and then adding the replacement. The scalar inherent ingest
    /// API is intentionally not used as upsert: it rejects duplicate IDs, and
    /// that distinct contract is covered by the duplicate-rejection law.
    ///
    /// Each seed runs the writer-upsert arm twice to prove replay, compares it
    /// with the independently sequenced delete/add arm, and uses a planted
    /// content mutation as a negative fixture. The mutation changes only the
    /// replacement payload, so a false-green cannot be explained by a changed
    /// ID, schema, query, or lifecycle route.
    #[cfg(feature = "perf-harness")]
    #[test]
    fn e63_upsert_delete_add_seed_matrix_replays_live_writer_observations() {
        use frankensearch_core::IndexableDocument;

        const SEEDS: [u64; 3] = [
            0xe630_0000_0000_0001,
            0xe630_0000_0000_0002,
            0xe630_0000_0000_0003,
        ];

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for seed in SEEDS {
                let document_id = format!("e63-upsert-delete-add-{seed:016x}");
                let original = IndexableDocument::new(
                    document_id.clone(),
                    format!("e63-original-token seed-{seed:016x}"),
                );
                let replacement = IndexableDocument::new(
                    document_id.clone(),
                    format!("e63-replacement-token seed-{seed:016x}"),
                );
                let invalid_replacement = IndexableDocument::new(
                    document_id,
                    format!("e63-invalid-mutation-token seed-{seed:016x}"),
                );
                let mut replacement_case = DifferentialCase::new(
                    format!("e63-upsert-delete-add-{seed:016x}"),
                    "e63-replacement-token",
                    16,
                );
                replacement_case.snippet_max_chars = None;
                replacement_case.tie_expansion_limit = 64;
                replacement_case.metadata.generator_id =
                    Some("e6.3-upsert-versus-delete-add-v1".to_owned());
                replacement_case.metadata.generator_seed = Some(seed);

                let upsert = e63_writer_replacement_observation(
                    &cx,
                    &original,
                    &replacement,
                    &replacement_case,
                    true,
                )
                .await;
                let replayed_upsert = e63_writer_replacement_observation(
                    &cx,
                    &original,
                    &replacement,
                    &replacement_case,
                    true,
                )
                .await;
                let delete_add = e63_writer_replacement_observation(
                    &cx,
                    &original,
                    &replacement,
                    &replacement_case,
                    false,
                )
                .await;
                let replay_comparison = compare_observations(
                    upsert.clone(),
                    replayed_upsert,
                    ComparatorConfig::default(),
                )
                .unwrap_or_else(|error| {
                    panic!("E6.3 seed {seed:#x} writer-upsert replay comparison failed: {error}")
                });
                assert_eq!(
                    replay_comparison.status,
                    ComparisonStatus::Exact,
                    "E6.3 seed {seed:#x} writer upsert must replay exactly"
                );
                let lifecycle_comparison =
                    compare_observations(upsert, delete_add, ComparatorConfig::default())
                        .unwrap_or_else(|error| {
                            panic!(
                                "E6.3 seed {seed:#x} upsert/delete-add comparison failed: {error}"
                            )
                        });
                assert_eq!(
                    lifecycle_comparison.status,
                    ComparisonStatus::Exact,
                    "E6.3 seed {seed:#x} writer upsert and delete/add diverged: {:?}",
                    lifecycle_comparison.divergences
                );

                let invalid = e63_writer_replacement_observation(
                    &cx,
                    &original,
                    &invalid_replacement,
                    &replacement_case,
                    true,
                )
                .await;
                let invalid_comparison = compare_observations(
                    e63_writer_replacement_observation(
                        &cx,
                        &original,
                        &replacement,
                        &replacement_case,
                        true,
                    )
                    .await,
                    invalid,
                    ComparatorConfig::default(),
                )
                .unwrap_or_else(|error| {
                    panic!("E6.3 seed {seed:#x} invalid writer-upsert comparison failed: {error}")
                });
                assert_eq!(
                    invalid_comparison.status,
                    ComparisonStatus::Failed,
                    "E6.3 seed {seed:#x} planted replacement-content mutation must not satisfy the lifecycle law"
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

        oracle_version_contract().expect("current oracle version contract");
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

    #[cfg(feature = "fuzz-harness")]
    /// Synthesizes a structurally valid failed fingerprint for persistence
    /// tests. It deliberately does not claim an engine divergence was
    /// re-executed; that belongs to the fuzz target's live subject/oracle run.
    fn typed_query_test_replay(
        input: &[u8],
        minimized_ast: TypedQueryTree,
    ) -> TypedQueryFuzzReplay {
        let workload =
            materialize_typed_query_fuzz_workload(input).expect("typed-query test workload");
        TypedQueryFuzzReplay {
            schema_version: TYPED_QUERY_FUZZ_REPLAY_SCHEMA_VERSION,
            generator_id: TYPED_QUERY_FUZZ_GENERATOR_ID.to_owned(),
            original_input: workload.original_input.clone(),
            original_seed: workload.seed,
            corpus_spec: workload.corpus_spec.clone(),
            corpus_manifest: workload.corpus_manifest.clone(),
            corpus_manifest_hash: workload.corpus_manifest_hash.clone(),
            vocabulary: workload.vocabulary.clone(),
            minimized_input: minimized_ast.canonical_input(),
            minimized_ast,
            minimized_query: minimized_ast.render(&workload.vocabulary),
            fingerprint: TypedQueryFailureFingerprint {
                status: ComparisonStatus::Failed,
                rank_class: RankClass::RankMismatch,
                first_divergence: Some("/comparison/subject/hits/0".to_owned()),
                divergences: vec![Divergence {
                    class: DivergenceClass::RankMismatch,
                    pointer: "/comparison/subject/hits/0".to_owned(),
                    oracle: "oracle-doc".to_owned(),
                    subject: "quill-doc".to_owned(),
                }],
            },
            divergence_class: DivergenceClass::RankMismatch,
        }
    }

    #[cfg(all(
        feature = "fuzz-harness",
        any(
            target_os = "android",
            target_os = "ios",
            target_os = "linux",
            target_os = "macos",
            target_os = "tvos",
            target_os = "visionos",
            target_os = "watchos"
        )
    ))]
    fn typed_query_replay_test_root(label: &str) -> std::path::PathBuf {
        static DIRECTORY_NONCE: AtomicUsize = AtomicUsize::new(0);

        let nonce = DIRECTORY_NONCE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "frankensearch-typed-query-replay-{label}-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&root).expect("create unique owned replay test directory");
        root
    }

    #[cfg(all(
        feature = "fuzz-harness",
        any(
            target_os = "android",
            target_os = "ios",
            target_os = "linux",
            target_os = "macos",
            target_os = "tvos",
            target_os = "visionos",
            target_os = "watchos"
        )
    ))]
    fn typed_query_write_new_owned_file(path: &std::path::Path, bytes: &[u8]) {
        use std::io::Write as _;

        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("create one new owned hostile replay artifact");
        file.write_all(bytes)
            .expect("write one new owned hostile replay artifact");
        file.sync_all()
            .expect("sync one new owned hostile replay artifact");
    }

    #[cfg(all(
        feature = "fuzz-harness",
        any(
            target_os = "android",
            target_os = "ios",
            target_os = "linux",
            target_os = "macos",
            target_os = "tvos",
            target_os = "visionos",
            target_os = "watchos"
        )
    ))]
    fn typed_query_replay_path(
        root: &std::path::Path,
        replay: &TypedQueryFuzzReplay,
    ) -> std::path::PathBuf {
        root.join(TYPED_QUERY_FUZZ_REPLAY_DIRECTORY)
            .join(typed_query_fuzz_replay_filename(
                &replay
                    .artifact_key()
                    .expect("canonical typed-query replay key"),
            ))
    }

    #[cfg(all(
        feature = "fuzz-harness",
        any(
            target_os = "android",
            target_os = "ios",
            target_os = "linux",
            target_os = "macos",
            target_os = "tvos",
            target_os = "visionos",
            target_os = "watchos"
        )
    ))]
    fn typed_query_rewrite_owned_file_same_length(path: &std::path::Path, bytes: &[u8]) {
        use std::io::Write as _;

        let original_len = std::fs::metadata(path)
            .expect("owned replay artifact exists before hostile rewrite")
            .len();
        assert_eq!(
            u64::try_from(bytes.len()).expect("hostile bytes fit u64"),
            original_len,
            "hostile rewrite must retain the original file length"
        );
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .expect("open one owned replay artifact for hostile same-inode rewrite");
        file.write_all(bytes)
            .expect("rewrite one owned replay artifact at the same length");
        file.sync_all()
            .expect("sync hostile same-inode replay rewrite");
    }

    #[cfg(feature = "fuzz-harness")]
    #[test]
    fn typed_query_seed_is_64_bit_and_pinned() {
        assert_eq!(
            typed_query_fuzz_seed(&[0, 1, 2, 3, 255]),
            0x45b5_a892_b3d3_b5e9,
            "the schema-v3 byte-to-corpus seed must stay a deliberate u64 mapping"
        );
    }

    #[cfg(feature = "fuzz-harness")]
    #[test]
    fn typed_query_replay_reconstructs_and_rejects_hostile_mutations() {
        let replay = typed_query_test_replay(&[22, 1, 7, 99], TypedQueryTree::MixedHitMiss(1, 7));
        let rebuilt = replay
            .replay_workload()
            .expect("the shared replay entrypoint must rebuild the minimized case");
        assert_eq!(rebuilt.ast, TypedQueryTree::MixedHitMiss(1, 7));
        assert_eq!(rebuilt.case.query, replay.minimized_query);
        assert_eq!(rebuilt.corpus_manifest_hash, replay.corpus_manifest_hash);

        let mut seed_tamper = replay.clone();
        seed_tamper.original_seed ^= 1;
        assert!(seed_tamper.replay_workload().is_err());

        let mut corpus_tamper = replay.clone();
        corpus_tamper.corpus_manifest_hash = "0".repeat(64);
        assert!(corpus_tamper.replay_workload().is_err());

        let mut ast_tamper = replay.clone();
        ast_tamper.minimized_ast = TypedQueryTree::OovOnly(7);
        assert!(ast_tamper.replay_workload().is_err());

        let mut query_tamper = replay.clone();
        query_tamper.minimized_query.push('!');
        assert!(query_tamper.replay_workload().is_err());

        let mut fingerprint_tamper = replay.clone();
        fingerprint_tamper.fingerprint.divergences[0].class = DivergenceClass::ScoreEpsilon;
        assert!(fingerprint_tamper.replay_workload().is_err());

        let same_ast_new_corpus =
            typed_query_test_replay(&[22, 1, 7, 98], TypedQueryTree::MixedHitMiss(1, 7));
        assert_ne!(
            replay.artifact_key().expect("first replay key"),
            same_ast_new_corpus
                .artifact_key()
                .expect("second replay key"),
            "same minimized AST from a different raw seed must not collide"
        );

        let mut fingerprint_collision = replay.clone();
        fingerprint_collision.fingerprint.first_divergence =
            Some("/comparison/subject/hits/1".to_owned());
        assert_ne!(
            replay.artifact_key().expect("original replay key"),
            fingerprint_collision
                .artifact_key()
                .expect("different fingerprint replay key"),
            "different failure fingerprints must not collide under one artifact key"
        );
    }

    #[cfg(all(
        feature = "fuzz-harness",
        any(
            target_os = "android",
            target_os = "ios",
            target_os = "linux",
            target_os = "macos",
            target_os = "tvos",
            target_os = "visionos",
            target_os = "watchos"
        )
    ))]
    #[test]
    fn typed_query_public_replay_persistence_authenticates_canonical_filename() {
        let replay = typed_query_test_replay(&[22, 1, 7, 99], TypedQueryTree::MixedHitMiss(1, 7));
        let root = typed_query_replay_test_root("public-persistence");
        let artifact = persist_typed_query_fuzz_replay(&root, &replay)
            .expect("persist with create-new in an owned directory");
        let canonical_bytes = replay.canonical_bytes().expect("canonical replay bytes");
        let artifact_key = replay.artifact_key().expect("canonical artifact key");
        let expected_filename = typed_query_fuzz_replay_filename(&artifact_key);
        let path = typed_query_replay_path(&root, &replay);
        assert_eq!(
            path.file_name().and_then(std::ffi::OsStr::to_str),
            Some(expected_filename.as_str()),
            "the caller-supplied load locator must use the canonical content-addressed name"
        );
        assert_eq!(
            artifact
                .replay_workload()
                .expect("persisted descriptor capability reconstructs the replay")
                .case
                .query,
            replay.minimized_query,
            "the public persist entrypoint must return a consumable descriptor capability"
        );
        assert_eq!(
            load_typed_query_fuzz_replay(&path)
                .expect("load canonical persisted replay")
                .replay_workload()
                .expect("loaded descriptor capability reconstructs the replay")
                .case
                .query,
            replay.minimized_query,
            "the public load entrypoint must return a consumable descriptor capability"
        );
        assert_eq!(
            persist_typed_query_fuzz_replay(&root, &replay)
                .expect("authenticate an already-created canonical replay without overwrite")
                .artifact_key()
                .expect("duplicate descriptor capability has the canonical key"),
            artifact_key,
            "a duplicate persist may only reuse the same authenticated create-new artifact"
        );

        let reject_under_original_key = |label: &str, bytes: Vec<u8>| {
            let hostile_root = typed_query_replay_test_root(label);
            let hostile_directory = hostile_root.join(TYPED_QUERY_FUZZ_REPLAY_DIRECTORY);
            std::fs::create_dir(&hostile_directory)
                .expect("create unique hostile typed-query directory");
            let hostile_path = hostile_directory.join(&expected_filename);
            typed_query_write_new_owned_file(&hostile_path, &bytes);
            assert!(
                matches!(
                    load_typed_query_fuzz_replay(&hostile_path),
                    Err(GauntletError::ManifestMismatch { .. })
                ),
                "{label} must not load under the original content-addressed key"
            );
            assert_eq!(
                std::fs::read(&hostile_path).expect("hostile artifact remains intact"),
                bytes,
                "{label} must not be overwritten during rejection"
            );
        };

        let replacement =
            typed_query_test_replay(&[22, 1, 7, 98], TypedQueryTree::MixedHitMiss(1, 7));
        assert_ne!(
            replacement.artifact_key().expect("replacement key"),
            artifact_key,
            "a different corpus identity must not reuse the original key"
        );
        reject_under_original_key(
            "valid replacement replay",
            replacement.canonical_bytes().expect("replacement bytes"),
        );

        let mut fingerprint_tamper = replay.clone();
        fingerprint_tamper.fingerprint.first_divergence =
            Some("/comparison/subject/hits/1".to_owned());
        assert_ne!(
            fingerprint_tamper
                .artifact_key()
                .expect("tampered fingerprint key"),
            artifact_key,
            "a fingerprint mutation must change the canonical key"
        );
        reject_under_original_key(
            "fingerprint mutation",
            fingerprint_tamper
                .canonical_bytes()
                .expect("fingerprint mutation bytes"),
        );
        let mut noncanonical_bytes = canonical_bytes.clone();
        noncanonical_bytes.push(b'\n');
        reject_under_original_key("noncanonical encoding", noncanonical_bytes);

        let mut seed_tamper = replay.clone();
        seed_tamper.original_seed ^= 1;
        reject_under_original_key(
            "seed mutation",
            serde_json::to_vec(&seed_tamper).expect("seed mutation bytes"),
        );

        let mut corpus_tamper = replay.clone();
        corpus_tamper.corpus_manifest_hash = "0".repeat(64);
        reject_under_original_key(
            "corpus identity mutation",
            serde_json::to_vec(&corpus_tamper).expect("corpus mutation bytes"),
        );

        let mut ast_tamper = replay.clone();
        ast_tamper.minimized_ast = TypedQueryTree::OovOnly(7);
        reject_under_original_key(
            "AST mutation",
            serde_json::to_vec(&ast_tamper).expect("AST mutation bytes"),
        );

        let mut query_tamper = replay.clone();
        query_tamper.minimized_query.push('!');
        reject_under_original_key(
            "query mutation",
            serde_json::to_vec(&query_tamper).expect("query mutation bytes"),
        );

        let wrong_filename_root = typed_query_replay_test_root("wrong-filename");
        let wrong_filename_directory = wrong_filename_root.join(TYPED_QUERY_FUZZ_REPLAY_DIRECTORY);
        std::fs::create_dir(&wrong_filename_directory)
            .expect("create wrong-filename typed-query directory");
        let wrong_filename = wrong_filename_directory.join("not-the-content-key.json");
        typed_query_write_new_owned_file(&wrong_filename, &canonical_bytes);
        assert!(matches!(
            load_typed_query_fuzz_replay(&wrong_filename),
            Err(GauntletError::ManifestMismatch { .. })
        ));

        let unknown_extension_root = typed_query_replay_test_root("unknown-extension");
        let unknown_extension_directory =
            unknown_extension_root.join(TYPED_QUERY_FUZZ_REPLAY_DIRECTORY);
        std::fs::create_dir(&unknown_extension_directory)
            .expect("create unknown-extension typed-query directory");
        let unknown_extension =
            unknown_extension_directory.join("replay-with-unknown-extension.replay");
        typed_query_write_new_owned_file(&unknown_extension, &canonical_bytes);
        assert!(matches!(
            load_typed_query_fuzz_replay(&unknown_extension),
            Err(GauntletError::ManifestMismatch { .. })
        ));

        let collision_root = typed_query_replay_test_root("create-new-collision");
        let collision_path = collision_root
            .join("typed_query_tree")
            .join(&expected_filename);
        std::fs::create_dir_all(
            collision_path
                .parent()
                .expect("collision path has a typed-query directory"),
        )
        .expect("create owned collision directory");
        typed_query_write_new_owned_file(&collision_path, b"unrelated existing replay");
        assert!(matches!(
            persist_typed_query_fuzz_replay(&collision_root, &replay),
            Err(GauntletError::ArtifactCollision { path: reported }) if reported == collision_path
        ));
        assert_eq!(
            std::fs::read(&collision_path).expect("collision artifact remains intact"),
            b"unrelated existing replay"
        );
        assert_eq!(
            load_typed_query_fuzz_replay(&path)
                .expect("original replay remains loadable")
                .replay_workload()
                .expect("original descriptor capability remains consumable")
                .case
                .query,
            replay.minimized_query,
            "hostile cases must not overwrite the original persisted artifact"
        );
    }

    #[cfg(all(
        feature = "fuzz-harness",
        any(
            target_os = "android",
            target_os = "ios",
            target_os = "linux",
            target_os = "macos",
            target_os = "tvos",
            target_os = "visionos",
            target_os = "watchos"
        )
    ))]
    #[test]
    fn typed_query_replay_sidecar_rejects_symlinks_and_nonregular_entries() {
        use std::os::unix::fs::symlink;

        let replay = typed_query_test_replay(&[22, 1, 7, 99], TypedQueryTree::MixedHitMiss(1, 7));
        let canonical_bytes = replay.canonical_bytes().expect("canonical replay bytes");
        let filename = typed_query_fuzz_replay_filename(
            &replay.artifact_key().expect("canonical artifact key"),
        );

        let final_target_root = typed_query_replay_test_root("final-symlink-target");
        let final_target_artifact = persist_typed_query_fuzz_replay(&final_target_root, &replay)
            .expect("persist final-symlink target");
        assert_eq!(
            final_target_artifact
                .replay_workload()
                .expect("final-symlink target capability remains consumable")
                .case
                .query,
            replay.minimized_query
        );
        let final_target = typed_query_replay_path(&final_target_root, &replay);
        let final_symlink_root = typed_query_replay_test_root("final-symlink");
        let final_symlink_directory = final_symlink_root.join(TYPED_QUERY_FUZZ_REPLAY_DIRECTORY);
        std::fs::create_dir(&final_symlink_directory).expect("create final-symlink directory");
        let final_symlink = final_symlink_directory.join(&filename);
        symlink(&final_target, &final_symlink).expect("plant final-component symlink");
        assert!(matches!(
            persist_typed_query_fuzz_replay(&final_symlink_root, &replay),
            Err(GauntletError::UnsafeStorePath { .. })
        ));
        assert!(matches!(
            load_typed_query_fuzz_replay(&final_symlink),
            Err(GauntletError::UnsafeStorePath { .. })
        ));
        assert!(
            std::fs::symlink_metadata(&final_symlink)
                .expect("final symlink remains intact")
                .file_type()
                .is_symlink()
        );
        assert_eq!(
            std::fs::read(&final_target).expect("final target remains intact"),
            canonical_bytes
        );

        let directory_target_root = typed_query_replay_test_root("directory-symlink-target");
        let directory_target_artifact =
            persist_typed_query_fuzz_replay(&directory_target_root, &replay)
                .expect("persist directory-symlink target");
        assert_eq!(
            directory_target_artifact
                .replay_workload()
                .expect("directory-symlink target capability remains consumable")
                .case
                .query,
            replay.minimized_query
        );
        let directory_target = typed_query_replay_path(&directory_target_root, &replay);
        let directory_symlink_root = typed_query_replay_test_root("directory-symlink");
        let directory_symlink = directory_symlink_root.join(TYPED_QUERY_FUZZ_REPLAY_DIRECTORY);
        symlink(
            directory_target
                .parent()
                .expect("directory target has sidecar parent"),
            &directory_symlink,
        )
        .expect("plant typed-query-tree directory symlink");
        let directory_symlink_path = directory_symlink.join(&filename);
        assert!(matches!(
            persist_typed_query_fuzz_replay(&directory_symlink_root, &replay),
            Err(GauntletError::UnsafeStorePath { .. })
        ));
        assert!(matches!(
            load_typed_query_fuzz_replay(&directory_symlink_path),
            Err(GauntletError::UnsafeStorePath { .. })
        ));
        assert!(
            std::fs::symlink_metadata(&directory_symlink)
                .expect("directory symlink remains intact")
                .file_type()
                .is_symlink()
        );
        assert_eq!(
            load_typed_query_fuzz_replay(&directory_target)
                .expect("directory symlink target remains loadable")
                .replay_workload()
                .expect("directory symlink target descriptor remains consumable")
                .case
                .query,
            replay.minimized_query
        );

        let nonregular_root = typed_query_replay_test_root("nonregular-collision");
        let nonregular_directory = nonregular_root.join(TYPED_QUERY_FUZZ_REPLAY_DIRECTORY);
        std::fs::create_dir(&nonregular_directory).expect("create nonregular sidecar directory");
        let nonregular_collision = nonregular_directory.join(&filename);
        std::fs::create_dir(&nonregular_collision).expect("plant nonregular collision directory");
        assert!(matches!(
            persist_typed_query_fuzz_replay(&nonregular_root, &replay),
            Err(GauntletError::UnsafeStorePath { .. })
        ));
        assert!(
            std::fs::metadata(&nonregular_collision)
                .expect("nonregular collision remains intact")
                .is_dir()
        );
    }

    #[cfg(all(
        feature = "fuzz-harness",
        any(
            target_os = "android",
            target_os = "ios",
            target_os = "linux",
            target_os = "macos",
            target_os = "tvos",
            target_os = "visionos",
            target_os = "watchos"
        )
    ))]
    #[test]
    fn typed_query_public_replay_rejects_post_io_final_and_parent_substitution() {
        let replay = typed_query_test_replay(&[22, 1, 7, 99], TypedQueryTree::MixedHitMiss(1, 7));
        let canonical_bytes = replay.canonical_bytes().expect("canonical replay bytes");
        let filename = typed_query_fuzz_replay_filename(
            &replay.artifact_key().expect("canonical artifact key"),
        );

        let create_root = typed_query_replay_test_root("create-final-substitution");
        let create_displaced = create_root
            .join(TYPED_QUERY_FUZZ_REPLAY_DIRECTORY)
            .join(format!("{filename}.displaced-create"));
        let create_bytes = canonical_bytes.clone();
        let create_displaced_for_hook = create_displaced.clone();
        install_typed_query_fuzz_replay_final_binding_hook(move |path| {
            std::fs::rename(path, &create_displaced_for_hook)
                .expect("move post-write owned canonical artifact aside without deletion");
            typed_query_write_new_owned_file(path, &create_bytes);
        });
        let create_path = create_root
            .join(TYPED_QUERY_FUZZ_REPLAY_DIRECTORY)
            .join(&filename);
        assert!(matches!(
            persist_typed_query_fuzz_replay(&create_root, &replay),
            Err(GauntletError::UnsafeStorePath { path }) if path == create_path
        ));
        assert_eq!(
            std::fs::read(&create_displaced).expect("displaced post-write artifact remains intact"),
            canonical_bytes
        );
        assert_eq!(
            std::fs::read(&create_path).expect("substituted canonical artifact remains intact"),
            canonical_bytes
        );

        let load_root = typed_query_replay_test_root("load-final-substitution");
        let load_artifact =
            persist_typed_query_fuzz_replay(&load_root, &replay).expect("persist load source");
        assert_eq!(
            load_artifact
                .replay_workload()
                .expect("load source descriptor capability remains consumable")
                .case
                .query,
            replay.minimized_query
        );
        let load_path = typed_query_replay_path(&load_root, &replay);
        let load_displaced = load_path.with_file_name(format!("{filename}.displaced-load"));
        let load_bytes = canonical_bytes.clone();
        let load_displaced_for_hook = load_displaced.clone();
        install_typed_query_fuzz_replay_final_binding_hook(move |path| {
            std::fs::rename(path, &load_displaced_for_hook)
                .expect("move post-read owned canonical artifact aside without deletion");
            typed_query_write_new_owned_file(path, &load_bytes);
        });
        assert!(matches!(
            load_typed_query_fuzz_replay(&load_path),
            Err(GauntletError::UnsafeStorePath { path }) if path == load_path
        ));
        assert_eq!(
            std::fs::read(&load_displaced).expect("displaced post-read artifact remains intact"),
            canonical_bytes
        );
        assert_eq!(
            std::fs::read(&load_path).expect("substituted load artifact remains intact"),
            canonical_bytes
        );

        let parent_root = typed_query_replay_test_root("load-parent-substitution");
        let parent_artifact = persist_typed_query_fuzz_replay(&parent_root, &replay)
            .expect("persist parent-substitution source");
        assert_eq!(
            parent_artifact
                .replay_workload()
                .expect("parent-substitution source descriptor remains consumable")
                .case
                .query,
            replay.minimized_query
        );
        let parent_path = typed_query_replay_path(&parent_root, &replay);
        let parent_directory = parent_path
            .parent()
            .expect("parent-substitution source has sidecar directory")
            .to_path_buf();
        let displaced_parent = parent_root.join("typed_query_tree-displaced-parent");
        let parent_bytes = canonical_bytes.clone();
        let parent_directory_for_hook = parent_directory.clone();
        let displaced_parent_for_hook = displaced_parent.clone();
        install_typed_query_fuzz_replay_final_binding_hook(move |path| {
            std::fs::rename(&parent_directory_for_hook, &displaced_parent_for_hook)
                .expect("move owned sidecar directory aside without deletion");
            std::fs::create_dir(&parent_directory_for_hook)
                .expect("create substituted owned sidecar directory");
            typed_query_write_new_owned_file(path, &parent_bytes);
        });
        assert!(matches!(
            load_typed_query_fuzz_replay(&parent_path),
            Err(GauntletError::UnsafeStorePath { path }) if path == parent_directory
        ));
        assert_eq!(
            std::fs::read(displaced_parent.join(&filename))
                .expect("displaced parent artifact remains intact"),
            canonical_bytes
        );
        assert_eq!(
            std::fs::read(&parent_path).expect("substituted parent artifact remains intact"),
            canonical_bytes
        );
    }

    #[cfg(all(
        feature = "fuzz-harness",
        any(
            target_os = "android",
            target_os = "ios",
            target_os = "linux",
            target_os = "macos",
            target_os = "tvos",
            target_os = "visionos",
            target_os = "watchos"
        )
    ))]
    #[test]
    fn typed_query_public_replay_capability_handles_late_substitution_and_rejects_same_inode_rewrite()
     {
        let replay = typed_query_test_replay(&[22, 1, 7, 99], TypedQueryTree::MixedHitMiss(1, 7));
        let canonical_bytes = replay.canonical_bytes().expect("canonical replay bytes");
        let filename = typed_query_fuzz_replay_filename(
            &replay.artifact_key().expect("canonical artifact key"),
        );

        let persist_root = typed_query_replay_test_root("late-persist-path-substitution");
        let persist_path = typed_query_replay_path(&persist_root, &replay);
        let persist_directory = persist_path
            .parent()
            .expect("persist path has a sidecar directory")
            .to_path_buf();
        let persist_displaced = persist_root.join("typed_query_tree-displaced-late-persist");
        let persist_decoy = vec![b'~'; canonical_bytes.len()];
        let persist_directory_for_hook = persist_directory.clone();
        let persist_displaced_for_hook = persist_displaced.clone();
        install_typed_query_fuzz_replay_post_display_verification_hook(move |path| {
            std::fs::rename(&persist_directory_for_hook, &persist_displaced_for_hook)
                .expect("move owned sidecar after display verification without deletion");
            std::fs::create_dir(&persist_directory_for_hook)
                .expect("create substituted owned sidecar directory");
            typed_query_write_new_owned_file(path, &persist_decoy);
        });
        let persisted = persist_typed_query_fuzz_replay(&persist_root, &replay)
            .expect("late display-path substitution must not replace the returned capability");
        assert_eq!(
            persisted
                .replay_workload()
                .expect("persisted capability must consume its original descriptor binding")
                .case
                .query,
            replay.minimized_query
        );
        assert_eq!(
            std::fs::read(persist_displaced.join(&filename))
                .expect("original late-persist artifact remains intact"),
            canonical_bytes
        );
        assert_eq!(
            std::fs::read(&persist_path).expect("ambient late-persist path resolves to the decoy"),
            vec![b'~'; canonical_bytes.len()],
            "the old PathBuf-returning API would have exposed this substituted entry"
        );

        let load_root = typed_query_replay_test_root("late-load-path-substitution");
        persist_typed_query_fuzz_replay(&load_root, &replay)
            .expect("persist late-load source descriptor capability");
        let load_path = typed_query_replay_path(&load_root, &replay);
        let load_directory = load_path
            .parent()
            .expect("load path has a sidecar directory")
            .to_path_buf();
        let load_displaced = load_root.join("typed_query_tree-displaced-late-load");
        let load_decoy = vec![b'!'; canonical_bytes.len()];
        let load_directory_for_hook = load_directory.clone();
        let load_displaced_for_hook = load_displaced.clone();
        install_typed_query_fuzz_replay_post_display_verification_hook(move |path| {
            std::fs::rename(&load_directory_for_hook, &load_displaced_for_hook)
                .expect("move owned load sidecar after display verification without deletion");
            std::fs::create_dir(&load_directory_for_hook)
                .expect("create substituted owned load sidecar directory");
            typed_query_write_new_owned_file(path, &load_decoy);
        });
        let loaded = load_typed_query_fuzz_replay(&load_path)
            .expect("late display-path substitution must not replace the loaded capability");
        assert_eq!(
            loaded
                .replay_workload()
                .expect("loaded capability must consume its original descriptor binding")
                .case
                .query,
            replay.minimized_query
        );
        assert_eq!(
            std::fs::read(load_displaced.join(&filename))
                .expect("original late-load artifact remains intact"),
            canonical_bytes
        );
        assert_eq!(
            std::fs::read(&load_path).expect("ambient late-load path resolves to the decoy"),
            vec![b'!'; canonical_bytes.len()]
        );

        let mutation_root = typed_query_replay_test_root("same-inode-equal-length-mutation");
        persist_typed_query_fuzz_replay(&mutation_root, &replay)
            .expect("persist same-inode mutation source descriptor capability");
        let mutation_path = typed_query_replay_path(&mutation_root, &replay);
        let mutation_bytes = vec![b'x'; canonical_bytes.len()];
        install_typed_query_fuzz_replay_post_display_verification_hook(move |path| {
            typed_query_rewrite_owned_file_same_length(path, &mutation_bytes);
        });
        assert!(matches!(
            load_typed_query_fuzz_replay(&mutation_path),
            Err(GauntletError::UnsafeStorePath { path }) if path == mutation_path
        ));
        assert_eq!(
            std::fs::read(&mutation_path).expect("same-inode hostile rewrite remains intact"),
            vec![b'x'; canonical_bytes.len()]
        );
    }

    #[cfg(all(
        feature = "fuzz-harness",
        any(
            target_os = "android",
            target_os = "ios",
            target_os = "linux",
            target_os = "macos",
            target_os = "tvos",
            target_os = "visionos",
            target_os = "watchos"
        )
    ))]
    #[test]
    fn typed_query_public_replay_capability_rejects_post_return_mutation_and_supports_concurrency()
    {
        use std::sync::{Arc, Barrier};

        let replay = typed_query_test_replay(&[22, 1, 7, 99], TypedQueryTree::MixedHitMiss(1, 7));
        let canonical_bytes = replay.canonical_bytes().expect("canonical replay bytes");

        let concurrent_root = typed_query_replay_test_root("concurrent-capability-consumption");
        let concurrent_artifact = Arc::new(
            persist_typed_query_fuzz_replay(&concurrent_root, &replay)
                .expect("persist one descriptor-bound artifact for concurrent consumption"),
        );
        let concurrent_path = typed_query_replay_path(&concurrent_root, &replay);
        let digest_barrier = Arc::new(Barrier::new(4));
        let path_for_digest_hook = concurrent_path.clone();
        let barrier_for_digest_hook = Arc::clone(&digest_barrier);
        let _digest_hook =
            crate::artifact::install_pinned_regular_file_before_digest_hook(move |path| {
                if path == path_for_digest_hook.as_path() {
                    barrier_for_digest_hook.wait();
                }
            });
        let expected_query = replay.minimized_query.clone();
        std::thread::scope(|scope| {
            let mut consumers = Vec::new();
            for _ in 0..4 {
                let artifact = Arc::clone(&concurrent_artifact);
                let expected_query = expected_query.clone();
                consumers.push(scope.spawn(move || {
                    assert_eq!(
                        artifact
                            .replay_workload()
                            .expect("concurrent positional replay consumption must succeed")
                            .case
                            .query,
                        expected_query
                    );
                }));
            }
            for consumer in consumers {
                consumer
                    .join()
                    .expect("concurrent replay consumer must not panic");
            }
        });

        let replacement_root = typed_query_replay_test_root("post-return-final-replacement");
        let replacement_artifact = persist_typed_query_fuzz_replay(&replacement_root, &replay)
            .expect("persist post-return replacement source");
        let replacement_path = typed_query_replay_path(&replacement_root, &replay);
        let displaced_replacement = replacement_path.with_file_name("displaced-replacement.json");
        std::fs::rename(&replacement_path, &displaced_replacement)
            .expect("move owned original entry aside after persist returned");
        typed_query_write_new_owned_file(&replacement_path, &canonical_bytes);
        assert!(matches!(
            replacement_artifact.replay_workload(),
            Err(GauntletError::UnsafeStorePath { path }) if path == replacement_path
        ));
        assert_eq!(
            std::fs::read(&displaced_replacement)
                .expect("displaced original post-return artifact remains intact"),
            canonical_bytes
        );

        let rewrite_root = typed_query_replay_test_root("post-return-same-inode-rewrite");
        let rewrite_artifact = persist_typed_query_fuzz_replay(&rewrite_root, &replay)
            .expect("persist post-return same-inode rewrite source");
        let rewrite_path = typed_query_replay_path(&rewrite_root, &replay);
        let rewritten_bytes = vec![b'x'; canonical_bytes.len()];
        typed_query_rewrite_owned_file_same_length(&rewrite_path, &rewritten_bytes);
        assert!(matches!(
            rewrite_artifact.replay_workload(),
            Err(GauntletError::UnsafeStorePath { path }) if path == rewrite_path
        ));
        assert_eq!(
            std::fs::read(&rewrite_path).expect("post-return same-inode rewrite remains intact"),
            rewritten_bytes
        );
    }

    #[cfg(all(
        feature = "fuzz-harness",
        any(
            target_os = "android",
            target_os = "ios",
            target_os = "linux",
            target_os = "macos",
            target_os = "tvos",
            target_os = "visionos",
            target_os = "watchos"
        )
    ))]
    #[test]
    fn typed_query_public_persist_concurrently_publishes_only_complete_same_key_bytes() {
        use std::sync::{Arc, Barrier};

        let replay = Arc::new(typed_query_test_replay(
            &[22, 1, 7, 99],
            TypedQueryTree::MixedHitMiss(1, 7),
        ));
        let root = typed_query_replay_test_root("concurrent-same-key-publication");
        let sidecar_directory = root.join(TYPED_QUERY_FUZZ_REPLAY_DIRECTORY);
        let publish_barrier = Arc::new(Barrier::new(2));
        let sidecar_for_hook = sidecar_directory.clone();
        let publish_barrier_for_hook = Arc::clone(&publish_barrier);
        let _publish_hook =
            crate::artifact::install_pinned_regular_file_before_publish_hook(move |directory| {
                if directory == sidecar_for_hook.as_path() {
                    publish_barrier_for_hook.wait();
                }
            });
        let (first, second) = std::thread::scope(|scope| {
            let first_root = &root;
            let first_replay = Arc::clone(&replay);
            let first = scope
                .spawn(move || persist_typed_query_fuzz_replay(first_root, first_replay.as_ref()));
            let second_root = &root;
            let second_replay = Arc::clone(&replay);
            let second = scope.spawn(move || {
                persist_typed_query_fuzz_replay(second_root, second_replay.as_ref())
            });
            (
                first.join().expect("first same-key writer must not panic"),
                second
                    .join()
                    .expect("second same-key writer must not panic"),
            )
        });
        let first = first.expect("first same-key writer must receive a complete artifact");
        let second = second.expect("second same-key writer must receive a complete artifact");
        assert_eq!(
            first
                .replay_workload()
                .expect("first same-key artifact must be consumable")
                .case
                .query,
            replay.as_ref().minimized_query
        );
        assert_eq!(
            second
                .replay_workload()
                .expect("second same-key artifact must be consumable")
                .case
                .query,
            replay.as_ref().minimized_query
        );
        assert_eq!(
            std::fs::read(typed_query_replay_path(&root, replay.as_ref()))
                .expect("same-key final artifact must be complete"),
            replay
                .as_ref()
                .canonical_bytes()
                .expect("canonical same-key replay bytes")
        );
    }

    #[cfg(all(
        feature = "fuzz-harness",
        any(
            target_os = "android",
            target_os = "ios",
            target_os = "linux",
            target_os = "macos",
            target_os = "tvos",
            target_os = "visionos",
            target_os = "watchos"
        )
    ))]
    #[test]
    fn typed_query_existing_success_syncs_before_returning_while_winner_is_parked() {
        use std::{
            sync::{
                Arc, Mutex,
                atomic::{AtomicUsize, Ordering},
                mpsc::{self, RecvTimeoutError},
            },
            time::Duration,
        };

        const RENDEZVOUS_TIMEOUT: Duration = Duration::from_secs(2);

        let replay = typed_query_test_replay(&[22, 1, 7, 99], TypedQueryTree::MixedHitMiss(1, 7));
        let root = typed_query_replay_test_root("existing-success-directory-sync");
        let sidecar_directory = root.join(TYPED_QUERY_FUZZ_REPLAY_DIRECTORY);
        let (sync_event_sender, sync_event_receiver) = mpsc::sync_channel(2);
        let (release_winner_sender, release_winner_receiver) = mpsc::sync_channel(1);
        let (release_loser_sender, release_loser_receiver) = mpsc::sync_channel(1);
        let release_winner_receiver = Arc::new(Mutex::new(release_winner_receiver));
        let release_loser_receiver = Arc::new(Mutex::new(release_loser_receiver));
        let sync_calls = Arc::new(AtomicUsize::new(0));
        let sidecar_directory_for_hook = sidecar_directory.clone();
        let sync_event_sender_for_hook = sync_event_sender.clone();
        let release_winner_receiver_for_hook = Arc::clone(&release_winner_receiver);
        let release_loser_receiver_for_hook = Arc::clone(&release_loser_receiver);
        let sync_calls_for_hook = Arc::clone(&sync_calls);
        let _sync_hook = crate::artifact::install_pinned_directory_before_sync_hook(move |path| {
            if path != sidecar_directory_for_hook.as_path() {
                return;
            }
            let (label, release_receiver) = match sync_calls_for_hook.fetch_add(1, Ordering::SeqCst)
            {
                0 => ("winner", Some(&release_winner_receiver_for_hook)),
                1 => ("loser", Some(&release_loser_receiver_for_hook)),
                _ => ("unexpected", None),
            };
            sync_event_sender_for_hook
                .send(label)
                .expect("bounded directory-sync rendezvous receiver must remain live");
            if let Some(release_receiver) = release_receiver {
                release_receiver
                    .lock()
                    .expect("bounded directory-sync release mutex must not be poisoned")
                    .recv_timeout(RENDEZVOUS_TIMEOUT)
                    .expect("bounded directory-sync rendezvous release must arrive");
            }
        });

        let winner_root = root.clone();
        let winner_replay = replay.clone();
        let winner = std::thread::spawn(move || {
            persist_typed_query_fuzz_replay(&winner_root, &winner_replay)
        });
        assert_eq!(
            sync_event_receiver
                .recv_timeout(RENDEZVOUS_TIMEOUT)
                .expect("winner must reach the post-rename, pre-directory-sync rendezvous"),
            "winner"
        );

        let loser_root = root.clone();
        let loser_replay = replay.clone();
        let loser =
            std::thread::spawn(move || persist_typed_query_fuzz_replay(&loser_root, &loser_replay));
        assert_eq!(
            sync_event_receiver
                .recv_timeout(RENDEZVOUS_TIMEOUT)
                .expect("loser must authenticate the existing entry and reach its directory sync"),
            "loser"
        );
        assert!(
            !loser.is_finished(),
            "loser must not return before its own held-directory sync completes"
        );
        assert!(matches!(
            sync_event_receiver.recv_timeout(Duration::from_millis(100)),
            Err(RecvTimeoutError::Timeout)
        ));

        release_loser_sender
            .send(())
            .expect("release the loser after its bounded sync rendezvous");
        let loser = loser
            .join()
            .expect("loser must not panic during existing-entry synchronization")
            .expect("loser must return a synchronized descriptor capability");
        assert_eq!(
            loser
                .replay_workload()
                .expect("synchronized loser capability must remain consumable")
                .case
                .query,
            replay.minimized_query
        );

        release_winner_sender
            .send(())
            .expect("release the winner after the loser has synchronized the directory");
        let winner = winner
            .join()
            .expect("winner must not panic after the bounded rendezvous")
            .expect("winner must return a synchronized descriptor capability");
        assert_eq!(
            winner
                .replay_workload()
                .expect("synchronized winner capability must remain consumable")
                .case
                .query,
            replay.minimized_query
        );
        assert_eq!(
            sync_calls.load(Ordering::SeqCst),
            2,
            "winner and loser must each issue the held-directory sync before success"
        );
    }

    #[cfg(feature = "fuzz-harness")]
    #[test]
    fn typed_query_grammar_covers_oov_mixed_and_lenient_malformed_forms() {
        let vocabulary = vec!["term0".to_owned(), "term1".to_owned()];
        let oov = TypedQueryTree::OovOnly(9);
        let mixed = TypedQueryTree::MixedHitMiss(1, 9);
        assert_eq!(oov.render(&vocabulary), "oovterm9");
        assert_eq!(mixed.render(&vocabulary), "term1 OR oovterm9");
        for tree in [
            TypedQueryTree::UnterminatedPhrase(1, 0),
            TypedQueryTree::TrailingBoolean(1),
            TypedQueryTree::MalformedNestedOperator(1, 0, 1),
            TypedQueryTree::MalformedOperator(1, 0),
            TypedQueryTree::MalformedField(1),
            TypedQueryTree::MalformedEscape(1),
            TypedQueryTree::MalformedBoost(1),
            TypedQueryTree::MalformedSlop(1, 0),
            oov,
            mixed,
        ] {
            assert_eq!(TypedQueryTree::from_input(&tree.canonical_input()), tree);
        }
    }

    #[cfg(feature = "fuzz-harness")]
    #[test]
    fn typed_query_malformed_lane_classifies_lenient_quill_and_oracle_acceptance() {
        let workload = materialize_typed_query_fuzz_workload(&[15, 1, 2, 3])
            .expect("malformed typed-query workload");
        let documents = workload
            .documents
            .iter()
            .cloned()
            .map(frankensearch_core::IndexableDocument::from)
            .collect::<Vec<_>>();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (subject, oracle) = scalar_g1a_fuzz_pair(&cx, &documents)
                .await
                .expect("committed malformed-lane engines");
            let subject_observation = subject
                .observe(&cx, &workload.case)
                .await
                .expect("Quill parse_lenient observation");
            let oracle_observation = oracle
                .observe(&cx, &workload.case)
                .await
                .expect("exact successful pinned-oracle observation");
            let asymmetry = subject
                .classify_typed_query_lenient_asymmetry(
                    workload.ast,
                    &workload.case.query,
                    &oracle_observation,
                )
                .expect("typed lenient asymmetry");
            assert!(
                !asymmetry.quill_diagnostic_kinds.is_empty(),
                "Quill must retain a malformed-syntax recovery diagnostic"
            );
            assert_ne!(
                asymmetry.recovered_quill_ast,
                Query::empty(),
                "this nested malformed query must recover a usable Quill AST rather than vanish"
            );
            assert_eq!(
                asymmetry.oracle_behavior,
                TypedQueryOracleBehavior::AcceptedWithoutAstDifferences
            );
            assert!(subject_observation.ast_differences.is_empty());
            assert!(oracle_observation.ast_differences.is_empty());
        });
    }

    #[cfg(feature = "fuzz-harness")]
    #[test]
    fn scalar_g1a_fuzz_shrink_factories_create_independent_fresh_lifecycles() {
        let (mut make_subject, mut make_oracle) = QuillSubject::scalar_g1a_fuzz_shrink_factories();
        let mut first_subject = make_subject().expect("first fresh fuzz subject");
        let mut second_subject = make_subject().expect("second fresh fuzz subject");
        let mut first_oracle = make_oracle().expect("first fresh fuzz oracle");
        let mut second_oracle = make_oracle().expect("second fresh fuzz oracle");

        assert_eq!(first_subject.descriptor().family, EngineFamily::Quill);
        assert_eq!(second_subject.descriptor().family, EngineFamily::Quill);
        assert_eq!(first_oracle.descriptor().family, EngineFamily::Tantivy);
        assert_eq!(second_oracle.descriptor().family, EngineFamily::Tantivy);
        assert_eq!(
            first_subject.semantic_contract(),
            SemanticContract::scalar_g1a()
        );
        assert_eq!(
            first_oracle.semantic_contract(),
            SemanticContract::scalar_g1a()
        );

        let corpus = crate::SyntheticCorpus::new(crate::SyntheticCorpusSpec {
            seed: 0x6273_6a77_fa57_0001,
            document_count: 16,
            vocabulary_size: 32,
            zipf_exponent: crate::ZipfExponent::S11,
            max_document_bytes: 256,
        })
        .expect("fresh-factory test corpus");
        let manifest = corpus.manifest().expect("fresh-factory test manifest");
        let documents = corpus.iter().collect::<Vec<_>>();
        let contract = SemanticContract::scalar_g1a();

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for engine in [
                &mut first_subject,
                &mut second_subject,
                &mut first_oracle,
                &mut second_oracle,
            ] {
                engine
                    .begin_corpus(&cx, &manifest, &contract)
                    .await
                    .expect("each factory result must begin a new lifecycle");
            }
            for engine in [
                &mut first_subject,
                &mut second_subject,
                &mut first_oracle,
                &mut second_oracle,
            ] {
                engine
                    .index_batch(&cx, &documents)
                    .await
                    .expect("each factory result must index independently");
            }
            for engine in [
                &mut first_subject,
                &mut second_subject,
                &mut first_oracle,
                &mut second_oracle,
            ] {
                let receipt = engine
                    .commit_corpus(&cx, &manifest, &contract)
                    .await
                    .expect("each factory result must commit its own corpus");
                assert_eq!(receipt.document_count, manifest.document_count);
                assert_eq!(
                    receipt.semantic_contract,
                    SemanticContract::scalar_g1a(),
                    "each independent factory lifecycle must retain scalar G1a"
                );
            }
        });
    }
}
