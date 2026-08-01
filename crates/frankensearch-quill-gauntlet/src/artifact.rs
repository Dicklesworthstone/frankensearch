use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use xxhash_rust::xxh3::Xxh3;

use crate::GauntletError;
use crate::comparator::{
    ComparatorConfig, LexicalBoundary, LexicalComparisonStatus, LexicalContractComparison,
    LexicalEngineRole, LexicalExposureContract, LexicalObservationContext, compare_observations,
};
use crate::engine::{EnginePairIdentity, HarnessRun};
use crate::generator::{
    GENERATOR_ID, GeneratedQueryCase, QuerySuiteSource, validate_generated_case_metadata,
};
use crate::runner::{
    CampaignContractMode, CampaignReport, DivergenceRegisterEntry, SemanticContract,
    lexical_backend_identity, lexical_query_contract_sha256,
};
use crate::version_contract::{OracleVersionContract, oracle_version_contract};

pub const OBJECT_SCHEMA_VERSION: u32 = 6;
pub const CANONICALIZATION_VERSION: u32 = 1;
/// Current mutable run-manifest schema.
///
/// Version 2 pins the referenced current object address to domain-separated
/// SHA-256. Version 1 carried legacy XXH3-64 addresses and is decode-only.
pub const RUN_MANIFEST_SCHEMA_VERSION: u32 = 2;
const HASH_DOMAIN_V1: &[u8] = b"frankensearch-quill-gauntlet:artifact-object:v1\0";
const HASH_DOMAIN_V2: &[u8] = b"frankensearch-quill-gauntlet:artifact-object:v2\0";
const HASH_DOMAIN_V3: &[u8] = b"frankensearch-quill-gauntlet:artifact-object:v3\0";
const HASH_DOMAIN_V5: &[u8] = b"frankensearch-quill-gauntlet:artifact-object:v5\0";
const HASH_DOMAIN_V6: &[u8] = b"frankensearch-quill-gauntlet:artifact-object:v6\0";
const PRODUCER_BUILD_IDENTITY_HASH_DOMAIN: &[u8] =
    b"frankensearch-quill-gauntlet:producer-build-identity:v1\0";
const PRODUCER_BUILD_IDENTITY_SCHEMA_VERSION: u32 = 1;
const PRODUCER_CONTRACT_VERSION_V4: &str = "frankensearch.quill-local-perf-producer.v4";
const MAX_CAMPAIGN_RESERVATION_BYTES: u64 = 512 * 1024 * 1024;
const MAX_CAMPAIGN_REPORT_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_CAMPAIGN_RUN_MANIFEST_BYTES: u64 = 2 * 1024 * 1024;
const MAX_CAMPAIGN_OBJECT_BYTES: u64 = 512 * 1024 * 1024;

/// How the gauntlet build script established the source identity embedded in
/// the exact producer binary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GauntletProducerSourceVerification {
    /// Decode-only sentinel for artifact schemas that predate this identity.
    #[default]
    LegacyMissing,
    /// The build script resolved the exact repository root and Git state.
    GitCheckoutVerified,
    /// A Git-less diagnostic build recorded an explicit caller identity.
    ExplicitUnverified,
    /// Neither a repository identity nor a complete explicit identity existed.
    Unavailable,
}

/// Build-script-sealed identity of the binary that produced an artifact.
///
/// Current artifact construction compares this complete record against the
/// values compiled into the executing gauntlet. Callers therefore cannot make
/// two engine descriptors agree on a fabricated source revision and have that
/// agreement mistaken for producer provenance. Dirty and unverified builds
/// remain recordable diagnostics; verified campaign admission rejects them.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GauntletProducerBuildIdentity {
    pub schema_version: u32,
    pub producer_contract_version: String,
    pub source_git_revision: String,
    pub source_git_dirty: bool,
    pub source_verification: GauntletProducerSourceVerification,
    pub cargo_lock_sha256: String,
    pub rustc_version_verbose_hex: String,
    pub target_triple: String,
    pub cargo_profile: String,
    pub enabled_features: Vec<String>,
    pub enabled_features_sha256: String,
    pub executable_sha256: String,
    pub executable_byte_len: u64,
}

impl GauntletProducerBuildIdentity {
    pub(crate) fn compiled() -> Result<Self, GauntletError> {
        let source_git_dirty = match env!("QUILL_PERF_PRODUCER_GIT_DIRTY") {
            "true" => true,
            "false" => false,
            _ => {
                return Err(GauntletError::InvalidContract {
                    reason: "compiled producer dirty state is not canonical".to_owned(),
                });
            }
        };
        let source_verification = match env!("QUILL_PERF_PRODUCER_SOURCE_VERIFICATION") {
            "git_checkout_verified" => GauntletProducerSourceVerification::GitCheckoutVerified,
            "explicit_unverified" => GauntletProducerSourceVerification::ExplicitUnverified,
            "unavailable" => GauntletProducerSourceVerification::Unavailable,
            _ => {
                return Err(GauntletError::InvalidContract {
                    reason: "compiled producer source-verification mode is unknown".to_owned(),
                });
            }
        };
        let (executable_sha256, executable_byte_len) = current_executable_identity()?;
        let identity = Self {
            schema_version: PRODUCER_BUILD_IDENTITY_SCHEMA_VERSION,
            producer_contract_version: env!("QUILL_PERF_PRODUCER_CONTRACT_VERSION").to_owned(),
            source_git_revision: env!("QUILL_PERF_PRODUCER_GIT_REVISION").to_owned(),
            source_git_dirty,
            source_verification,
            cargo_lock_sha256: env!("QUILL_PERF_PRODUCER_CARGO_LOCK_SHA256").to_owned(),
            rustc_version_verbose_hex: env!("QUILL_PERF_PRODUCER_RUSTC_VV_HEX").to_owned(),
            target_triple: env!("QUILL_PERF_PRODUCER_TARGET_TRIPLE").to_owned(),
            cargo_profile: env!("QUILL_PERF_PRODUCER_CARGO_PROFILE").to_owned(),
            enabled_features: env!("QUILL_PERF_PRODUCER_ENABLED_FEATURES")
                .split(',')
                .filter(|feature| !feature.is_empty())
                .map(str::to_owned)
                .collect(),
            enabled_features_sha256: env!("QUILL_PERF_PRODUCER_ENABLED_FEATURES_SHA256").to_owned(),
            executable_sha256,
            executable_byte_len,
        };
        identity.validate_structure()?;
        Ok(identity)
    }

    pub(crate) fn identity_hash(&self) -> Result<String, GauntletError> {
        self.validate_structure()?;
        let mut hasher = Sha256::new();
        hasher.update(PRODUCER_BUILD_IDENTITY_HASH_DOMAIN);
        hasher.update(serde_json::to_vec(self)?);
        Ok(lower_hex(&hasher.finalize()))
    }

    pub(crate) fn rustc_version_verbose(&self) -> Result<String, GauntletError> {
        let bytes = decode_lower_hex(&self.rustc_version_verbose_hex).ok_or_else(|| {
            GauntletError::InvalidContract {
                reason: "compiled producer rustc identity is not canonical lowercase hex"
                    .to_owned(),
            }
        })?;
        String::from_utf8(bytes).map_err(|error| GauntletError::InvalidContract {
            reason: format!("compiled producer rustc identity is not UTF-8: {error}"),
        })
    }

    pub(crate) fn validate_structure(&self) -> Result<(), GauntletError> {
        let source_is_well_formed = match self.source_verification {
            GauntletProducerSourceVerification::GitCheckoutVerified
            | GauntletProducerSourceVerification::ExplicitUnverified => {
                is_lower_hex_len(&self.source_git_revision, 40)
            }
            GauntletProducerSourceVerification::Unavailable => {
                self.source_git_revision == "unavailable" && self.source_git_dirty
            }
            GauntletProducerSourceVerification::LegacyMissing => false,
        };
        let cargo_lock_is_well_formed = self.cargo_lock_sha256 == "unavailable"
            || is_lower_hex_len(&self.cargo_lock_sha256, 64);
        let rustc_is_well_formed = !self.rustc_version_verbose_hex.is_empty()
            && self.rustc_version_verbose_hex.len() <= 32 * 1024
            && self.rustc_version_verbose_hex.len().is_multiple_of(2)
            && is_lower_hex_text(&self.rustc_version_verbose_hex);
        let text_is_well_formed = [self.target_triple.as_str(), self.cargo_profile.as_str()]
            .into_iter()
            .all(|value| {
                !value.is_empty()
                    && value.len() <= 256
                    && value.trim() == value
                    && value.bytes().all(|byte| byte.is_ascii_graphic())
            });
        let features_are_canonical = self.enabled_features.iter().all(|feature| {
            !feature.is_empty()
                && feature.len() <= 128
                && feature.bytes().all(|byte| {
                    byte.is_ascii_lowercase()
                        || byte.is_ascii_digit()
                        || matches!(byte, b'_' | b'-')
                })
        }) && self
            .enabled_features
            .windows(2)
            .all(|pair| pair[0] < pair[1]);
        let mut feature_hasher = Sha256::new();
        feature_hasher.update(self.enabled_features.join("\n").as_bytes());
        let expected_features_sha256 = lower_hex(&feature_hasher.finalize());
        if self.schema_version != PRODUCER_BUILD_IDENTITY_SCHEMA_VERSION
            || self.producer_contract_version != PRODUCER_CONTRACT_VERSION_V4
            || !source_is_well_formed
            || !cargo_lock_is_well_formed
            || !rustc_is_well_formed
            || !text_is_well_formed
            || !features_are_canonical
            || !is_lower_hex_len(&self.enabled_features_sha256, 64)
            || self.enabled_features_sha256 != expected_features_sha256
            || !is_lower_hex_len(&self.executable_sha256, 64)
            || self.executable_byte_len == 0
        {
            return Err(GauntletError::InvalidContract {
                reason: "artifact producer build identity is missing, malformed, or internally inconsistent"
                    .to_owned(),
            });
        }
        Ok(())
    }

    pub(crate) fn validate_engines(
        &self,
        engines: &EnginePairIdentity,
    ) -> Result<(), GauntletError> {
        for descriptor in [&engines.subject, &engines.oracle] {
            if descriptor.source_revision != self.source_git_revision
                || descriptor.source_dirty != self.source_git_dirty
            {
                return Err(GauntletError::InvalidContract {
                    reason: "engine descriptor is not bound to the artifact's sealed producer build identity"
                        .to_owned(),
                });
            }
        }
        Ok(())
    }

    pub(crate) fn validate_matches_compiled(&self) -> Result<(), GauntletError> {
        self.validate_structure()?;
        if self != &Self::compiled()? {
            return Err(GauntletError::InvalidContract {
                reason: "producer identity does not match the exact executing gauntlet binary"
                    .to_owned(),
            });
        }
        Ok(())
    }

    pub(crate) fn validate_admissible(&self) -> Result<(), GauntletError> {
        self.validate_structure()?;
        if self.source_verification != GauntletProducerSourceVerification::GitCheckoutVerified
            || self.source_git_dirty
            || !is_lower_hex_len(&self.cargo_lock_sha256, 64)
        {
            return Err(GauntletError::InvalidContract {
                reason: "admissible artifact evidence requires a clean Git-verified producer and an exact Cargo.lock identity"
                    .to_owned(),
            });
        }
        Ok(())
    }
}

fn current_executable_identity() -> Result<(String, u64), GauntletError> {
    static IDENTITY: std::sync::OnceLock<Result<(String, u64), String>> =
        std::sync::OnceLock::new();
    IDENTITY
        .get_or_init(capture_current_executable_identity)
        .clone()
        .map_err(|reason| GauntletError::InvalidContract { reason })
}

fn capture_current_executable_identity() -> Result<(String, u64), String> {
    #[cfg(target_os = "linux")]
    let path = PathBuf::from("/proc/self/exe");
    #[cfg(not(target_os = "linux"))]
    let path = std::env::current_exe()
        .map_err(|error| format!("cannot resolve the producer executable: {error}"))?;
    let mut file = File::open(&path)
        .map_err(|error| format!("cannot open the producer executable: {error}"))?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("cannot stat the producer executable: {error}"))?;
    if !metadata.is_file() {
        return Err("producer executable identity requires a regular file".to_owned());
    }
    let mut hasher = Sha256::new();
    let mut byte_len = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("cannot hash the producer executable: {error}"))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        byte_len = byte_len
            .checked_add(u64::try_from(read).map_err(|error| {
                format!("producer executable read length does not fit u64: {error}")
            })?)
            .ok_or_else(|| "producer executable length overflowed u64".to_owned())?;
    }
    if byte_len == 0 || byte_len != metadata.len() {
        return Err("producer executable changed while its identity was captured".to_owned());
    }
    Ok((lower_hex(&hasher.finalize()), byte_len))
}

fn is_lower_hex_text(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn decode_lower_hex(value: &str) -> Option<Vec<u8>> {
    if !value.len().is_multiple_of(2) || !is_lower_hex_text(value) {
        return None;
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let high = lower_hex_nibble(pair[0])?;
            let low = lower_hex_nibble(pair[1])?;
            Some((high << 4) | low)
        })
        .collect()
}

const fn lower_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn is_lower_hex_len(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len && is_lower_hex_text(value)
}

/// Immutable campaign context omitted from legacy one-case artifacts.
///
/// The hashes are opaque references to the exact corpus/query manifests; their
/// referenced bundles are verified by the campaign report/replay layer. This
/// object locally binds those references to the complete rich query, semantic
/// profile, pagination, and reviewed-divergence evidence that cannot be
/// represented by the raw-query-only [`crate::DifferentialCase`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignArtifactContext {
    pub corpus_manifest_hash: String,
    pub query_manifest_hash: String,
    pub query_suite_source: QuerySuiteSource,
    pub query_source_identity_sha256: String,
    pub semantic_contract: SemanticContract,
    pub contract_mode: CampaignContractMode,
    pub query_seed: u64,
    pub query: GeneratedQueryCase,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registered_divergence: Option<DivergenceRegisterEntry>,
}

/// Explicit total-contract scope carried by every current artifact.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case", deny_unknown_fields)]
pub enum ArtifactLexicalContractEvidence {
    /// Decode-only marker for pre-v3 objects, which are no longer admissible.
    #[default]
    LegacyPreV3Missing,
    /// The object intentionally proves only the legacy rich result envelope.
    RankEnvelopeOnly,
    /// Complete replayable ordinary `LexicalSearch` comparison.
    CoreLexicalV3 {
        comparison: Box<LexicalContractComparison>,
    },
}

/// Immutable comparison object. Run-local provenance is deliberately absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactObject {
    pub object_schema_version: u32,
    pub canonicalization_version: u32,
    pub oracle_version: OracleVersionContract,
    #[serde(default)]
    pub producer_build_identity: GauntletProducerBuildIdentity,
    pub engines: EnginePairIdentity,
    pub case: crate::DifferentialCase,
    pub comparator_config: ComparatorConfig,
    pub comparison: crate::ComparisonReport,
    #[serde(default)]
    pub lexical_contract: ArtifactLexicalContractEvidence,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub campaign: Option<CampaignArtifactContext>,
}

impl ArtifactObject {
    /// Build an immutable object from one completed harness run.
    ///
    /// # Errors
    ///
    /// Returns an error when the committed oracle version contract is invalid.
    pub fn from_diagnostic_run(run: HarnessRun) -> Result<Self, GauntletError> {
        run.validate_for_creation()?;
        Ok(Self {
            object_schema_version: OBJECT_SCHEMA_VERSION,
            canonicalization_version: CANONICALIZATION_VERSION,
            oracle_version: oracle_version_contract()?,
            producer_build_identity: run.producer_build_identity,
            engines: run.engines,
            case: run.case,
            comparator_config: run.comparator_config,
            comparison: run.comparison,
            lexical_contract: ArtifactLexicalContractEvidence::RankEnvelopeOnly,
            campaign: None,
        })
    }

    pub(crate) fn from_run(run: HarnessRun) -> Result<Self, GauntletError> {
        Self::from_diagnostic_run(run)
    }

    /// Build an immutable object for one case in a generated campaign.
    ///
    /// # Errors
    ///
    /// Returns an error when the committed oracle version contract is invalid.
    pub(crate) fn from_campaign_run(
        run: HarnessRun,
        campaign: CampaignArtifactContext,
        lexical_contract: ArtifactLexicalContractEvidence,
    ) -> Result<Self, GauntletError> {
        let mut object = Self::from_run(run)?;
        object.lexical_contract = lexical_contract;
        object.campaign = Some(campaign);
        Ok(object)
    }

    /// Canonical compact JSON bytes used as the immutable object body.
    ///
    /// Hashed DTOs use fixed struct field order, typed metadata, integer score
    /// bits, and preserved vector order. The output has no trailing newline.
    ///
    /// # Errors
    ///
    /// Returns a JSON serialization error if the schema stops being encodable.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, GauntletError> {
        Ok(serde_json::to_vec(self)?)
    }

    /// Compute the versioned, domain-separated object address.
    ///
    /// Legacy v1/v2 objects retain their historical XXH3-64 address so they can
    /// be decoded and diagnosed. Legacy v3, pre-run-identity v5, and current
    /// v6 objects use distinct SHA-256 domains. Reserved pre-policy v4 has no
    /// registered hash domain and is never admissible evidence.
    ///
    /// For a legacy object decoded from historical bytes, this method hashes
    /// the DTO's current canonical reserialization, including fields added
    /// with serde defaults. It is therefore not an original-byte verifier.
    /// Durable legacy-address verification would have to hash the stored bytes
    /// before decoding. The current verified-campaign loader instead rejects
    /// legacy schemas before address admission because they cannot satisfy the
    /// total lexical contract.
    ///
    /// # Errors
    ///
    /// Returns an error when canonical serialization fails.
    pub fn object_hash(&self) -> Result<String, GauntletError> {
        let bytes = self.canonical_bytes()?;
        hash_object_bytes(&bytes, self.object_schema_version)
    }

    pub(crate) fn validate(&self) -> Result<(), GauntletError> {
        validate_current_object_schema(self.object_schema_version)
            .map_err(|reason| GauntletError::InvalidContract { reason })?;
        if self.canonicalization_version != CANONICALIZATION_VERSION
            || self.oracle_version != oracle_version_contract()?
        {
            return Err(GauntletError::InvalidContract {
                reason: "artifact canonicalization or embedded oracle contract is invalid"
                    .to_owned(),
            });
        }
        self.producer_build_identity.validate_structure()?;
        self.producer_build_identity
            .validate_engines(&self.engines)?;
        self.engines.validate_gauntlet_contract()?;
        if self.engines.comparison_mode == crate::ComparisonMode::CrossEngine
            && self.engines.oracle.crate_version != self.oracle_version.lexical_package_version
        {
            return Err(GauntletError::InvalidContract {
                reason: "artifact oracle package does not match its independently embedded version contract"
                    .to_owned(),
            });
        }
        self.comparator_config.validate_contract()?;
        validate_generated_case_metadata(&self.case)?;
        if self.campaign.is_none()
            && self.case.metadata.generator_id.as_deref() == Some(GENERATOR_ID)
        {
            return Err(GauntletError::InvalidContract {
                reason: "current generator provenance requires campaign manifest context"
                    .to_owned(),
            });
        }
        if let Some(campaign) = &self.campaign {
            campaign.validate_against(
                &self.engines,
                &self.case,
                &self.comparison,
                &self.lexical_contract,
            )?;
        } else if self.lexical_contract != ArtifactLexicalContractEvidence::RankEnvelopeOnly {
            return Err(GauntletError::InvalidContract {
                reason: "standalone artifact has an invalid total lexical evidence scope"
                    .to_owned(),
            });
        }
        self.case.validate_observations(
            &self.engines,
            &self.comparison.subject,
            &self.comparison.oracle,
        )?;
        let recomputed = compare_observations(
            self.comparison.subject.clone(),
            self.comparison.oracle.clone(),
            self.comparator_config,
        )?;
        if recomputed != self.comparison {
            return Err(GauntletError::InvalidContract {
                reason: "artifact comparison report does not match its observations".to_owned(),
            });
        }
        Ok(())
    }

    pub(crate) fn validate_admissible(&self) -> Result<(), GauntletError> {
        self.validate()?;
        self.producer_build_identity.validate_admissible()
    }
}

impl CampaignArtifactContext {
    fn validate_against(
        &self,
        engines: &EnginePairIdentity,
        case: &crate::DifferentialCase,
        comparison: &crate::ComparisonReport,
        lexical_contract: &ArtifactLexicalContractEvidence,
    ) -> Result<(), GauntletError> {
        let hashes_are_canonical = [
            self.corpus_manifest_hash.as_str(),
            self.query_manifest_hash.as_str(),
            self.query_source_identity_sha256.as_str(),
        ]
        .into_iter()
        .all(is_lower_sha256)
            && self.semantic_contract.validate().is_ok()
            && engines.semantic_contract.as_ref() == Some(&self.semantic_contract);
        let query_matches = self.query.id == case.fixture_id
            && self.query.query == case.query
            && self.query.limit == case.limit
            && self.query.offset == case.offset
            && self.query.count_requested == case.count_requested;
        let corpus_matches =
            case.metadata.corpus_hash.as_deref() == Some(self.corpus_manifest_hash.as_str());
        let generated_metadata_matches = match self.query_suite_source {
            QuerySuiteSource::Generated => {
                case.metadata.generator_id.as_deref() == Some(GENERATOR_ID)
                    && case.metadata.generator_seed == Some(self.query_seed)
            }
            QuerySuiteSource::ExplicitCases => {
                case.metadata.generator_id.is_none() && case.metadata.generator_seed.is_none()
            }
        } && corpus_matches;
        let register_matches = self.registered_divergence.as_ref().is_none_or(|entry| {
            entry.validate().is_ok() && entry.matches_comparison(&self.query, comparison)
        });
        if !hashes_are_canonical
            || !query_matches
            || !generated_metadata_matches
            || !register_matches
        {
            return Err(GauntletError::InvalidContract {
                reason:
                    "campaign artifact context or lexical evidence does not match its manifests, engines, or differential case".to_owned(),
            });
        }
        match (self.contract_mode, lexical_contract) {
            (
                CampaignContractMode::RankEnvelopeOnly,
                ArtifactLexicalContractEvidence::RankEnvelopeOnly,
            ) => Ok(()),
            (
                CampaignContractMode::CoreLexicalV3,
                ArtifactLexicalContractEvidence::CoreLexicalV3 {
                    comparison: lexical,
                },
            ) => self.validate_core_lexical_contract(engines, lexical),
            (
                CampaignContractMode::RankEnvelopeOnly | CampaignContractMode::CoreLexicalV3,
                ArtifactLexicalContractEvidence::LegacyPreV3Missing
                | ArtifactLexicalContractEvidence::RankEnvelopeOnly
                | ArtifactLexicalContractEvidence::CoreLexicalV3 { .. },
            ) => Err(GauntletError::InvalidContract {
                reason: "campaign contract mode does not match its lexical evidence schema"
                    .to_owned(),
            }),
        }
    }

    fn validate_core_lexical_contract(
        &self,
        engines: &EnginePairIdentity,
        comparison: &LexicalContractComparison,
    ) -> Result<(), GauntletError> {
        if self.query.syntax != crate::QuerySyntax::Default {
            return Err(GauntletError::InvalidContract {
                reason: "core lexical evidence cannot be attached to a non-default query"
                    .to_owned(),
            });
        }
        comparison.validate_replay()?;
        let query_contract_sha256 = lexical_query_contract_sha256(&self.semantic_contract)?;
        let limit =
            usize::try_from(self.query.limit).map_err(|_| GauntletError::InvalidContract {
                reason: "artifact lexical query limit does not fit usize".to_owned(),
            })?;
        let subject_backend =
            lexical_backend_identity(&engines.subject, &self.corpus_manifest_hash)?;
        let oracle_backend = lexical_backend_identity(&engines.oracle, &self.corpus_manifest_hash)?;
        let expected_subject_context = LexicalObservationContext::new(
            LexicalBoundary::FullSearch,
            subject_backend,
            self.corpus_manifest_hash.clone(),
            query_contract_sha256.clone(),
            &self.query.query,
            self.query_seed,
            limit,
            LexicalExposureContract::CORE_LEXICAL_SEARCH,
        )?;
        let expected_oracle_context = LexicalObservationContext::new(
            LexicalBoundary::FullSearch,
            oracle_backend,
            self.corpus_manifest_hash.clone(),
            query_contract_sha256,
            &self.query.query,
            self.query_seed,
            limit,
            LexicalExposureContract::CORE_LEXICAL_SEARCH,
        )?;
        let subject = &comparison.subject;
        let oracle = &comparison.oracle;
        if subject.engine_role() != LexicalEngineRole::Subject
            || oracle.engine_role() != LexicalEngineRole::Oracle
            || subject.snapshot_sha256() != self.corpus_manifest_hash
            || oracle.snapshot_sha256() != self.corpus_manifest_hash
            || subject.full_search().context != expected_subject_context
            || oracle.full_search().context != expected_oracle_context
        {
            return Err(GauntletError::InvalidContract {
                reason:
                    "core lexical evidence is not bound to the artifact query, snapshot, roles, or backend descriptors"
                        .to_owned(),
            });
        }
        if comparison.status == LexicalComparisonStatus::Equivalent
            && comparison.first_mismatch.is_some()
        {
            return Err(GauntletError::InvalidContract {
                reason: "equivalent core lexical comparison retains a first mismatch".to_owned(),
            });
        }
        Ok(())
    }
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Mutable run provenance referencing one immutable object hash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunManifest {
    pub schema_version: u32,
    pub run_id: String,
    pub object_hash: String,
    pub provenance: BTreeMap<String, String>,
}

/// Fully encoded paths and bytes, prepared without filesystem mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedArtifact {
    object_hash: String,
    object_path: PathBuf,
    object_bytes: Vec<u8>,
    run_path: PathBuf,
    run_manifest: RunManifest,
    run_manifest_bytes: Vec<u8>,
    run_location: PreparedRunLocation,
    producer_build_identity: GauntletProducerBuildIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PreparedRunLocation {
    Standalone,
    Campaign {
        campaign_run_id: String,
        ordinal: usize,
    },
}

impl PreparedArtifact {
    #[must_use]
    pub fn object_hash(&self) -> &str {
        &self.object_hash
    }

    #[must_use]
    pub fn object_path(&self) -> &Path {
        &self.object_path
    }

    #[must_use]
    pub fn object_bytes(&self) -> &[u8] {
        &self.object_bytes
    }

    #[must_use]
    pub fn run_path(&self) -> &Path {
        &self.run_path
    }

    #[must_use]
    pub const fn run_manifest(&self) -> &RunManifest {
        &self.run_manifest
    }

    #[must_use]
    pub fn run_manifest_bytes(&self) -> &[u8] {
        &self.run_manifest_bytes
    }
}

/// Store rooted at `.gauntlet`, with immutable objects separated from runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactStore {
    root: PathBuf,
}

impl Default for ArtifactStore {
    fn default() -> Self {
        Self::new(".gauntlet")
    }
}

impl ArtifactStore {
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Atomically reserve a campaign run ID before either engine executes.
    ///
    /// A reservation is immutable and single-use, including when the prior
    /// campaign failed before producing per-query artifacts. The campaign
    /// directory itself is the reservation; if marker publication fails, the
    /// empty directory records an aborted reservation and remains single-use.
    /// This prevents stale run references from being mistaken for a retry.
    pub(crate) fn reserve_campaign_run(
        &self,
        run_id: &str,
        manifest_bytes: &[u8],
    ) -> Result<(), GauntletError> {
        validate_run_id(run_id)?;
        if u64::try_from(manifest_bytes.len()).unwrap_or(u64::MAX) > MAX_CAMPAIGN_RESERVATION_BYTES
        {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "campaign reservation exceeds its durable file-size budget".to_owned(),
            });
        }
        let root = PinnedDirectory::ensure_path(&self.root)?;
        let campaigns = root.ensure_child(OsStr::new("campaigns"))?;
        let Some(campaign) = campaigns.create_child_exclusive(OsStr::new(run_id))? else {
            return Err(GauntletError::RunManifestConflict {
                path: self.root.join("campaigns").join(run_id),
            });
        };
        campaign.lock_exclusive()?;
        campaign.publish_no_clobber(OsStr::new("reservation.json"), manifest_bytes)?;
        Ok(())
    }

    /// Encode an admissible standalone object and run manifest without writing
    /// files.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid identities/contracts, a dirty or
    /// unverified producer, unsafe run IDs, or serialization failures.
    pub(crate) fn prepare(
        &self,
        run_id: &str,
        object: &ArtifactObject,
        provenance: BTreeMap<String, String>,
    ) -> Result<PreparedArtifact, GauntletError> {
        let run_path = self.root.join("runs").join(format!("{run_id}.json"));
        let compiled_producer = GauntletProducerBuildIdentity::compiled()?;
        self.prepare_at(
            run_id,
            run_path,
            PreparedRunLocation::Standalone,
            false,
            true,
            &compiled_producer,
            object,
            provenance,
        )
    }

    /// Encode a standalone diagnostic object without claiming admissible
    /// evidence. The compiled producer binding remains exact, while dirty or
    /// explicitly unverified source state stays recordable.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed identity, a fabricated producer binding,
    /// an unsafe run ID, or serialization failure.
    pub fn prepare_diagnostic(
        &self,
        run_id: &str,
        object: &ArtifactObject,
        provenance: BTreeMap<String, String>,
    ) -> Result<PreparedArtifact, GauntletError> {
        let run_path = self.root.join("runs").join(format!("{run_id}.json"));
        let compiled_producer = GauntletProducerBuildIdentity::compiled()?;
        self.prepare_at(
            run_id,
            run_path,
            PreparedRunLocation::Standalone,
            false,
            false,
            &compiled_producer,
            object,
            provenance,
        )
    }

    pub(crate) fn prepare_campaign_case(
        &self,
        campaign_run_id: &str,
        ordinal: usize,
        object: &ArtifactObject,
        provenance: BTreeMap<String, String>,
    ) -> Result<PreparedArtifact, GauntletError> {
        validate_run_id(campaign_run_id)?;
        let run_id = format!("{campaign_run_id}.q{ordinal:06}");
        let run_path = self
            .root
            .join("campaigns")
            .join(campaign_run_id)
            .join("cases")
            .join(format!("q{ordinal:06}.json"));
        let compiled_producer = GauntletProducerBuildIdentity::compiled()?;
        self.prepare_at(
            &run_id,
            run_path,
            PreparedRunLocation::Campaign {
                campaign_run_id: campaign_run_id.to_owned(),
                ordinal,
            },
            true,
            false,
            &compiled_producer,
            object,
            provenance,
        )
    }

    fn prepare_at(
        &self,
        run_id: &str,
        run_path: PathBuf,
        run_location: PreparedRunLocation,
        require_campaign_context: bool,
        require_admissible: bool,
        expected_producer: &GauntletProducerBuildIdentity,
        object: &ArtifactObject,
        provenance: BTreeMap<String, String>,
    ) -> Result<PreparedArtifact, GauntletError> {
        validate_run_id(run_id)?;
        object.validate()?;
        if require_admissible {
            object.producer_build_identity.validate_admissible()?;
        }
        if &object.producer_build_identity != expected_producer {
            return Err(GauntletError::InvalidContract {
                reason: "artifact producer identity does not match the executing compiled producer"
                    .to_owned(),
            });
        }
        if object.campaign.is_some() != require_campaign_context {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "artifact campaign context does not match its run namespace".to_owned(),
            });
        }
        let object_bytes = serialize_json_bounded(
            object,
            MAX_CAMPAIGN_OBJECT_BYTES,
            "artifact object exceeds its durable file-size budget",
        )?;
        let object_hash = hash_object_bytes(&object_bytes, object.object_schema_version)?;
        let run_manifest = RunManifest {
            schema_version: RUN_MANIFEST_SCHEMA_VERSION,
            run_id: run_id.to_owned(),
            object_hash: object_hash.clone(),
            provenance,
        };
        let run_manifest_bytes = serialize_json_bounded(
            &run_manifest,
            MAX_CAMPAIGN_RUN_MANIFEST_BYTES,
            "run manifest exceeds its durable file-size budget",
        )?;
        Ok(PreparedArtifact {
            object_path: self
                .root
                .join("objects")
                .join(format!("{object_hash}.json")),
            run_path,
            object_hash,
            object_bytes,
            run_manifest,
            run_manifest_bytes,
            run_location,
            producer_build_identity: object.producer_build_identity.clone(),
        })
    }

    /// Persist an already prepared object and run reference without overwrites.
    ///
    /// Existing object bytes must match exactly. Existing run IDs must reference
    /// exactly the same manifest. The store never deletes or replaces files.
    ///
    /// # Errors
    ///
    /// Returns I/O, object-collision, or run-conflict errors.
    pub fn persist(&self, prepared: &PreparedArtifact) -> Result<(), GauntletError> {
        self.validate_prepared(prepared)?;
        let root = PinnedDirectory::ensure_path(&self.root)?;
        let objects = root.ensure_child(OsStr::new("objects"))?;

        let (run_directory, run_file_name, _campaign_lock) = match &prepared.run_location {
            PreparedRunLocation::Standalone => {
                let runs = root.ensure_child(OsStr::new("runs"))?;
                let file_name = OsString::from(format!("{}.json", prepared.run_manifest.run_id));
                (runs, file_name, None)
            }
            PreparedRunLocation::Campaign {
                campaign_run_id,
                ordinal,
            } => {
                let campaigns = root.open_child(OsStr::new("campaigns"))?;
                let campaign = campaigns.open_child(OsStr::new(campaign_run_id))?;
                campaign.lock_exclusive()?;
                let _ = campaign.read_regular_bounded(
                    OsStr::new("reservation.json"),
                    MAX_CAMPAIGN_RESERVATION_BYTES,
                )?;
                if campaign.entry_exists(OsStr::new("report.json"))? {
                    return Err(GauntletError::RunManifestConflict {
                        path: self
                            .root
                            .join("campaigns")
                            .join(campaign_run_id)
                            .join("report.json"),
                    });
                }
                let cases = campaign.ensure_child(OsStr::new("cases"))?;
                let file_name = OsString::from(format!("q{ordinal:06}.json"));
                (cases, file_name, Some(campaign))
            }
        };

        objects.write_once_or_verify(
            OsStr::new(&format!("{}.json", prepared.object_hash)),
            &prepared.object_bytes,
            ExistingFileKind::Object,
            MAX_CAMPAIGN_OBJECT_BYTES,
        )?;
        run_directory.write_once_or_verify(
            &run_file_name,
            &prepared.run_manifest_bytes,
            ExistingFileKind::Run,
            MAX_CAMPAIGN_RUN_MANIFEST_BYTES,
        )?;
        Ok(())
    }

    /// Load a completed campaign only after replaying every durable evidence link.
    ///
    /// The report, reservation, case references, immutable objects, comparator
    /// outcomes, divergence classifications, and mismatch aggregates are all
    /// read through pinned directory descriptors and verified before return.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsafe path, incomplete campaign, noncanonical
    /// file, or any provenance/evidence mismatch.
    pub fn load_verified_campaign(&self, run_id: &str) -> Result<CampaignReport, GauntletError> {
        validate_run_id(run_id)?;
        let (root, campaign) = self.open_pinned_campaign(run_id)?;
        let report_bytes =
            campaign.read_regular_bounded(OsStr::new("report.json"), MAX_CAMPAIGN_REPORT_BYTES)?;
        let report: CampaignReport = serde_json::from_slice(&report_bytes)?;
        if matches!(report.schema_version, 3 | 4) {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "legacy campaign report schema lacks the current total lexical contract and is non-admissible; rerun the campaign".to_owned(),
            });
        }
        if report.run_id != run_id {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "completed campaign report has the wrong run ID".to_owned(),
            });
        }
        if !canonical_json_matches(&report, &report_bytes)? {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "completed campaign report is noncanonical".to_owned(),
            });
        }
        drop(report_bytes);
        report.validate_contract()?;
        self.verify_campaign_evidence(&root, &campaign, &report)?;
        Ok(report)
    }

    /// Validate all stored case evidence and atomically publish the sole
    /// campaign-completion marker.
    pub(crate) fn complete_campaign(&self, report: &CampaignReport) -> Result<(), GauntletError> {
        report.validate_creation_environment()?;
        validate_run_id(&report.run_id)?;
        let (root, campaign) = self.open_pinned_campaign(&report.run_id)?;
        self.verify_campaign_evidence(&root, &campaign, report)?;
        let report_bytes = serialize_json_bounded(
            report,
            MAX_CAMPAIGN_REPORT_BYTES,
            "campaign report exceeds its durable file-size budget",
        )?;
        campaign.write_once_or_verify(
            OsStr::new("report.json"),
            &report_bytes,
            ExistingFileKind::Run,
            MAX_CAMPAIGN_REPORT_BYTES,
        )
    }

    fn open_pinned_campaign(
        &self,
        run_id: &str,
    ) -> Result<(PinnedDirectory, PinnedDirectory), GauntletError> {
        let root = PinnedDirectory::open_path(&self.root)?;
        let campaigns = root.open_child(OsStr::new("campaigns"))?;
        let campaign = campaigns.open_child(OsStr::new(run_id))?;
        campaign.lock_exclusive()?;
        Ok((root, campaign))
    }

    fn verify_campaign_evidence(
        &self,
        root: &PinnedDirectory,
        campaign: &PinnedDirectory,
        report: &CampaignReport,
    ) -> Result<(), GauntletError> {
        let reservation_bytes = campaign.read_regular_bounded(
            OsStr::new("reservation.json"),
            MAX_CAMPAIGN_RESERVATION_BYTES,
        )?;
        if reservation_bytes != report.reservation_bytes_unchecked()? {
            return Err(GauntletError::RunManifestConflict {
                path: self
                    .root
                    .join("campaigns")
                    .join(&report.run_id)
                    .join("reservation.json"),
            });
        }
        drop(reservation_bytes);

        let selected = report.selected_queries()?;
        let cases = campaign.open_child_optional(OsStr::new("cases"))?;
        let objects = if report
            .cases
            .iter()
            .any(|result| result.artifact_hash.is_some())
        {
            Some(root.open_child(OsStr::new("objects"))?)
        } else {
            None
        };
        let mut expected_case_names = std::collections::BTreeSet::new();
        let mut campaign_producer_identity: Option<GauntletProducerBuildIdentity> = None;
        let mut evidence = report.begin_evidence_validation()?;
        for (ordinal, (query, result)) in selected.iter().zip(&report.cases).enumerate() {
            let case_name = OsString::from(format!("q{ordinal:06}.json"));
            if result.artifact_hash.is_none() {
                if let Some(cases) = &cases {
                    if cases.entry_exists(&case_name)? {
                        return Err(GauntletError::InvalidPreparedArtifact {
                            reason: "infrastructure-error case has an unexpected run manifest"
                                .to_owned(),
                        });
                    }
                }
                evidence.observe(None)?;
                continue;
            }

            let cases = cases
                .as_ref()
                .ok_or_else(|| GauntletError::InvalidPreparedArtifact {
                    reason: "campaign is missing its case artifact directory".to_owned(),
                })?;
            expected_case_names.insert(case_name.clone());
            let run_bytes =
                cases.read_regular_bounded(&case_name, MAX_CAMPAIGN_RUN_MANIFEST_BYTES)?;
            let run_manifest: RunManifest = serde_json::from_slice(&run_bytes)?;
            let expected_run_id = format!("{}.q{ordinal:06}", report.run_id);
            let expected_provenance = BTreeMap::from([
                ("campaign_run_id".to_owned(), report.run_id.clone()),
                ("query_class".to_owned(), result.query_class.clone()),
                ("query_source".to_owned(), query.source.clone()),
            ]);
            if !canonical_json_matches(&run_manifest, &run_bytes)?
                || run_manifest.schema_version != RUN_MANIFEST_SCHEMA_VERSION
                || run_manifest.run_id != expected_run_id
                || result.artifact_hash.as_deref() != Some(run_manifest.object_hash.as_str())
                || run_manifest.provenance != expected_provenance
            {
                return Err(GauntletError::InvalidPreparedArtifact {
                    reason: "campaign case run manifest does not match the final report".to_owned(),
                });
            }
            drop(run_bytes);

            let object_name = OsString::from(format!("{}.json", run_manifest.object_hash));
            let object_bytes = objects
                .as_ref()
                .ok_or_else(|| GauntletError::InvalidPreparedArtifact {
                    reason: "campaign is missing its artifact object directory".to_owned(),
                })?
                .read_regular_bounded(&object_name, MAX_CAMPAIGN_OBJECT_BYTES)?;
            let object: ArtifactObject = serde_json::from_slice(&object_bytes)?;
            validate_current_object_schema(object.object_schema_version)
                .map_err(|reason| GauntletError::InvalidPreparedArtifact { reason })?;
            let object_hash = hash_object_bytes(&object_bytes, object.object_schema_version)?;
            if !canonical_json_matches(&object, &object_bytes)?
                || object_hash != run_manifest.object_hash
            {
                return Err(GauntletError::InvalidPreparedArtifact {
                    reason: "campaign object bytes or content address are inconsistent".to_owned(),
                });
            }
            if report.config.require_provenance {
                object.validate_admissible()?;
            } else {
                object.validate()?;
            }
            if object.producer_build_identity != report.producer_build_identity {
                return Err(GauntletError::InvalidPreparedArtifact {
                    reason:
                        "campaign case producer identity does not match the reservation and report"
                            .to_owned(),
                });
            }
            if let Some(expected_identity) = &campaign_producer_identity {
                if &object.producer_build_identity != expected_identity {
                    return Err(GauntletError::InvalidPreparedArtifact {
                        reason: "campaign case artifacts were produced by different binaries"
                            .to_owned(),
                    });
                }
            } else {
                campaign_producer_identity = Some(object.producer_build_identity.clone());
            }
            drop(object_bytes);
            evidence.observe(Some((&object, &object_hash)))?;
        }

        if let Some(cases) = &cases {
            let observed_case_names =
                cases.entry_names(expected_case_names.len().saturating_add(1))?;
            if observed_case_names != expected_case_names {
                return Err(GauntletError::InvalidPreparedArtifact {
                    reason: "campaign contains an unexpected case artifact reference".to_owned(),
                });
            }
        } else if !expected_case_names.is_empty() {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "campaign is missing expected case artifact references".to_owned(),
            });
        }

        evidence.finish()?;
        Ok(())
    }

    fn validate_prepared(&self, prepared: &PreparedArtifact) -> Result<(), GauntletError> {
        let object: ArtifactObject = serde_json::from_slice(&prepared.object_bytes)?;
        object.validate()?;
        if object.producer_build_identity != prepared.producer_build_identity {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "prepared artifact producer identity changed after creation".to_owned(),
            });
        }
        if !canonical_json_matches(&object, &prepared.object_bytes)?
            || hash_object_bytes(&prepared.object_bytes, object.object_schema_version)?
                != prepared.object_hash
            || prepared.object_path
                != self
                    .root
                    .join("objects")
                    .join(format!("{}.json", prepared.object_hash))
        {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "object bytes, hash, or store path are inconsistent".to_owned(),
            });
        }
        validate_run_id(&prepared.run_manifest.run_id)?;
        let expected_run_path = match &prepared.run_location {
            PreparedRunLocation::Standalone => self
                .root
                .join("runs")
                .join(format!("{}.json", prepared.run_manifest.run_id)),
            PreparedRunLocation::Campaign {
                campaign_run_id,
                ordinal,
            } => {
                validate_run_id(campaign_run_id)?;
                let expected_run_id = format!("{campaign_run_id}.q{ordinal:06}");
                if prepared.run_manifest.run_id != expected_run_id {
                    return Err(GauntletError::InvalidPreparedArtifact {
                        reason: "campaign run manifest ID is inconsistent".to_owned(),
                    });
                }
                self.root
                    .join("campaigns")
                    .join(campaign_run_id)
                    .join("cases")
                    .join(format!("q{ordinal:06}.json"))
            }
        };
        if prepared.run_manifest.schema_version != RUN_MANIFEST_SCHEMA_VERSION
            || prepared.run_manifest.object_hash != prepared.object_hash
            || !canonical_json_matches(&prepared.run_manifest, &prepared.run_manifest_bytes)?
            || prepared.run_path != expected_run_path
        {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "run manifest bytes, object reference, or store path are inconsistent"
                    .to_owned(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum ExistingFileKind {
    Object,
    Run,
}

fn hash_object_bytes(bytes: &[u8], schema_version: u32) -> Result<String, GauntletError> {
    match schema_version {
        1 | 2 => {
            let domain = if schema_version == 1 {
                HASH_DOMAIN_V1
            } else {
                HASH_DOMAIN_V2
            };
            let mut hasher = Xxh3::new();
            hasher.update(domain);
            hasher.update(bytes);
            Ok(format!("{:016x}", hasher.digest()))
        }
        3 => {
            let mut hasher = Sha256::new();
            hasher.update(HASH_DOMAIN_V3);
            hasher.update(bytes);
            Ok(lower_hex(&hasher.finalize()))
        }
        5 => {
            let mut hasher = Sha256::new();
            hasher.update(HASH_DOMAIN_V5);
            hasher.update(bytes);
            Ok(lower_hex(&hasher.finalize()))
        }
        6 => {
            let mut hasher = Sha256::new();
            hasher.update(HASH_DOMAIN_V6);
            hasher.update(bytes);
            Ok(lower_hex(&hasher.finalize()))
        }
        _ => Err(GauntletError::InvalidContract {
            reason: format!(
                "artifact object schema version {schema_version} has no registered hash domain"
            ),
        }),
    }
}

fn validate_current_object_schema(schema_version: u32) -> Result<(), String> {
    match schema_version {
        OBJECT_SCHEMA_VERSION => Ok(()),
        1 | 2 | 3 => Err(format!(
            "legacy artifact v{schema_version} lacks the current total lexical contract and is non-admissible; rerun the campaign"
        )),
        4 => Err(
            "reserved pre-policy artifact v4 is non-admissible; rerun under the complete current evidence contract"
                .to_owned(),
        ),
        5 => Err(
            "pre-run-identity artifact v5 is non-admissible; rerun with producer identity captured before observation"
                .to_owned(),
        ),
        _ => Err(format!(
            "artifact object schema version {schema_version} is unsupported and non-admissible; rerun under the complete current evidence contract"
        )),
    }
}

fn lower_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut output = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        write!(output, "{byte:02x}").expect("writing to String is infallible");
    }
    output
}

struct BoundedJsonWriter {
    bytes: Vec<u8>,
    max_bytes: usize,
    limit_exceeded: bool,
}

impl Write for BoundedJsonWriter {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        let Some(new_len) = self.bytes.len().checked_add(buffer.len()) else {
            self.limit_exceeded = true;
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "canonical JSON length overflowed",
            ));
        };
        if new_len > self.max_bytes {
            self.limit_exceeded = true;
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "canonical JSON exceeds its byte budget",
            ));
        }
        self.bytes.try_reserve(buffer.len()).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                format!("unable to reserve bounded canonical JSON: {error}"),
            )
        })?;
        self.bytes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn serialize_json_bounded<T: Serialize>(
    value: &T,
    max_bytes: u64,
    limit_reason: &str,
) -> Result<Vec<u8>, GauntletError> {
    let max_bytes =
        usize::try_from(max_bytes).map_err(|_| GauntletError::InvalidPreparedArtifact {
            reason: "durable JSON byte budget does not fit this platform".to_owned(),
        })?;
    let mut writer = BoundedJsonWriter {
        bytes: Vec::new(),
        max_bytes,
        limit_exceeded: false,
    };
    let result = serde_json::to_writer(&mut writer, value);
    if writer.limit_exceeded {
        return Err(GauntletError::InvalidPreparedArtifact {
            reason: limit_reason.to_owned(),
        });
    }
    result?;
    Ok(writer.bytes)
}

struct CanonicalJsonMatcher<'a> {
    expected: &'a [u8],
    offset: usize,
    matches: bool,
}

impl Write for CanonicalJsonMatcher<'_> {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        let Some(end) = self.offset.checked_add(buffer.len()) else {
            self.matches = false;
            self.offset = usize::MAX;
            return Ok(buffer.len());
        };
        if end > self.expected.len() || (self.matches && self.expected[self.offset..end] != *buffer)
        {
            self.matches = false;
        }
        self.offset = end;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn canonical_json_matches<T: Serialize>(value: &T, expected: &[u8]) -> Result<bool, GauntletError> {
    let mut matcher = CanonicalJsonMatcher {
        expected,
        offset: 0,
        matches: true,
    };
    serde_json::to_writer(&mut matcher, value)?;
    Ok(matcher.matches && matcher.offset == expected.len())
}

#[cfg(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos"
))]
pub struct PinnedDirectory {
    file: File,
    display_path: PathBuf,
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos"
)))]
pub struct PinnedDirectory;

#[cfg(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos"
))]
impl PinnedDirectory {
    fn open_path(path: &Path) -> Result<Self, GauntletError> {
        Self::walk_path(path, false)
    }

    pub(crate) fn ensure_path(path: &Path) -> Result<Self, GauntletError> {
        Self::walk_path(path, true)
    }

    fn walk_path(path: &Path, ensure_components: bool) -> Result<Self, GauntletError> {
        use rustix::fs::{Mode, open};

        if path.as_os_str().is_empty() {
            return Err(GauntletError::UnsafeStorePath {
                path: path.to_path_buf(),
            });
        }
        let mut names = Vec::<OsString>::new();
        for component in path.components() {
            match component {
                std::path::Component::RootDir | std::path::Component::CurDir => {}
                std::path::Component::Normal(name) => names.push(name.to_owned()),
                std::path::Component::ParentDir | std::path::Component::Prefix(_) => {
                    return Err(GauntletError::UnsafeStorePath {
                        path: path.to_path_buf(),
                    });
                }
            }
        }
        let base = if path.is_absolute() {
            Path::new("/")
        } else {
            Path::new(".")
        };
        let descriptor =
            open(base, directory_open_flags(), Mode::empty()).map_err(std::io::Error::from)?;
        let mut current = Self {
            file: File::from(descriptor),
            display_path: base.to_path_buf(),
        };
        for name in &names {
            current = if ensure_components {
                current.ensure_child(name)?
            } else {
                current.open_child(name)?
            };
        }
        current.display_path = path.to_path_buf();
        Ok(current)
    }

    fn open_child(&self, name: &OsStr) -> Result<Self, GauntletError> {
        self.open_child_optional(name)?.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "directory does not exist: {}",
                    self.display_path.join(name).display()
                ),
            )
            .into()
        })
    }

    fn open_child_optional(&self, name: &OsStr) -> Result<Option<Self>, GauntletError> {
        use rustix::fs::{Mode, openat};
        use rustix::io::Errno;

        validate_child_name(&self.display_path, name)?;
        match openat(&self.file, name, directory_open_flags(), Mode::empty()) {
            Ok(descriptor) => Ok(Some(Self {
                file: File::from(descriptor),
                display_path: self.display_path.join(name),
            })),
            Err(Errno::NOENT) => Ok(None),
            Err(Errno::LOOP | Errno::NOTDIR) => Err(GauntletError::UnsafeStorePath {
                path: self.display_path.join(name),
            }),
            Err(error) => Err(std::io::Error::from(error).into()),
        }
    }

    fn ensure_child(&self, name: &OsStr) -> Result<Self, GauntletError> {
        use rustix::fs::{Mode, mkdirat};
        use rustix::io::Errno;

        if let Some(child) = self.open_child_optional(name)? {
            return Ok(child);
        }
        match mkdirat(&self.file, name, Mode::RWXU | Mode::RWXG | Mode::RWXO) {
            Ok(()) => self.file.sync_all()?,
            Err(Errno::EXIST) => {}
            Err(error) => return Err(std::io::Error::from(error).into()),
        }
        self.open_child(name)
    }

    fn create_child_exclusive(&self, name: &OsStr) -> Result<Option<Self>, GauntletError> {
        use rustix::fs::{Mode, mkdirat};
        use rustix::io::Errno;

        validate_child_name(&self.display_path, name)?;
        match mkdirat(&self.file, name, Mode::RWXU | Mode::RWXG | Mode::RWXO) {
            Ok(()) => {
                self.file.sync_all()?;
                self.open_child(name).map(Some)
            }
            Err(Errno::EXIST) => Ok(None),
            Err(error) => Err(std::io::Error::from(error).into()),
        }
    }

    fn lock_exclusive(&self) -> Result<(), GauntletError> {
        use rustix::fs::{FlockOperation, flock};

        flock(&self.file, FlockOperation::LockExclusive)
            .map_err(std::io::Error::from)
            .map_err(Into::into)
    }

    fn entry_exists(&self, name: &OsStr) -> Result<bool, GauntletError> {
        use rustix::fs::{AtFlags, statat};
        use rustix::io::Errno;

        validate_child_name(&self.display_path, name)?;
        match statat(&self.file, name, AtFlags::SYMLINK_NOFOLLOW) {
            Ok(_) => Ok(true),
            Err(Errno::NOENT) => Ok(false),
            Err(error) => Err(std::io::Error::from(error).into()),
        }
    }

    pub(crate) fn read_regular_bounded(
        &self,
        name: &OsStr,
        max_bytes: u64,
    ) -> Result<Vec<u8>, GauntletError> {
        use rustix::fs::{FileType, Mode, OFlags, fstat, openat};

        validate_child_name(&self.display_path, name)?;
        let descriptor = openat(
            &self.file,
            name,
            OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::empty(),
        )
        .map_err(std::io::Error::from)?;
        let stat = fstat(&descriptor).map_err(std::io::Error::from)?;
        let size = u64::try_from(stat.st_size).unwrap_or(u64::MAX);
        if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile || size > max_bytes {
            return Err(GauntletError::UnsafeStorePath {
                path: self.display_path.join(name),
            });
        }
        let file = File::from(descriptor);
        let capacity = usize::try_from(size).map_err(|_| GauntletError::UnsafeStorePath {
            path: self.display_path.join(name),
        })?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(capacity).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                format!("unable to reserve bounded artifact read: {error}"),
            )
        })?;
        file.take(max_bytes.saturating_add(1))
            .read_to_end(&mut bytes)?;
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > max_bytes {
            return Err(GauntletError::UnsafeStorePath {
                path: self.display_path.join(name),
            });
        }
        Ok(bytes)
    }

    pub(crate) fn entry_names(
        &self,
        max_entries: usize,
    ) -> Result<std::collections::BTreeSet<OsString>, GauntletError> {
        use rustix::fs::Dir;
        use std::os::unix::ffi::OsStrExt as _;

        let mut names = std::collections::BTreeSet::new();
        for entry in Dir::read_from(&self.file).map_err(std::io::Error::from)? {
            let entry = entry.map_err(std::io::Error::from)?;
            let bytes = entry.file_name().to_bytes();
            if bytes == b"." || bytes == b".." {
                continue;
            }
            names.insert(OsStr::from_bytes(bytes).to_owned());
            if names.len() > max_entries {
                return Err(GauntletError::InvalidPreparedArtifact {
                    reason: "campaign case directory exceeds its bounded evidence set".to_owned(),
                });
            }
        }
        Ok(names)
    }

    fn publish_no_clobber(&self, name: &OsStr, bytes: &[u8]) -> Result<(), GauntletError> {
        self.publish_no_clobber_io(name, bytes).map_err(Into::into)
    }

    #[cfg(all(test, feature = "tantivy-oracle", feature = "pruning-conformance"))]
    pub(crate) fn publish_unique_no_clobber(
        &self,
        temporary_name: &OsStr,
        target_name: &OsStr,
        bytes: &[u8],
    ) -> Result<(), GauntletError> {
        use rustix::fs::{
            FlockOperation, Mode, OFlags, RenameFlags, flock, fstat, openat, renameat_with,
        };

        validate_child_name(&self.display_path, temporary_name)?;
        validate_child_name(&self.display_path, target_name)?;
        flock(&self.file, FlockOperation::LockExclusive).map_err(std::io::Error::from)?;
        if self.entry_exists(target_name)? {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "artifact already exists",
            )
            .into());
        }
        let descriptor = openat(
            &self.file,
            temporary_name,
            OFlags::RDWR
                | OFlags::CREATE
                | OFlags::EXCL
                | OFlags::CLOEXEC
                | OFlags::NOFOLLOW
                | OFlags::NONBLOCK,
            Mode::RUSR | Mode::WUSR,
        )
        .map_err(std::io::Error::from)?;
        let stat = fstat(&descriptor).map_err(std::io::Error::from)?;
        if rustix::fs::FileType::from_raw_mode(stat.st_mode) != rustix::fs::FileType::RegularFile
            || stat.st_size != 0
        {
            return Err(GauntletError::UnsafeStorePath {
                path: self.display_path.join(temporary_name),
            });
        }
        let mut temporary = File::from(descriptor);
        temporary.write_all(bytes)?;
        temporary.sync_all()?;
        renameat_with(
            &self.file,
            temporary_name,
            &self.file,
            target_name,
            RenameFlags::NOREPLACE,
        )
        .map_err(std::io::Error::from)?;
        self.file.sync_all()?;
        Ok(())
    }

    fn write_once_or_verify(
        &self,
        name: &OsStr,
        bytes: &[u8],
        kind: ExistingFileKind,
        max_bytes: u64,
    ) -> Result<(), GauntletError> {
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > max_bytes {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "artifact exceeds its durable file-size budget".to_owned(),
            });
        }
        match self.publish_no_clobber_io(name, bytes) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                let comparison_limit = u64::try_from(bytes.len())
                    .unwrap_or(u64::MAX)
                    .saturating_add(1)
                    .min(max_bytes);
                let existing = self.read_regular_bounded(name, comparison_limit)?;
                if existing == bytes {
                    self.file.sync_all()?;
                    Ok(())
                } else {
                    Err(match kind {
                        ExistingFileKind::Object => GauntletError::ArtifactCollision {
                            path: self.display_path.join(name),
                        },
                        ExistingFileKind::Run => GauntletError::RunManifestConflict {
                            path: self.display_path.join(name),
                        },
                    })
                }
            }
            Err(error) => Err(error.into()),
        }
    }

    fn publish_no_clobber_io(&self, name: &OsStr, bytes: &[u8]) -> std::io::Result<()> {
        use rustix::fs::{
            FlockOperation, Mode, OFlags, RenameFlags, flock, fstat, openat, renameat_with,
        };

        validate_child_name_io(name)?;
        flock(&self.file, FlockOperation::LockExclusive).map_err(std::io::Error::from)?;
        if self
            .entry_exists(name)
            .map_err(|error| gauntlet_to_io(&error))?
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "artifact already exists",
            ));
        }
        let mut pending_name = OsString::from(".");
        pending_name.push(name);
        pending_name.push(".pending");
        let temporary = openat(
            &self.file,
            &pending_name,
            OFlags::RDWR | OFlags::CREATE | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::RUSR | Mode::WUSR,
        )
        .map_err(std::io::Error::from)?;
        let stat = fstat(&temporary).map_err(std::io::Error::from)?;
        let staged_size = u64::try_from(stat.st_size).unwrap_or(u64::MAX);
        if rustix::fs::FileType::from_raw_mode(stat.st_mode) != rustix::fs::FileType::RegularFile
            || staged_size > u64::try_from(bytes.len()).unwrap_or(u64::MAX)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "staged artifact exceeds the canonical bytes",
            ));
        }
        let mut temporary = File::from(temporary);
        temporary.seek(SeekFrom::Start(0))?;
        let capacity = usize::try_from(staged_size).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "staged artifact length cannot fit in memory",
            )
        })?;
        let mut existing = Vec::new();
        existing.try_reserve_exact(capacity).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                format!("unable to reserve bounded staged-artifact read: {error}"),
            )
        })?;
        (&mut temporary)
            .take(
                u64::try_from(bytes.len())
                    .unwrap_or(u64::MAX)
                    .saturating_add(1),
            )
            .read_to_end(&mut existing)?;
        if existing.len() > bytes.len() || !bytes.starts_with(&existing) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "staged artifact is not a prefix of the canonical bytes",
            ));
        }
        temporary.seek(SeekFrom::End(0))?;
        temporary.write_all(&bytes[existing.len()..])?;
        temporary.sync_all()?;
        renameat_with(
            &self.file,
            &pending_name,
            &self.file,
            name,
            RenameFlags::NOREPLACE,
        )
        .map_err(std::io::Error::from)?;
        self.file.sync_all()
    }
}

#[cfg(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos"
))]
fn directory_open_flags() -> rustix::fs::OFlags {
    rustix::fs::OFlags::RDONLY
        | rustix::fs::OFlags::CLOEXEC
        | rustix::fs::OFlags::NOFOLLOW
        | rustix::fs::OFlags::DIRECTORY
        | rustix::fs::OFlags::NONBLOCK
}

fn validate_child_name(parent: &Path, name: &OsStr) -> Result<(), GauntletError> {
    validate_child_name_io(name).map_err(|_| GauntletError::UnsafeStorePath {
        path: parent.join(name),
    })
}

fn validate_child_name_io(name: &OsStr) -> std::io::Result<()> {
    let mut components = Path::new(name).components();
    if matches!(components.next(), Some(std::path::Component::Normal(_)))
        && components.next().is_none()
    {
        Ok(())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "artifact child name is not one safe path component",
        ))
    }
}

fn gauntlet_to_io(error: &GauntletError) -> std::io::Error {
    std::io::Error::other(error.to_string())
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos"
)))]
impl PinnedDirectory {
    fn unsupported<T>() -> Result<T, GauntletError> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "descriptor-relative artifact storage is unsupported on this platform",
        )
        .into())
    }

    fn open_path(_path: &Path) -> Result<Self, GauntletError> {
        Self::unsupported()
    }

    pub(crate) fn ensure_path(_path: &Path) -> Result<Self, GauntletError> {
        Self::unsupported()
    }

    fn open_child(&self, _name: &OsStr) -> Result<Self, GauntletError> {
        Self::unsupported()
    }

    fn open_child_optional(&self, _name: &OsStr) -> Result<Option<Self>, GauntletError> {
        Self::unsupported()
    }

    fn ensure_child(&self, _name: &OsStr) -> Result<Self, GauntletError> {
        Self::unsupported()
    }

    fn create_child_exclusive(&self, _name: &OsStr) -> Result<Option<Self>, GauntletError> {
        Self::unsupported()
    }

    fn lock_exclusive(&self) -> Result<(), GauntletError> {
        Self::unsupported()
    }

    fn entry_exists(&self, _name: &OsStr) -> Result<bool, GauntletError> {
        Self::unsupported()
    }

    pub(crate) fn read_regular_bounded(
        &self,
        _name: &OsStr,
        _max_bytes: u64,
    ) -> Result<Vec<u8>, GauntletError> {
        Self::unsupported()
    }

    pub(crate) fn entry_names(
        &self,
        _max_entries: usize,
    ) -> Result<std::collections::BTreeSet<OsString>, GauntletError> {
        Self::unsupported()
    }

    fn publish_no_clobber(&self, _name: &OsStr, _bytes: &[u8]) -> Result<(), GauntletError> {
        Self::unsupported()
    }

    #[cfg(all(test, feature = "tantivy-oracle", feature = "pruning-conformance"))]
    pub(crate) fn publish_unique_no_clobber(
        &self,
        _temporary_name: &OsStr,
        _target_name: &OsStr,
        _bytes: &[u8],
    ) -> Result<(), GauntletError> {
        Self::unsupported()
    }

    fn write_once_or_verify(
        &self,
        _name: &OsStr,
        _bytes: &[u8],
        _kind: ExistingFileKind,
        _max_bytes: u64,
    ) -> Result<(), GauntletError> {
        Self::unsupported()
    }
}

fn validate_run_id(run_id: &str) -> Result<(), GauntletError> {
    let safe = !run_id.is_empty()
        && run_id.len() <= 128
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::TANTIVY_ORACLE_CONFIG_HASH;
    use crate::{
        ComparisonMode, CountState, DifferentialCase, EngineDescriptor, EngineFamily,
        EngineObservation, NativeTieKey, RankedHit, compare_observations,
    };

    fn representative_observation(
        hits: Vec<RankedHit>,
        snippets: BTreeMap<String, String>,
    ) -> EngineObservation {
        EngineObservation {
            hits,
            cutoff_tie_group: Vec::new(),
            cutoff_tie_complete: true,
            offset_tie_group: Vec::new(),
            offset_tie_complete: false,
            snippets,
            match_count: CountState::Value(2),
            doc_count: 2,
            ast_differences: Vec::new(),
        }
    }

    fn sample_object() -> ArtifactObject {
        let oracle_version = oracle_version_contract().expect("version contract");
        let producer_build_identity =
            GauntletProducerBuildIdentity::compiled().expect("compiled producer identity");
        let producer_revision = producer_build_identity.source_git_revision.clone();
        let producer_dirty = producer_build_identity.source_git_dirty;
        let subject = EngineDescriptor {
            family: EngineFamily::Quill,
            implementation: "quill-stub".to_owned(),
            crate_version: "0.2.1".to_owned(),
            source_revision: producer_revision.clone(),
            source_dirty: producer_dirty,
            config_hash: "01".to_owned(),
        };
        let oracle = EngineDescriptor {
            family: EngineFamily::Tantivy,
            implementation: "frankensearch-lexical/tantivy-index".to_owned(),
            crate_version: oracle_version.lexical_package_version.clone(),
            source_revision: producer_revision,
            source_dirty: producer_dirty,
            config_hash: TANTIVY_ORACLE_CONFIG_HASH.to_owned(),
        };
        let subject_observation = representative_observation(
            vec![
                RankedHit {
                    doc_id: "β/~second".to_owned(),
                    score_bits: 4.0_f32.to_bits(),
                    native_tie_key: NativeTieKey::QuillDocId { doc_id: 1 },
                },
                RankedHit {
                    doc_id: "α-first".to_owned(),
                    score_bits: 4.0_f32.to_bits(),
                    native_tie_key: NativeTieKey::QuillDocId { doc_id: 2 },
                },
            ],
            BTreeMap::from([
                ("α-first".to_owned(), "<b>α</b> body".to_owned()),
                ("β/~second".to_owned(), "<b>β</b> body".to_owned()),
            ]),
        );
        let oracle_observation = representative_observation(
            vec![
                RankedHit {
                    doc_id: "α-first".to_owned(),
                    score_bits: 4.0_f32.to_bits(),
                    native_tie_key: NativeTieKey::TantivyDocAddress {
                        segment_ord: 3,
                        doc_id: 8,
                    },
                },
                RankedHit {
                    doc_id: "β/~second".to_owned(),
                    score_bits: 4.0_f32.to_bits(),
                    native_tie_key: NativeTieKey::TantivyDocAddress {
                        segment_ord: 3,
                        doc_id: 9,
                    },
                },
            ],
            BTreeMap::from([
                ("β/~second".to_owned(), "<b>β</b> body".to_owned()),
                ("α-first".to_owned(), "<b>α</b> body".to_owned()),
            ]),
        );
        let comparator_config = ComparatorConfig::default();
        let comparison =
            compare_observations(subject_observation, oracle_observation, comparator_config)
                .expect("representative comparison");
        let mut case = DifferentialCase::new("artifact-β/~smoke", "rust β", 2);
        case.metadata.generator_id = Some("quill-generator-v1".to_owned());
        case.metadata.generator_seed = Some(42);
        case.metadata.corpus_hash = Some("0123456789abcdef".to_owned());
        ArtifactObject {
            object_schema_version: OBJECT_SCHEMA_VERSION,
            canonicalization_version: CANONICALIZATION_VERSION,
            oracle_version,
            producer_build_identity,
            engines: EnginePairIdentity::new(ComparisonMode::CrossEngine, subject, oracle)
                .expect("distinct engines"),
            case,
            comparator_config,
            comparison,
            lexical_contract: ArtifactLexicalContractEvidence::RankEnvelopeOnly,
            campaign: None,
        }
    }

    fn sample_campaign_object() -> ArtifactObject {
        let mut object = sample_object();
        let semantic_contract = SemanticContract::shipping_default();
        object
            .engines
            .bind_semantic_contract(semantic_contract.clone())
            .expect("bind semantics");
        let corpus_manifest_hash = "a".repeat(64);
        object.case.metadata = crate::DifferentialCaseMetadata {
            generator_id: None,
            generator_seed: None,
            corpus_hash: Some(corpus_manifest_hash.clone()),
        };
        object.campaign = Some(CampaignArtifactContext {
            corpus_manifest_hash,
            query_manifest_hash: "b".repeat(64),
            query_suite_source: QuerySuiteSource::ExplicitCases,
            query_source_identity_sha256: "c".repeat(64),
            semantic_contract,
            contract_mode: CampaignContractMode::RankEnvelopeOnly,
            query_seed: 42,
            query: GeneratedQueryCase {
                id: object.case.fixture_id.clone(),
                syntax: crate::QuerySyntax::Default,
                query_kind: crate::GeneratedQueryKind::Term,
                query: object.case.query.clone(),
                limit: object.case.limit,
                offset: object.case.offset,
                count_requested: object.case.count_requested,
                filters: crate::GeneratedQueryFilters::default(),
                expected_divergence: None,
                source: "artifact-unit-test".to_owned(),
            },
            registered_divergence: None,
        });
        object
    }

    fn golden_producer_build_identity() -> GauntletProducerBuildIdentity {
        let mut feature_hasher = Sha256::new();
        feature_hasher.update([]);
        GauntletProducerBuildIdentity {
            schema_version: PRODUCER_BUILD_IDENTITY_SCHEMA_VERSION,
            producer_contract_version: PRODUCER_CONTRACT_VERSION_V4.to_owned(),
            source_git_revision: "a".repeat(40),
            source_git_dirty: false,
            source_verification: GauntletProducerSourceVerification::GitCheckoutVerified,
            cargo_lock_sha256: "b".repeat(64),
            rustc_version_verbose_hex: "72757374632d5656".to_owned(),
            target_triple: "x86_64-unknown-linux-gnu".to_owned(),
            cargo_profile: "test".to_owned(),
            enabled_features: Vec::new(),
            enabled_features_sha256: lower_hex(&feature_hasher.finalize()),
            executable_sha256: "c".repeat(64),
            executable_byte_len: 12_345,
        }
    }

    fn bind_producer_identity(
        object: &mut ArtifactObject,
        identity: GauntletProducerBuildIdentity,
    ) {
        object.engines.subject.source_revision = identity.source_git_revision.clone();
        object.engines.subject.source_dirty = identity.source_git_dirty;
        object.engines.oracle.source_revision = identity.source_git_revision.clone();
        object.engines.oracle.source_dirty = identity.source_git_dirty;
        object.producer_build_identity = identity;
    }

    #[test]
    fn run_ids_reference_one_immutable_object() {
        let object = sample_object();
        let store = ArtifactStore::default();
        let first = store
            .prepare_diagnostic("run-one", &object, BTreeMap::new())
            .expect("first preparation");
        let second = store
            .prepare_diagnostic("run-two", &object, BTreeMap::new())
            .expect("second preparation");

        assert_eq!(first.object_hash, second.object_hash);
        assert_eq!(first.object_bytes, second.object_bytes);
        assert_ne!(first.run_manifest_bytes, second.run_manifest_bytes);
        assert_eq!(
            first.object_path,
            Path::new(".gauntlet")
                .join("objects")
                .join(format!("{}.json", first.object_hash))
        );
    }

    #[test]
    fn canonical_bytes_and_hash_are_repeatable() {
        let mut first = sample_object();
        first
            .case
            .metadata
            .generator_id
            .clone_from(&Some("quill-generator-v1".to_owned()));
        let second = first.clone();
        assert_eq!(
            first.canonical_bytes().unwrap(),
            second.canonical_bytes().unwrap()
        );
        assert_eq!(first.object_hash().unwrap(), second.object_hash().unwrap());
    }

    #[test]
    fn prepare_rejects_oracle_descriptor_outside_compiled_producer() {
        let mut object = sample_object();
        object.engines.oracle.source_revision = "f".repeat(40);
        assert!(
            ArtifactStore::default()
                .prepare("bad-oracle-producer", &object, BTreeMap::new())
                .is_err()
        );
    }

    #[test]
    fn prepare_rejects_two_descriptors_that_agree_on_a_fabricated_build_identity() {
        let mut object = sample_object();
        bind_producer_identity(&mut object, golden_producer_build_identity());
        object
            .validate()
            .expect("fabricated build is structurally self-consistent diagnostic data");
        let error = ArtifactStore::default()
            .prepare("fabricated-producer", &object, BTreeMap::new())
            .expect_err("preparation must compare against the compiled producer identity");
        assert!(matches!(
            error,
            GauntletError::InvalidContract { ref reason }
                if reason.contains("does not match the executing compiled producer")
        ));
    }

    #[test]
    fn dirty_or_unverified_producers_remain_diagnostic_but_are_not_admissible() {
        let mut dirty = sample_object();
        dirty.producer_build_identity.source_git_dirty = true;
        dirty.engines.subject.source_dirty = true;
        dirty.engines.oracle.source_dirty = true;
        dirty
            .validate()
            .expect("truthful dirty producer identity remains structurally recordable");
        assert!(matches!(
            dirty.validate_admissible(),
            Err(GauntletError::InvalidContract { ref reason })
                if reason.contains("clean Git-verified producer")
        ));

        let mut unverified = sample_object();
        unverified.producer_build_identity.source_git_revision = "d".repeat(40);
        unverified.engines.subject.source_revision = "d".repeat(40);
        unverified.engines.oracle.source_revision = "d".repeat(40);
        unverified.producer_build_identity.source_verification =
            GauntletProducerSourceVerification::ExplicitUnverified;
        unverified
            .validate()
            .expect("truthful unverified producer identity remains structurally recordable");
        assert!(matches!(
            unverified.validate_admissible(),
            Err(GauntletError::InvalidContract { ref reason })
                if reason.contains("clean Git-verified producer")
        ));
    }

    #[test]
    fn admissible_preparation_accepts_one_clean_verified_sealed_identity() {
        let store = ArtifactStore::default();
        let mut object = sample_object();
        let producer_identity = golden_producer_build_identity();
        bind_producer_identity(&mut object, producer_identity.clone());
        let prepared = store
            .prepare_at(
                "admissible-production",
                store.root().join("runs/admissible-production.json"),
                PreparedRunLocation::Standalone,
                false,
                true,
                &producer_identity,
                &object,
                BTreeMap::new(),
            )
            .expect("clean Git-verified exact producer identity is admissible");
        assert_eq!(prepared.producer_build_identity, producer_identity);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn explicit_unverified_diagnostic_can_persist_but_not_use_admissible_prepare() {
        let temp = tempfile::tempdir().expect("temporary parent");
        let store = ArtifactStore::new(temp.path().join("gauntlet"));
        let mut object = sample_object();
        let mut diagnostic_identity = object.producer_build_identity.clone();
        diagnostic_identity.source_git_revision = "d".repeat(40);
        diagnostic_identity.source_verification =
            GauntletProducerSourceVerification::ExplicitUnverified;
        bind_producer_identity(&mut object, diagnostic_identity.clone());

        let admissible_error = store
            .prepare("unverified-production", &object, BTreeMap::new())
            .expect_err("unverified build must not enter the admissible standalone namespace");
        assert!(matches!(
            admissible_error,
            GauntletError::InvalidContract { ref reason }
                if reason.contains("clean Git-verified producer")
        ));

        let run_path = store.root().join("runs/unverified-diagnostic.json");
        let prepared = store
            .prepare_at(
                "unverified-diagnostic",
                run_path,
                PreparedRunLocation::Standalone,
                false,
                false,
                &diagnostic_identity,
                &object,
                BTreeMap::new(),
            )
            .expect("explicit diagnostic preparation remains available");
        store
            .persist(&prepared)
            .expect("explicit diagnostic publication remains durable");
        assert!(prepared.object_path().is_file());
        assert!(prepared.run_path().is_file());
    }

    #[test]
    fn prepare_rejects_build_identity_field_tampering() {
        let mut object = sample_object();
        object.producer_build_identity.cargo_lock_sha256 = "c".repeat(64);
        object
            .validate()
            .expect("tampered lock hash remains structurally canonical");
        assert!(
            ArtifactStore::default()
                .prepare_diagnostic("tampered-build", &object, BTreeMap::new())
                .is_err()
        );
    }

    #[test]
    fn prepare_rejects_tampered_independent_oracle_contract_pin() {
        let mut object = sample_object();
        object.oracle_version.lexical_git_revision = "f".repeat(40);
        assert!(
            ArtifactStore::default()
                .prepare("bad-oracle-contract", &object, BTreeMap::new())
                .is_err()
        );
    }

    #[test]
    fn prepare_rejects_unpinned_oracle_configuration() {
        let mut object = sample_object();
        object.engines.oracle.config_hash = "different-schema".to_owned();
        assert!(
            ArtifactStore::default()
                .prepare("bad-oracle-config", &object, BTreeMap::new())
                .is_err()
        );
    }

    #[test]
    fn prepare_rejects_fabricated_exact_report() {
        let mut object = sample_object();
        object.comparison.status = crate::ComparisonStatus::Exact;
        object.comparison.rank_class = crate::RankClass::RankExact;
        object.comparison.divergences.clear();
        object.comparison.first_divergence = None;
        assert!(
            ArtifactStore::default()
                .prepare("forged-report", &object, BTreeMap::new())
                .is_err()
        );
    }

    #[test]
    fn committed_v1_object_decodes_but_is_nonadmissible_and_uses_raw_byte_addressing() {
        const LEGACY_V1_BYTES: &[u8] = include_bytes!("../fixtures/artifact-object-v1.json");
        let object: ArtifactObject =
            serde_json::from_slice(LEGACY_V1_BYTES).expect("decode committed v1 object");
        assert_eq!(object.object_schema_version, 1);

        let original_byte_hash =
            hash_object_bytes(LEGACY_V1_BYTES, 1).expect("hash original v1 bytes");
        assert_eq!(original_byte_hash.len(), 16);
        assert_ne!(
            original_byte_hash,
            object
                .object_hash()
                .expect("decoded DTO remains diagnosable"),
            "a decoded legacy DTO must not masquerade as an original-byte verifier"
        );

        let error = object
            .validate()
            .expect_err("committed v1 object must require a campaign rerun");
        assert!(matches!(
            error,
            GauntletError::InvalidContract { ref reason }
                if reason.contains("legacy artifact")
                    && reason.contains("non-admissible")
                    && reason.contains("rerun")
        ));
    }

    #[test]
    fn synthetic_v2_object_is_decode_only_and_never_admissible() {
        let mut object = sample_object();
        object.object_schema_version = 2;
        assert_eq!(
            object
                .object_hash()
                .expect("registered v2 address remains diagnosable")
                .len(),
            16
        );
        let error = object
            .validate()
            .expect_err("pre-v3 object must require a campaign rerun");
        assert!(matches!(
            error,
            GauntletError::InvalidContract { ref reason }
                if reason.contains("legacy artifact")
                    && reason.contains("non-admissible")
                    && reason.contains("rerun")
        ));
    }

    #[test]
    fn committed_v3_object_is_decode_only_and_cannot_reinterpret_the_contract_pin_as_source() {
        const LEGACY_V3_BYTES: &[u8] = include_bytes!("../fixtures/artifact-object-v3.json");
        let object: ArtifactObject =
            serde_json::from_slice(LEGACY_V3_BYTES).expect("decode committed v3 object");
        assert_eq!(object.object_schema_version, 3);
        assert_eq!(
            hash_object_bytes(LEGACY_V3_BYTES, 3).expect("registered v3 diagnostic address"),
            "6a1e822aa157626b7abd1b7f7b496b7c44c4006928376ae23653137e88227506"
        );
        let error = object
            .validate()
            .expect_err("v3 producer provenance must require a campaign rerun");
        assert!(matches!(
            error,
            GauntletError::InvalidContract { ref reason }
                if reason.contains("legacy artifact")
                    && reason.contains("non-admissible")
                    && reason.contains("rerun")
        ));
    }

    #[test]
    fn reserved_v4_pre_policy_fixture_is_not_admissible_as_current_evidence() {
        // Salej's diagnostic proof bundle is deliberately independent from the
        // generic ArtifactStore schema. Keep this pre-policy v4 shape only as a
        // negative canary until bd-artifactstore-v4-evidence-admission-zlhvo
        // replaces it with the closed diagnostic/evidence admission contract.
        const PRE_POLICY_V4_BYTES: &[u8] = include_bytes!("../fixtures/artifact-object-v4.json");
        let object: ArtifactObject =
            serde_json::from_slice(PRE_POLICY_V4_BYTES).expect("decode pre-policy v4 canary");
        assert_eq!(object.object_schema_version, 4);
        assert!(matches!(
            object.validate(),
            Err(GauntletError::InvalidContract { ref reason })
                if reason.contains("reserved pre-policy artifact v4")
        ));
        assert!(matches!(
            hash_object_bytes(PRE_POLICY_V4_BYTES, 4),
            Err(GauntletError::InvalidContract { ref reason })
                if reason.contains("no registered hash domain")
        ));
    }

    #[test]
    fn unknown_future_object_schema_is_never_interpreted_as_current_evidence() {
        let mut object = sample_object();
        object.object_schema_version = OBJECT_SCHEMA_VERSION.saturating_add(1);
        assert!(matches!(
            object.validate(),
            Err(GauntletError::InvalidContract { ref reason })
                if reason.contains("unsupported") && reason.contains("non-admissible")
        ));
        assert!(matches!(
            object.object_hash(),
            Err(GauntletError::InvalidContract { ref reason })
                if reason.contains("no registered hash domain")
        ));
    }

    #[test]
    fn legacy_run_manifest_cannot_reference_a_current_sha256_object() {
        let store = ArtifactStore::default();
        let mut prepared = store
            .prepare_diagnostic("legacy-run-manifest", &sample_object(), BTreeMap::new())
            .expect("prepare current artifact");
        assert_eq!(
            prepared.run_manifest.schema_version,
            RUN_MANIFEST_SCHEMA_VERSION
        );
        assert_eq!(prepared.object_hash.len(), 64);

        prepared.run_manifest.schema_version = 1;
        prepared.run_manifest_bytes =
            serde_json::to_vec(&prepared.run_manifest).expect("encode legacy-shaped manifest");
        assert!(
            store.validate_prepared(&prepared).is_err(),
            "legacy manifest semantics must not ambiguously carry a current SHA-256 address"
        );
    }

    #[test]
    fn current_generator_metadata_requires_campaign_context() {
        let mut object = sample_object();
        object.case.metadata.generator_id = Some(GENERATOR_ID.to_owned());
        object.case.metadata.generator_seed = Some(42);
        object.case.metadata.corpus_hash = Some("0".repeat(64));

        assert!(
            ArtifactStore::default()
                .prepare("missing-campaign-context", &object, BTreeMap::new())
                .is_err()
        );
    }

    #[test]
    fn comparator_policy_changes_object_hash() {
        let first = sample_object();
        let mut second = first.clone();
        second.comparator_config = second
            .comparator_config
            .with_score_epsilon_reason(crate::ScoreEpsilonReason::PlatformLibm);
        second.comparison = compare_observations(
            second.comparison.subject.clone(),
            second.comparison.oracle.clone(),
            second.comparator_config,
        )
        .expect("comparison under recorded policy");
        assert_ne!(first.object_hash().unwrap(), second.object_hash().unwrap());
    }

    #[test]
    fn canonical_object_golden_bytes_and_hash_are_pinned() {
        let mut object = sample_object();
        bind_producer_identity(&mut object, golden_producer_build_identity());
        let canonical = object.canonical_bytes().unwrap();
        let golden_with_newline = include_bytes!("../fixtures/artifact-object-v5.json");
        let golden = golden_with_newline
            .strip_suffix(b"\n")
            .expect("golden fixture must end in exactly one LF");
        assert_eq!(
            object.object_hash().unwrap(),
            "b1413738df8834c270a270a6757a72f209066ff40fd4c82ab24ddfa219b7fddf"
        );
        assert_eq!(canonical, golden);
    }

    #[test]
    fn object_embeds_both_engines_and_the_exact_version_contract() {
        let object = sample_object();
        assert_eq!(object.engines.subject.family, EngineFamily::Quill);
        assert_eq!(object.engines.oracle.family, EngineFamily::Tantivy);
        assert_eq!(object.oracle_version, oracle_version_contract().unwrap());
        assert_eq!(object.object_hash().unwrap().len(), 64);
        let encoded = serde_json::to_value(&object).expect("serialize object");
        let pointer = object
            .comparison
            .first_divergence
            .as_deref()
            .expect("representative divergence");
        assert!(encoded.pointer(pointer).is_some());
    }

    #[test]
    fn score_bit_dtos_preserve_negative_zero_and_nan_payloads() {
        let values = [(-0.0_f32).to_bits(), f32::from_bits(0x7fc0_1234).to_bits()];
        let json = serde_json::to_vec(&values).expect("serialize bits");
        let decoded: [u32; 2] = serde_json::from_slice(&json).expect("deserialize bits");
        assert_eq!(decoded, values);
    }

    #[test]
    fn unsafe_run_ids_are_rejected() {
        let object = sample_object();
        let store = ArtifactStore::default();
        assert!(
            store
                .prepare("../escape", &object, BTreeMap::new())
                .is_err()
        );
        assert!(store.prepare("", &object, BTreeMap::new()).is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn persist_is_idempotent_and_rejects_cross_store_preparation() {
        let temp = tempfile::tempdir().expect("temporary parent");
        let first_store = ArtifactStore::new(temp.path().join("first"));
        let second_store = ArtifactStore::new(temp.path().join("second"));
        let prepared = first_store
            .prepare_diagnostic("run-one", &sample_object(), BTreeMap::new())
            .expect("prepare artifact");

        first_store.persist(&prepared).expect("first publication");
        first_store
            .persist(&prepared)
            .expect("idempotent publication");
        assert_eq!(
            std::fs::read(prepared.object_path()).expect("read object"),
            prepared.object_bytes()
        );
        assert!(second_store.persist(&prepared).is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn campaign_and_standalone_run_namespaces_do_not_alias() {
        let temp = tempfile::tempdir().expect("temporary parent");
        let store = ArtifactStore::new(temp.path().join("gauntlet"));
        store
            .reserve_campaign_run("foo", br#"{"schema_version":1}"#)
            .expect("reserve campaign");
        let standalone_marker_name = store
            .prepare_diagnostic("foo.campaign", &sample_object(), BTreeMap::new())
            .expect("standalone marker-like ID");
        let standalone_case_name = store
            .prepare_diagnostic("foo.q000000", &sample_object(), BTreeMap::new())
            .expect("standalone case-like ID");
        let campaign_case = store
            .prepare_campaign_case("foo", 0, &sample_campaign_object(), BTreeMap::new())
            .expect("campaign case");

        store
            .persist(&standalone_marker_name)
            .expect("standalone marker-like run");
        store
            .persist(&standalone_case_name)
            .expect("standalone case-like run");
        store.persist(&campaign_case).expect("campaign case run");

        assert_ne!(standalone_case_name.run_path(), campaign_case.run_path());
        assert!(standalone_marker_name.run_path().is_file());
        assert!(standalone_case_name.run_path().is_file());
        assert!(campaign_case.run_path().is_file());
        assert!(
            store
                .root()
                .join("campaigns/foo/reservation.json")
                .is_file()
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn concurrent_identical_publications_both_succeed() {
        use std::sync::{Arc, Barrier};

        let temp = tempfile::tempdir().expect("temporary parent");
        let store = Arc::new(ArtifactStore::new(temp.path().join("gauntlet")));
        let prepared = Arc::new(
            store
                .prepare_diagnostic("concurrent", &sample_object(), BTreeMap::new())
                .expect("prepare artifact"),
        );
        let barrier = Arc::new(Barrier::new(2));
        let workers = (0..2)
            .map(|_| {
                let store = Arc::clone(&store);
                let prepared = Arc::clone(&prepared);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    store.persist(&prepared)
                })
            })
            .collect::<Vec<_>>();

        for worker in workers {
            worker.join().expect("worker did not panic").unwrap();
        }
        assert_eq!(
            std::fs::read_dir(store.root().join("objects"))
                .expect("read object directory")
                .count(),
            1
        );
        assert_eq!(
            std::fs::read_dir(store.root().join("runs"))
                .expect("read run directory")
                .count(),
            1
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn partial_staging_file_is_resumed_before_atomic_publish() {
        let temp = tempfile::tempdir().expect("temporary parent");
        let store = ArtifactStore::new(temp.path().join("gauntlet"));
        let prepared = store
            .prepare_diagnostic("resume-staging", &sample_object(), BTreeMap::new())
            .expect("prepare artifact");
        std::fs::create_dir(store.root()).expect("create store root");
        std::fs::create_dir(store.root().join("objects")).expect("create objects directory");
        std::fs::create_dir(store.root().join("runs")).expect("create runs directory");
        let final_name = prepared
            .object_path()
            .file_name()
            .expect("object file name");
        let mut pending_name = OsString::from(".");
        pending_name.push(final_name);
        pending_name.push(".pending");
        let split = prepared.object_bytes().len() / 2;
        std::fs::write(
            store.root().join("objects").join(pending_name),
            &prepared.object_bytes()[..split],
        )
        .expect("write partial staging prefix");

        store.persist(&prepared).expect("resume publication");
        assert_eq!(
            std::fs::read(prepared.object_path()).expect("read published object"),
            prepared.object_bytes()
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn reused_run_id_with_different_provenance_is_rejected() {
        let temp = tempfile::tempdir().expect("temporary parent");
        let store = ArtifactStore::new(temp.path().join("gauntlet"));
        let first = store
            .prepare_diagnostic(
                "same-run",
                &sample_object(),
                BTreeMap::from([("worker".to_owned(), "one".to_owned())]),
            )
            .expect("first preparation");
        let second = store
            .prepare_diagnostic(
                "same-run",
                &sample_object(),
                BTreeMap::from([("worker".to_owned(), "two".to_owned())]),
            )
            .expect("second preparation");
        store.persist(&first).expect("first publication");
        assert!(matches!(
            store.persist(&second),
            Err(GauntletError::RunManifestConflict { .. })
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn salej_descriptor_relative_store_creation_creates_every_missing_component() {
        let temp = tempfile::tempdir().expect("temporary parent");
        let root = temp
            .path()
            .join("first-missing")
            .join("second-missing")
            .join("gauntlet");
        let store = ArtifactStore::new(root.clone());
        let prepared = store
            .prepare_diagnostic("nested-root", &sample_object(), BTreeMap::new())
            .expect("prepare nested-root artifact");

        store
            .persist(&prepared)
            .expect("create and publish through every descriptor-relative component");

        assert!(root.is_dir());
        assert!(prepared.object_path().is_file());
        assert!(prepared.run_path().is_file());
    }

    #[cfg(all(target_os = "linux", unix))]
    #[test]
    fn symlinked_store_subdirectory_is_rejected() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().expect("temporary parent");
        let root = temp.path().join("gauntlet");
        std::fs::create_dir(&root).expect("create root");
        let redirect = temp.path().join("redirect");
        std::fs::create_dir(&redirect).expect("create redirect");
        symlink(&redirect, root.join("objects")).expect("create symlink");
        let store = ArtifactStore::new(root);
        let prepared = store
            .prepare_diagnostic("symlink", &sample_object(), BTreeMap::new())
            .expect("prepare artifact");
        assert!(matches!(
            store.persist(&prepared),
            Err(GauntletError::UnsafeStorePath { .. })
        ));
    }

    #[cfg(all(target_os = "linux", unix))]
    #[test]
    fn symlinked_store_ancestor_is_rejected() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().expect("temporary parent");
        let redirect = temp.path().join("redirect");
        std::fs::create_dir(&redirect).expect("create redirect");
        let link = temp.path().join("link");
        symlink(&redirect, &link).expect("create ancestor symlink");
        let store = ArtifactStore::new(link.join("gauntlet"));
        let prepared = store
            .prepare_diagnostic("symlink-ancestor", &sample_object(), BTreeMap::new())
            .expect("prepare artifact");
        assert!(matches!(
            store.persist(&prepared),
            Err(GauntletError::UnsafeStorePath { .. })
        ));
    }
}
