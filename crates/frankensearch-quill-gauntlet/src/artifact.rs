use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use xxhash_rust::xxh3::Xxh3;

use crate::GauntletError;
use crate::comparator::{
    ComparatorConfig, Divergence, LexicalBoundary, LexicalComparisonStatus,
    LexicalContractComparison, LexicalEngineRole, LexicalExposureContract,
    LexicalObservationContext, RankClass, compare_observations_stored_v7,
};
use crate::engine::{EnginePairIdentity, HarnessRun};
use crate::generator::{
    GeneratedQueryCase, QuerySuiteSource, is_supported_stored_generator_id,
    validate_generated_case_metadata, validate_stored_query_generator_identity,
};
use crate::runner::{
    CampaignContractMode, CampaignReport, DivergenceRegisterEntry, SemanticContract,
    lexical_backend_identity, lexical_query_contract_sha256,
};
use crate::version_contract::{OracleVersionContract, oracle_version_contract};

const ARTIFACT_OBJECT_V7_SCHEMA_VERSION: u32 = 7;
pub const OBJECT_SCHEMA_VERSION: u32 = ARTIFACT_OBJECT_V7_SCHEMA_VERSION;
const ARTIFACT_OBJECT_V7_CANONICALIZATION_VERSION: u32 = 1;
pub const CANONICALIZATION_VERSION: u32 = ARTIFACT_OBJECT_V7_CANONICALIZATION_VERSION;
pub const OBJECT_HASH_SCHEME_V7_SHA256: &str =
    "frankensearch-quill-gauntlet/artifact-object/v7/sha256";
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
const HASH_DOMAIN_V7: &[u8] = b"frankensearch-quill-gauntlet:artifact-object:v7\0";
const PRODUCER_BUILD_IDENTITY_V2_HASH_DOMAIN: &[u8] =
    b"frankensearch-quill-gauntlet:producer-build-identity:v2\0";
const PRODUCER_BUILD_IDENTITY_V2_SCHEMA_VERSION: u32 = 2;
const PRODUCER_BUILD_IDENTITY_SCHEMA_VERSION: u32 = PRODUCER_BUILD_IDENTITY_V2_SCHEMA_VERSION;
const PRODUCER_CONTRACT_VERSION_V5: &str = "frankensearch.quill-local-perf-producer.v5";
const CURRENT_PRODUCER_CONTRACT_VERSION: &str = PRODUCER_CONTRACT_VERSION_V5;
const MAX_CAMPAIGN_RESERVATION_BYTES: u64 = 512 * 1024 * 1024;
const MAX_CAMPAIGN_REPORT_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_CAMPAIGN_RUN_MANIFEST_BYTES: u64 = 2 * 1024 * 1024;
const MAX_CAMPAIGN_OBJECT_BYTES: u64 = 512 * 1024 * 1024;
const MAX_CAMPAIGN_COMPLETION_RECEIPT_BYTES: u64 = 16 * 1024;
const CAMPAIGN_COMPLETION_RECEIPT_V1_SCHEMA_VERSION: u32 = 1;
const CAMPAIGN_COMPLETION_RECEIPT_V1_HASH_DOMAIN: &[u8] =
    b"frankensearch-quill-gauntlet:campaign-completion-receipt:v1\0";

#[derive(Deserialize)]
struct CampaignReportSchemaProbe {
    schema_version: u32,
}

#[derive(Deserialize)]
struct ArtifactObjectSchemaProbe {
    object_schema_version: u32,
}

/// Closed, machine-readable disposition of a serialized artifact generation.
///
/// Classification examines only the duplicate-safe schema probe. It does not
/// deserialize a historical generation into the current DTO and never grants
/// trust or admission authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializedSchemaDisposition {
    UnauthenticatedLegacy { schema_version: u32 },
    ReservedRejected { schema_version: u32 },
    LegacyIntegrityCeiling { schema_version: u32 },
    CurrentIntegrityContractCandidate { schema_version: u32 },
}

impl SerializedSchemaDisposition {
    #[must_use]
    pub const fn schema_version(self) -> u32 {
        match self {
            Self::UnauthenticatedLegacy { schema_version }
            | Self::ReservedRejected { schema_version }
            | Self::LegacyIntegrityCeiling { schema_version }
            | Self::CurrentIntegrityContractCandidate { schema_version } => schema_version,
        }
    }
}

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

/// How the running process proved the bytes of its own executable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GauntletExecutableVerification {
    /// Decode-only sentinel for producer receipts predating this contract.
    #[default]
    LegacyMissing,
    /// Linux hashed the kernel-held `/proc/self/exe` image.
    ProcfsRunningImage,
    /// The platform exposed only a path snapshot. This remains useful for
    /// diagnostics but is never admissible performance evidence.
    PathSnapshot,
}

/// Build-script-sealed identity of the binary that produced an artifact.
///
/// Current artifact construction compares this complete record against the
/// values compiled into the executing gauntlet. Callers therefore cannot make
/// two engine descriptors agree on a fabricated source revision and have that
/// agreement mistaken for producer provenance. Dirty and unverified builds
/// remain recordable diagnostics; the current integrity contract rejects them.
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
    #[serde(default)]
    pub executable_verification: GauntletExecutableVerification,
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
        let (executable_sha256, executable_byte_len, executable_verification) =
            current_executable_identity()?;
        let identity = Self {
            schema_version: PRODUCER_BUILD_IDENTITY_SCHEMA_VERSION,
            producer_contract_version: env!("QUILL_ARTIFACT_PRODUCER_CONTRACT_VERSION").to_owned(),
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
            executable_verification,
        };
        identity.validate_creation_contract()?;
        Ok(identity)
    }

    pub(crate) fn identity_hash(&self) -> Result<String, GauntletError> {
        self.validate_stored_v2()?;
        let mut hasher = Sha256::new();
        hasher.update(PRODUCER_BUILD_IDENTITY_V2_HASH_DOMAIN);
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

    /// Validate the immutable producer-identity v2 archive contract.
    pub(crate) fn validate_stored_v2(&self) -> Result<(), GauntletError> {
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
        if self.schema_version != PRODUCER_BUILD_IDENTITY_V2_SCHEMA_VERSION
            || self.producer_contract_version != PRODUCER_CONTRACT_VERSION_V5
            || !source_is_well_formed
            || !cargo_lock_is_well_formed
            || !rustc_is_well_formed
            || !text_is_well_formed
            || !features_are_canonical
            || !is_lower_hex_len(&self.enabled_features_sha256, 64)
            || self.enabled_features_sha256 != expected_features_sha256
            || !is_lower_hex_len(&self.executable_sha256, 64)
            || self.executable_byte_len == 0
            || self.executable_verification == GauntletExecutableVerification::LegacyMissing
        {
            return Err(GauntletError::InvalidContract {
                reason: "artifact producer build identity is missing, malformed, or internally inconsistent"
                    .to_owned(),
            });
        }
        Ok(())
    }

    /// Validate a newly created receipt against the current producer schema.
    fn validate_creation_contract(&self) -> Result<(), GauntletError> {
        self.validate_stored_v2()?;
        if self.schema_version != PRODUCER_BUILD_IDENTITY_SCHEMA_VERSION
            || self.producer_contract_version != CURRENT_PRODUCER_CONTRACT_VERSION
        {
            return Err(GauntletError::InvalidContract {
                reason: "new producer identity does not use the current schema and contract"
                    .to_owned(),
            });
        }
        Ok(())
    }

    pub(crate) fn validate_builtin_engines(
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
        self.validate_creation_contract()?;
        if self != &Self::compiled()? {
            return Err(GauntletError::InvalidContract {
                reason: "producer identity does not match the exact executing gauntlet binary"
                    .to_owned(),
            });
        }
        Ok(())
    }

    pub(crate) fn validate_stored_sealed_v2(&self) -> Result<(), GauntletError> {
        self.validate_stored_v2()?;
        let executable_is_held = self
            .target_triple
            .split('-')
            .any(|component| component == "linux")
            && self.executable_verification == GauntletExecutableVerification::ProcfsRunningImage;
        if self.source_verification != GauntletProducerSourceVerification::GitCheckoutVerified
            || self.source_git_dirty
            || !is_lower_hex_len(&self.cargo_lock_sha256, 64)
            || !executable_is_held
        {
            return Err(GauntletError::InvalidContract {
                reason: "sealed built-in integrity requires a clean Git-verified producer, an exact Cargo.lock identity, and a kernel-held running executable image; path snapshots are diagnostic-only"
                    .to_owned(),
            });
        }
        Ok(())
    }

    /// Re-resolve the exact live checkout immediately before evidence
    /// persistence. This closes the Cargo build-script cache gap for newly
    /// created untracked files and index flags such as assume-unchanged or
    /// skip-worktree. Archived replay deliberately never calls this method.
    pub(crate) fn validate_live_source_checkout(&self) -> Result<(), GauntletError> {
        self.validate_stored_sealed_v2()?;
        let expected_root = producer_workspace_root();
        validate_live_git_checkout(&expected_root, &expected_root, &self.source_git_revision)
    }

    pub(crate) fn require_features(&self, required: &[&str]) -> Result<(), GauntletError> {
        self.validate_stored_v2()?;
        if let Some(missing) = required.iter().find(|feature| {
            !self
                .enabled_features
                .iter()
                .any(|enabled| enabled == **feature)
        }) {
            return Err(GauntletError::InvalidContract {
                reason: format!(
                    "built-in integrity producer is missing required Cargo feature {missing:?}"
                ),
            });
        }
        Ok(())
    }
}

fn current_executable_identity()
-> Result<(String, u64, GauntletExecutableVerification), GauntletError> {
    static IDENTITY: std::sync::OnceLock<
        Result<(String, u64, GauntletExecutableVerification), String>,
    > = std::sync::OnceLock::new();
    IDENTITY
        .get_or_init(capture_current_executable_identity)
        .clone()
        .map_err(|reason| GauntletError::InvalidContract { reason })
}

fn capture_current_executable_identity()
-> Result<(String, u64, GauntletExecutableVerification), String> {
    #[cfg(target_os = "linux")]
    let path = PathBuf::from("/proc/self/exe");
    #[cfg(not(target_os = "linux"))]
    let path = std::env::current_exe()
        .map_err(|error| format!("cannot resolve the producer executable: {error}"))?;
    let mut file = File::open(&path)
        .map_err(|error| format!("cannot open the producer executable: {error}"))?;
    let metadata_before = file
        .metadata()
        .map_err(|error| format!("cannot stat the producer executable: {error}"))?;
    if !metadata_before.is_file() {
        return Err("producer executable identity requires a regular file".to_owned());
    }
    let mut hasher = Sha256::new();
    let mut byte_len = 0_u64;
    let mut buffer = vec![0_u8; 64 * 1024];
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
    let metadata_after = file
        .metadata()
        .map_err(|error| format!("cannot restat the producer executable: {error}"))?;
    if byte_len == 0
        || byte_len != metadata_before.len()
        || metadata_before.len() != metadata_after.len()
        || metadata_before.modified().ok() != metadata_after.modified().ok()
    {
        return Err("producer executable changed while its identity was captured".to_owned());
    }
    let executable_sha256 = lower_hex(&hasher.finalize());
    #[cfg(target_os = "linux")]
    let verification = GauntletExecutableVerification::ProcfsRunningImage;
    #[cfg(not(target_os = "linux"))]
    let verification = GauntletExecutableVerification::PathSnapshot;
    Ok((executable_sha256, byte_len, verification))
}

fn producer_workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn validate_live_git_checkout(
    repository: &Path,
    expected_root: &Path,
    expected_revision: &str,
) -> Result<(), GauntletError> {
    if !is_lower_hex_len(expected_revision, 40) {
        return Err(GauntletError::InvalidContract {
            reason: "live producer Git preflight requires an exact lowercase commit identity"
                .to_owned(),
        });
    }
    let repository = repository
        .canonicalize()
        .map_err(|error| GauntletError::InvalidContract {
            reason: format!("cannot canonicalize live producer checkout: {error}"),
        })?;
    let expected_root =
        expected_root
            .canonicalize()
            .map_err(|error| GauntletError::InvalidContract {
                reason: format!("cannot canonicalize producer workspace root: {error}"),
            })?;
    let reported_root = live_git_text(&repository, &["rev-parse", "--show-toplevel"])?;
    let reported_root = PathBuf::from(reported_root)
        .canonicalize()
        .map_err(|error| GauntletError::InvalidContract {
            reason: format!("cannot canonicalize live Git root: {error}"),
        })?;
    let revision = live_git_text(&repository, &["rev-parse", "HEAD"])?;
    let status = live_git_bytes(
        &repository,
        &["status", "--porcelain=v1", "-z", "--untracked-files=all"],
    )?;
    let verbose_index = live_git_bytes(&repository, &["ls-files", "-v", "-z"])?;
    let tagged_index = live_git_bytes(&repository, &["ls-files", "-t", "-z"])?;
    let hides_worktree_changes = verbose_index
        .split(|byte| *byte == 0)
        .filter_map(|entry| entry.first())
        .any(u8::is_ascii_lowercase)
        || tagged_index
            .split(|byte| *byte == 0)
            .any(|entry| entry.starts_with(b"S "));
    if reported_root != expected_root
        || revision != expected_revision
        || !status.is_empty()
        || hides_worktree_changes
    {
        return Err(GauntletError::InvalidContract {
            reason: "live producer checkout is not the exact clean, fully visible revision sealed into the executing binary"
                .to_owned(),
        });
    }
    Ok(())
}

fn live_git_bytes(repository: &Path, args: &[&str]) -> Result<Vec<u8>, GauntletError> {
    let mut command = std::process::Command::new("git");
    command.arg("-C").arg(repository).args(args);
    for (name, _) in std::env::vars_os() {
        if name.as_encoded_bytes().starts_with(b"GIT_") {
            command.env_remove(name);
        }
    }
    let output = command
        .output()
        .map_err(|error| GauntletError::InvalidContract {
            reason: format!("cannot execute live producer Git preflight: {error}"),
        })?;
    if !output.status.success() {
        return Err(GauntletError::InvalidContract {
            reason: "live producer Git preflight failed closed".to_owned(),
        });
    }
    Ok(output.stdout)
}

fn live_git_text(repository: &Path, args: &[&str]) -> Result<String, GauntletError> {
    let output = live_git_bytes(repository, args)?;
    let output = String::from_utf8(output).map_err(|error| GauntletError::InvalidContract {
        reason: format!("live producer Git identity is not UTF-8: {error}"),
    })?;
    Ok(output.trim().to_owned())
}

fn is_lower_hex_text(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn decode_lower_hex(value: &str) -> Option<Vec<u8>> {
    if !is_lower_hex_text(value) {
        return None;
    }
    let (pairs, remainder) = value.as_bytes().as_chunks::<2>();
    if !remainder.is_empty() {
        return None;
    }
    pairs
        .iter()
        .map(|&[high, low]| {
            let high = lower_hex_nibble(high)?;
            let low = lower_hex_nibble(low)?;
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
    pub query_generator_schema_version: u32,
    pub query_generator_id: String,
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
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
#[serde(tag = "scope", rename_all = "snake_case", deny_unknown_fields)]
pub enum ArtifactLexicalContractEvidence {
    /// Decode-only marker for pre-v3 objects, which are no longer admissible.
    #[default]
    LegacyPreV3Missing,
    /// The object intentionally proves only the legacy rich result envelope.
    RankEnvelopeOnly,
    /// Complete replayable ordinary lexical-read comparison.
    CoreLexicalV3 {
        comparison: Box<LexicalContractComparison>,
    },
}

impl<'de> Deserialize<'de> for ArtifactLexicalContractEvidence {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(tag = "scope", rename_all = "snake_case", deny_unknown_fields)]
        enum StrictWire {
            LegacyPreV3Missing {},
            RankEnvelopeOnly {},
            CoreLexicalV3 {
                comparison: Box<LexicalContractComparison>,
            },
        }

        Ok(match StrictWire::deserialize(deserializer)? {
            StrictWire::LegacyPreV3Missing {} => Self::LegacyPreV3Missing,
            StrictWire::RankEnvelopeOnly {} => Self::RankEnvelopeOnly,
            StrictWire::CoreLexicalV3 { comparison } => Self::CoreLexicalV3 { comparison },
        })
    }
}

/// Dependency claim carried by an artifact.
///
/// Diagnostics record the adapters' own descriptors without pretending that
/// an arbitrary oracle is the built-in Tantivy wrapper. Evidence produced by
/// the typed built-in lane additionally seals the exact oracle dependency
/// contract used by that binary.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ArtifactOracleDependency {
    /// Decode-only sentinel for artifacts predating an explicit role claim.
    #[default]
    LegacyMissing,
    /// No claim beyond the independently recorded diagnostic descriptors.
    DiagnosticUnspecified,
    /// Exact dependency contract for the linked Tantivy oracle adapter.
    BuiltInTantivy { contract: OracleVersionContract },
}

impl<'de> Deserialize<'de> for ArtifactOracleDependency {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
        enum StrictWire {
            LegacyMissing {},
            DiagnosticUnspecified {},
            BuiltInTantivy { contract: OracleVersionContract },
        }

        Ok(match StrictWire::deserialize(deserializer)? {
            StrictWire::LegacyMissing {} => Self::LegacyMissing,
            StrictWire::DiagnosticUnspecified {} => Self::DiagnosticUnspecified,
            StrictWire::BuiltInTantivy { contract } => Self::BuiltInTantivy { contract },
        })
    }
}

/// Historical v1 dependency DTO retained solely so legacy object bytes decode
/// far enough to receive an explicit non-admissible schema verdict.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyOracleVersionContractV1 {
    schema_version: u32,
    tantivy_version: String,
    tantivy_checksum_sha256: String,
    lexical_package: String,
    lexical_package_version: String,
    lexical_git_revision: String,
    source_dirty_allowed: bool,
}

/// Immutable comparison object. Run-local provenance is deliberately absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ArtifactObject {
    pub object_schema_version: u32,
    pub canonicalization_version: u32,
    /// Maximum trust the serialized object could carry after all required
    /// validation. This is self-description, never proof that validation ran.
    #[serde(default)]
    pub trust_ceiling: ArtifactTrustCeiling,
    /// Durable description of the execution path that produced these bytes.
    /// It is not an admission or authorization capability.
    #[serde(default)]
    pub execution_role: ArtifactExecutionRole,
    #[serde(
        default,
        rename = "oracle_version",
        skip_serializing_if = "Option::is_none"
    )]
    legacy_oracle_version: Option<LegacyOracleVersionContractV1>,
    #[serde(default)]
    oracle_dependency: ArtifactOracleDependency,
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

/// Closed, validated identity surface used when a current artifact first
/// observes a divergence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactDivergenceBinding {
    pub object_schema_version: u32,
    pub object_hash_scheme: &'static str,
    pub object_hash: String,
    pub producer_identity_sha256: String,
    pub oracle_dependency_identity_sha256: String,
    pub oracle_lexical_contract_audit_revision: String,
    pub corpus_manifest_sha256: String,
    pub query_manifest_sha256: String,
    pub query_suite_source: QuerySuiteSource,
    pub query_source_identity_sha256: String,
    pub fixture_id: String,
    pub rank_class: RankClass,
    pub divergences: Vec<Divergence>,
}

impl ArtifactObject {
    pub(crate) const fn oracle_dependency(&self) -> &ArtifactOracleDependency {
        &self.oracle_dependency
    }

    /// Return the maximum trust permitted by this object's schema version.
    ///
    /// This does not validate the DTO, verify its address, or establish that
    /// relational replay occurred. Callers must never use this ceiling as an
    /// achieved trust classification.
    ///
    /// # Errors
    ///
    /// Returns an error when the schema version is reserved or has no registered
    /// trust classification.
    pub fn schema_trust_ceiling(&self) -> Result<ArtifactTrustCeiling, GauntletError> {
        match self.object_schema_version {
            1..=3 | 5 => Ok(ArtifactTrustCeiling::UnauthenticatedLegacy),
            4 => Err(GauntletError::InvalidContract {
                reason: "reserved pre-policy artifact v4 has no trust classification".to_owned(),
            }),
            6 | ARTIFACT_OBJECT_V7_SCHEMA_VERSION => Ok(ArtifactTrustCeiling::IntegrityOnly),
            schema_version => Err(GauntletError::InvalidContract {
                reason: format!(
                    "artifact object schema version {schema_version} has no trust classification"
                ),
            }),
        }
    }

    /// Build an immutable object from one completed harness run.
    ///
    /// # Errors
    ///
    /// Returns an error when the committed oracle version contract is invalid.
    pub fn from_diagnostic_run(run: HarnessRun) -> Result<Self, GauntletError> {
        run.validate_diagnostic_creation()?;
        Ok(Self {
            object_schema_version: OBJECT_SCHEMA_VERSION,
            canonicalization_version: CANONICALIZATION_VERSION,
            trust_ceiling: ArtifactTrustCeiling::IntegrityOnly,
            execution_role: ArtifactExecutionRole::Diagnostic,
            legacy_oracle_version: None,
            oracle_dependency: ArtifactOracleDependency::DiagnosticUnspecified,
            producer_build_identity: run.producer_build_identity,
            engines: run.engines,
            case: run.case,
            comparator_config: run.comparator_config,
            comparison: run.comparison,
            lexical_contract: ArtifactLexicalContractEvidence::RankEnvelopeOnly,
            campaign: None,
        })
    }

    /// Build a diagnostic object for one case in a generated campaign.
    ///
    /// # Errors
    ///
    /// Returns an error when the committed oracle version contract is invalid.
    pub(crate) fn from_diagnostic_campaign_run(
        run: HarnessRun,
        campaign: CampaignArtifactContext,
        lexical_contract: ArtifactLexicalContractEvidence,
    ) -> Result<Self, GauntletError> {
        let mut object = Self::from_diagnostic_run(run)?;
        object.lexical_contract = lexical_contract;
        object.campaign = Some(campaign);
        Ok(object)
    }

    /// Build an integrity-bound object for a typed built-in execution.
    pub(crate) fn from_builtin_campaign_execution(
        run: HarnessRun,
        campaign: CampaignArtifactContext,
        lexical_contract: ArtifactLexicalContractEvidence,
    ) -> Result<Self, GauntletError> {
        run.validate_builtin_evidence_creation()?;
        let object = Self {
            object_schema_version: OBJECT_SCHEMA_VERSION,
            canonicalization_version: CANONICALIZATION_VERSION,
            trust_ceiling: ArtifactTrustCeiling::IntegrityOnly,
            execution_role: ArtifactExecutionRole::BuiltInExecution,
            legacy_oracle_version: None,
            oracle_dependency: ArtifactOracleDependency::BuiltInTantivy {
                contract: oracle_version_contract()?,
            },
            producer_build_identity: run.producer_build_identity,
            engines: run.engines,
            case: run.case,
            comparator_config: run.comparator_config,
            comparison: run.comparison,
            lexical_contract,
            campaign: Some(campaign),
        };
        object.validate_current_builtin_integrity()?;
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
    /// before decoding. The current integrity-checked campaign loader instead rejects
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
        validate_stored_object_schema(self.object_schema_version)
            .map_err(|reason| GauntletError::InvalidContract { reason })?;
        if self.trust_ceiling != self.schema_trust_ceiling()? {
            return Err(GauntletError::InvalidContract {
                reason: "artifact trust ceiling does not match its schema".to_owned(),
            });
        }
        if self.canonicalization_version != ARTIFACT_OBJECT_V7_CANONICALIZATION_VERSION {
            return Err(GauntletError::InvalidContract {
                reason: "artifact canonicalization contract is invalid".to_owned(),
            });
        }
        if self.legacy_oracle_version.is_some() {
            return Err(GauntletError::InvalidContract {
                reason: "current artifact must not carry the legacy oracle-version field"
                    .to_owned(),
            });
        }
        self.producer_build_identity.validate_stored_v2()?;
        self.engines.validate_stored_contract()?;
        match (self.execution_role, &self.oracle_dependency) {
            (ArtifactExecutionRole::LegacyMissing, _) => {
                return Err(GauntletError::InvalidContract {
                    reason: "artifact execution role is missing".to_owned(),
                });
            }
            (_, ArtifactOracleDependency::LegacyMissing) => {
                return Err(GauntletError::InvalidContract {
                    reason: "artifact dependency role is missing".to_owned(),
                });
            }
            (
                ArtifactExecutionRole::Diagnostic,
                ArtifactOracleDependency::DiagnosticUnspecified,
            ) => {
                if self.engines.has_builtin_profile() {
                    return Err(GauntletError::InvalidContract {
                        reason:
                            "diagnostic artifact must not carry a built-in adapter/profile receipt"
                                .to_owned(),
                    });
                }
            }
            (
                ArtifactExecutionRole::BuiltInExecution,
                ArtifactOracleDependency::BuiltInTantivy { contract },
            ) => {
                contract.validate_stored_structure()?;
                if !self.engines.has_builtin_profile()
                    || self.engines.comparison_mode != crate::ComparisonMode::CrossEngine
                    || self.engines.oracle.crate_version != contract.lexical_package_version
                {
                    return Err(GauntletError::InvalidContract {
                        reason: "artifact oracle package does not match its embedded built-in dependency contract"
                            .to_owned(),
                    });
                }
            }
            (
                ArtifactExecutionRole::Diagnostic,
                ArtifactOracleDependency::BuiltInTantivy { .. },
            )
            | (
                ArtifactExecutionRole::BuiltInExecution,
                ArtifactOracleDependency::DiagnosticUnspecified,
            ) => {
                return Err(GauntletError::InvalidContract {
                    reason: "artifact execution role does not match its dependency role".to_owned(),
                });
            }
        }
        self.comparator_config.validate_stored_v7()?;
        validate_generated_case_metadata(&self.case)?;
        if self.campaign.is_none()
            && self
                .case
                .metadata
                .generator_id
                .as_deref()
                .is_some_and(is_supported_stored_generator_id)
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
        let recomputed = compare_observations_stored_v7(
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

    /// Validate the self-contained built-in integrity claim carried by bytes.
    ///
    /// This deliberately does not consult the executing binary, its linked
    /// dependencies, or the current checkout. This deliberately grants no
    /// authority beyond [`ArtifactTrustCeiling::IntegrityOnly`].
    pub(crate) fn validate_stored_builtin_integrity(&self) -> Result<(), GauntletError> {
        self.validate()?;
        self.producer_build_identity.validate_stored_sealed_v2()?;
        self.producer_build_identity
            .require_features(&["tantivy_oracle"])?;
        match &self.oracle_dependency {
            ArtifactOracleDependency::BuiltInTantivy { contract } => {
                contract.validate_stored_structure()?;
            }
            ArtifactOracleDependency::LegacyMissing
            | ArtifactOracleDependency::DiagnosticUnspecified => {
                return Err(GauntletError::InvalidContract {
                    reason:
                        "stored built-in integrity requires a typed Tantivy dependency contract"
                            .to_owned(),
                });
            }
        }
        self.producer_build_identity
            .validate_builtin_engines(&self.engines)?;
        self.engines.validate_stored_contract()
    }

    /// Validate a newly-created integrity object against the linked dependency.
    ///
    /// The exact executing producer is bound for reproducibility, not external
    /// authentication. Passing this check never mints admission authority.
    pub(crate) fn validate_current_builtin_integrity(&self) -> Result<(), GauntletError> {
        self.validate_stored_builtin_integrity()?;
        if self.object_schema_version != OBJECT_SCHEMA_VERSION {
            return Err(GauntletError::InvalidContract {
                reason: "new built-in integrity requires the current artifact object schema"
                    .to_owned(),
            });
        }
        self.comparator_config.validate_contract()?;
        match &self.oracle_dependency {
            ArtifactOracleDependency::BuiltInTantivy { contract }
                if contract == &oracle_version_contract()? => {}
            ArtifactOracleDependency::LegacyMissing
            | ArtifactOracleDependency::DiagnosticUnspecified
            | ArtifactOracleDependency::BuiltInTantivy { .. } => {
                return Err(GauntletError::InvalidContract {
                    reason: "new built-in integrity requires the exact current Tantivy dependency contract"
                        .to_owned(),
                });
            }
        }
        self.engines.validate_builtin_contract()
    }

    /// Return the complete identity tuple a v2 divergence observation must
    /// carry. The tuple cannot be built from shape-only caller strings: it is
    /// derived only after the role-bound artifact clears self-contained stored
    /// integrity validation. This intentionally remains usable by a newer
    /// verifier after the original producer or linked dependency has changed.
    pub(crate) fn divergence_binding(&self) -> Result<ArtifactDivergenceBinding, GauntletError> {
        self.validate_stored_builtin_integrity()?;
        let ArtifactOracleDependency::BuiltInTantivy { contract } = &self.oracle_dependency else {
            return Err(GauntletError::InvalidContract {
                reason: "divergence binding requires a built-in Tantivy dependency role".to_owned(),
            });
        };
        let campaign = self
            .campaign
            .as_ref()
            .ok_or_else(|| GauntletError::InvalidContract {
                reason: "divergence binding requires integrity-bound campaign manifest context"
                    .to_owned(),
            })?;
        Ok(ArtifactDivergenceBinding {
            object_schema_version: self.object_schema_version,
            object_hash_scheme: OBJECT_HASH_SCHEME_V7_SHA256,
            object_hash: self.object_hash()?,
            producer_identity_sha256: self.producer_build_identity.identity_hash()?,
            oracle_dependency_identity_sha256: contract.identity_sha256()?,
            oracle_lexical_contract_audit_revision: contract
                .lexical_contract_audit_revision
                .clone(),
            corpus_manifest_sha256: campaign.corpus_manifest_hash.clone(),
            query_manifest_sha256: campaign.query_manifest_hash.clone(),
            query_suite_source: campaign.query_suite_source,
            query_source_identity_sha256: campaign.query_source_identity_sha256.clone(),
            fixture_id: self.case.fixture_id.clone(),
            rank_class: self.comparison.rank_class,
            divergences: self.comparison.divergences.clone(),
        })
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
            && validate_stored_query_generator_identity(
                self.query_generator_schema_version,
                &self.query_generator_id,
            )
            .is_ok()
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
                case.metadata.generator_id.as_deref() == Some(self.query_generator_id.as_str())
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

const ARTIFACTSTORE_V4_SOURCE_SNAPSHOT_SCHEMA_VERSION: u32 = 1;
const ARTIFACTSTORE_V4_SOURCE_SNAPSHOT_HASH_DOMAIN: &[u8] =
    b"frankensearch.artifactstore.v4.source\0";
const MAX_ARTIFACTSTORE_V4_SOURCE_SNAPSHOT_BYTES: u64 = 16 * 1024 * 1024;
const ARTIFACTSTORE_V4_BUILD_SNAPSHOT_SCHEMA_VERSION: u32 = 1;
const ARTIFACTSTORE_V4_BUILD_SNAPSHOT_HASH_DOMAIN: &[u8] =
    b"frankensearch.artifactstore.v4.build\0";
const MAX_ARTIFACTSTORE_V4_BUILD_SNAPSHOT_BYTES: u64 = 16 * 1024 * 1024;

/// Immutable Source-to-Build link for an `ArtifactStore` v4 receipt chain.
///
/// This is deliberately separate from the legacy `CampaignReport` identity:
/// a report may reference the chain, but cannot substitute for either v4
/// object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactStoreV4SourceBuildBinding {
    pub source_identity_sha256: String,
    pub build_identity_sha256: String,
}

impl ArtifactStoreV4SourceBuildBinding {
    fn new(
        source: &ArtifactStoreV4SourceSnapshot,
        build: &ArtifactStoreV4BuildSnapshot,
    ) -> Result<Self, GauntletError> {
        source.validate()?;
        build.validate()?;
        if build.source_identity_sha256 != source.identity_sha256 {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "ArtifactStore v4 Build snapshot is not bound to its Source snapshot"
                    .to_owned(),
            });
        }
        Ok(Self {
            source_identity_sha256: source.identity_sha256.clone(),
            build_identity_sha256: build.identity_sha256.clone(),
        })
    }

    pub(crate) fn validate(&self) -> Result<(), GauntletError> {
        if !is_lower_sha256(&self.source_identity_sha256)
            || !is_lower_sha256(&self.build_identity_sha256)
        {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "ArtifactStore v4 Source/Build binding has malformed identities".to_owned(),
            });
        }
        Ok(())
    }
}

/// Two immutable snapshots that form the first two `ArtifactStore` v4 objects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactStoreV4SourceBuildSnapshots {
    source: ArtifactStoreV4SourceSnapshot,
    build: ArtifactStoreV4BuildSnapshot,
}

impl ArtifactStoreV4SourceBuildSnapshots {
    /// Construct a fully bound Source-to-Build pair.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError`] when either snapshot fails its own
    /// revalidation, or when the build snapshot's recorded source identity
    /// does not equal the source snapshot's identity.
    pub fn new(
        source: ArtifactStoreV4SourceSnapshot,
        build: ArtifactStoreV4BuildSnapshot,
    ) -> Result<Self, GauntletError> {
        ArtifactStoreV4SourceBuildBinding::new(&source, &build)?;
        Ok(Self { source, build })
    }

    /// Collect the current Linux producer's source workspace and the build
    /// facts bound into its running `/proc/self/exe` image.
    ///
    /// A clean Git checkout receives tracked-source selection plus live
    /// checkout fencing. A Git-less producer instead receives a complete
    /// observable-workspace selection and a typed source-provenance Build
    /// input. That records authentic but unadmitted diagnostic evidence; it
    /// never satisfies the separate sealed-admission contract. The resulting
    /// Build object carries the kernel-held executable digest, not a
    /// replaceable path.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError`] when source selection, descriptor capture,
    /// compiled producer identity, or Linux running-image binding fails.
    pub fn collect_current_linux() -> Result<Self, GauntletError> {
        #[cfg(target_os = "linux")]
        {
            let producer = GauntletProducerBuildIdentity::compiled()?;
            producer.validate_creation_contract()?;
            // `CARGO_MANIFEST_DIR` is compiled into the binary and the
            // workspace-relative spelling deliberately contains `../..`.
            // Pinning rejects parent components, so resolve that trusted
            // producer location before handing it to descriptor-safe capture.
            let root = producer_workspace_root().canonicalize().map_err(|error| {
                GauntletError::InvalidPreparedArtifact {
                    reason: format!(
                        "ArtifactStore v4 cannot canonicalize the producer workspace root: {error}"
                    ),
                }
            })?;
            let has_live_git_provenance = producer.source_verification
                == GauntletProducerSourceVerification::GitCheckoutVerified
                && !producer.source_git_dirty;
            if has_live_git_provenance {
                producer.validate_live_source_checkout()?;
            }
            let snapshots =
                Self::collect_linux_workspace(&root, &producer, has_live_git_provenance)?;
            // The first checkout check fences stale construction. Repeating it
            // after descriptor capture rejects a tracked file changed between
            // selection and publication rather than admitting a mixed source
            // generation against the already-running executable. Git-less
            // producers remain explicitly unadmitted diagnostics instead.
            if has_live_git_provenance {
                producer.validate_live_source_checkout()?;
            }
            Ok(snapshots)
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(GauntletError::InvalidPreparedArtifact {
                reason: "ArtifactStore v4 current-process collection requires Linux /proc/self/exe"
                    .to_owned(),
            })
        }
    }

    fn collect_linux_workspace(
        root: &Path,
        producer: &GauntletProducerBuildIdentity,
        has_live_git_provenance: bool,
    ) -> Result<Self, GauntletError> {
        producer.validate_stored_v2()?;
        let selected = if has_live_git_provenance {
            collect_tracked_compiler_inputs(root)?
        } else {
            collect_observable_workspace_inputs(root)?
        };
        let source = ArtifactStoreV4SourceSnapshot::capture_selected(
            root,
            selected,
            MAX_ARTIFACTSTORE_V4_SOURCE_SNAPSHOT_BYTES,
        )?;
        let build = ArtifactStoreV4BuildSnapshot::new(
            source.identity_sha256.clone(),
            collect_current_linux_build_inputs(root, &source, producer)?,
        )?;
        Self::new(source, build)
    }

    #[must_use]
    pub const fn source(&self) -> &ArtifactStoreV4SourceSnapshot {
        &self.source
    }

    #[must_use]
    pub const fn build(&self) -> &ArtifactStoreV4BuildSnapshot {
        &self.build
    }

    pub(crate) fn binding(&self) -> Result<ArtifactStoreV4SourceBuildBinding, GauntletError> {
        ArtifactStoreV4SourceBuildBinding::new(&self.source, &self.build)
    }
}

/// File-kind witness for one compiler-visible source input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactStoreV4SourceEntryKind {
    File,
    Symlink,
}

/// Why the compiler-visible source entry must be part of the snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactStoreV4SourceInclusionReason {
    Tracked,
    Untracked,
    IgnoredGenerated,
    WorkspaceMember,
    PathDependency,
    CargoLock,
    CargoConfig,
    ToolchainConfig,
    TargetConfig,
    BuildScriptInput,
    BuildScriptOutput,
}

/// One canonical, content-addressed compiler-visible source input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactStoreV4SourceEntry {
    pub relative_path: String,
    pub kind: ArtifactStoreV4SourceEntryKind,
    pub inclusion_reasons: Vec<ArtifactStoreV4SourceInclusionReason>,
    pub mode: u32,
    pub byte_len: u64,
    pub sha256: String,
    pub symlink_target: Option<String>,
    pub resolved_target_path: Option<String>,
}

/// Immutable ordered source-input witness for `ArtifactStore` v4.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactStoreV4SourceSnapshot {
    pub schema_version: u32,
    pub entries: Vec<ArtifactStoreV4SourceEntry>,
    pub identity_sha256: String,
}

impl ArtifactStoreV4SourceSnapshot {
    /// Construct and bind a canonical snapshot from descriptor-stable caller inputs.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError`] when the entries do not canonicalize into a
    /// valid snapshot, or when the derived domain-separated identity fails the
    /// same revalidation [`Self::validate`] applies to a decoded snapshot.
    pub fn new(entries: Vec<ArtifactStoreV4SourceEntry>) -> Result<Self, GauntletError> {
        let mut snapshot = Self {
            schema_version: ARTIFACTSTORE_V4_SOURCE_SNAPSHOT_SCHEMA_VERSION,
            entries,
            identity_sha256: String::new(),
        };
        snapshot.validate_entries()?;
        snapshot.identity_sha256 = snapshot.computed_identity_sha256()?;
        Ok(snapshot)
    }

    /// Revalidate a decoded snapshot and its domain-separated identity.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError`] when the schema version is not the expected
    /// one, when the recorded identity is not a lowercase SHA-256, or when
    /// recomputing the identity over the entries does not reproduce it.
    pub fn validate(&self) -> Result<(), GauntletError> {
        if self.schema_version != ARTIFACTSTORE_V4_SOURCE_SNAPSHOT_SCHEMA_VERSION
            || !is_lower_sha256(&self.identity_sha256)
        {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "ArtifactStore v4 source snapshot has an invalid schema or identity"
                    .to_owned(),
            });
        }
        self.validate_entries()?;
        if self.identity_sha256 != self.computed_identity_sha256()? {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason:
                    "ArtifactStore v4 source snapshot identity does not match canonical entries"
                        .to_owned(),
            });
        }
        Ok(())
    }

    /// Capture selected compiler-visible inputs through descriptor-stable reads.
    ///
    /// The caller must provide the complete compiler-visible path set. This
    /// primitive rejects any selected input that is not a regular file or an
    /// in-snapshot symlink to a captured regular file.
    ///
    /// # Errors
    ///
    /// Returns an error for ambiguous paths, unsupported file kinds, invalid
    /// inclusion reasons, an unstable input read, or a symlink that escapes
    /// the selected source tree.
    pub fn capture_selected(
        root: &Path,
        selected: BTreeMap<String, Vec<ArtifactStoreV4SourceInclusionReason>>,
        max_entry_bytes: u64,
    ) -> Result<Self, GauntletError> {
        if selected.is_empty() || max_entry_bytes == 0 {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason:
                    "ArtifactStore v4 source capture requires selected inputs and a byte budget"
                        .to_owned(),
            });
        }
        let root_directory = PinnedDirectory::open_path(root)?;
        let mut entries = Vec::with_capacity(selected.len());
        for (relative_path, inclusion_reasons) in selected {
            if !is_canonical_source_relative_path(&relative_path) {
                return Err(GauntletError::InvalidPreparedArtifact {
                    reason: "ArtifactStore v4 source capture received an ambiguous selected path"
                        .to_owned(),
                });
            }
            entries.push(root_directory.capture_v4_source_entry(
                &relative_path,
                inclusion_reasons,
                max_entry_bytes,
            )?);
        }
        Self::new(entries)
    }

    fn validate_entries(&self) -> Result<(), GauntletError> {
        let mut previous = None;
        for entry in &self.entries {
            if !is_canonical_source_relative_path(&entry.relative_path)
                || !is_lower_sha256(&entry.sha256)
                || entry.inclusion_reasons.is_empty()
                || entry
                    .inclusion_reasons
                    .windows(2)
                    .any(|pair| pair[0] >= pair[1])
                || previous.is_some_and(|path: &str| path >= entry.relative_path.as_str())
            {
                return Err(GauntletError::InvalidPreparedArtifact {
                    reason: "ArtifactStore v4 source snapshot entries are not canonical and strictly ordered".to_owned(),
                });
            }
            match (&entry.kind, &entry.symlink_target) {
                (ArtifactStoreV4SourceEntryKind::File, None)
                    if entry.resolved_target_path.is_none() => {}
                (ArtifactStoreV4SourceEntryKind::Symlink, Some(target))
                    if !target.is_empty()
                        && entry
                            .resolved_target_path
                            .as_deref()
                            .is_some_and(is_canonical_source_relative_path) => {}
                _ => {
                    return Err(GauntletError::InvalidPreparedArtifact {
                        reason:
                            "ArtifactStore v4 source snapshot file kind and symlink target disagree"
                                .to_owned(),
                    });
                }
            }
            previous = Some(entry.relative_path.as_str());
        }
        for entry in &self.entries {
            let Some(resolved_target_path) = entry.resolved_target_path.as_deref() else {
                continue;
            };
            let target = self
                .entries
                .binary_search_by(|candidate| {
                    candidate.relative_path.as_str().cmp(resolved_target_path)
                })
                .ok()
                .and_then(|index| self.entries.get(index));
            if !matches!(
                target.map(|target| target.kind),
                Some(ArtifactStoreV4SourceEntryKind::File)
            ) {
                return Err(GauntletError::InvalidPreparedArtifact {
                    reason:
                        "ArtifactStore v4 source snapshot symlink target is not a captured file"
                            .to_owned(),
                });
            }
        }
        Ok(())
    }

    fn computed_identity_sha256(&self) -> Result<String, GauntletError> {
        let bytes = serialize_json_bounded(
            &(self.schema_version, &self.entries),
            MAX_ARTIFACTSTORE_V4_SOURCE_SNAPSHOT_BYTES,
            "ArtifactStore v4 source snapshot exceeds its byte budget",
        )?;
        let mut hasher = Sha256::new();
        hasher.update(ARTIFACTSTORE_V4_SOURCE_SNAPSHOT_HASH_DOMAIN);
        hasher.update(bytes);
        Ok(lower_hex(&hasher.finalize()))
    }
}

fn is_canonical_source_relative_path(path: &str) -> bool {
    !path.is_empty()
        && !path.contains('\\')
        && path
            .split('/')
            .all(|component| !component.is_empty() && component != "." && component != "..")
}

fn resolve_source_symlink_target(
    source_path: &str,
    raw_target: &str,
) -> Result<String, GauntletError> {
    if raw_target.is_empty() || raw_target.starts_with('/') || raw_target.contains('\\') {
        return Err(GauntletError::InvalidPreparedArtifact {
            reason:
                "ArtifactStore v4 source symlink target is ambiguous or escapes the source root"
                    .to_owned(),
        });
    }
    let mut components = source_path
        .rsplit_once('/')
        .map_or_else(Vec::new, |(parent, _)| {
            parent.split('/').map(str::to_owned).collect()
        });
    for component in raw_target.split('/') {
        match component {
            "" | "." => {
                return Err(GauntletError::InvalidPreparedArtifact {
                    reason: "ArtifactStore v4 source symlink target is not canonical".to_owned(),
                });
            }
            ".." => {
                if components.pop().is_none() {
                    return Err(GauntletError::InvalidPreparedArtifact {
                        reason: "ArtifactStore v4 source symlink target escapes the source root"
                            .to_owned(),
                    });
                }
            }
            component if component.contains('\\') => {
                return Err(GauntletError::InvalidPreparedArtifact {
                    reason: "ArtifactStore v4 source symlink target is not canonical".to_owned(),
                });
            }
            component => components.push(component.to_owned()),
        }
    }
    let resolved = components.join("/");
    if !is_canonical_source_relative_path(&resolved) {
        return Err(GauntletError::InvalidPreparedArtifact {
            reason: "ArtifactStore v4 source symlink target is not a source-relative path"
                .to_owned(),
        });
    }
    Ok(resolved)
}

/// Class of compiler-visible Build input bound into `ArtifactStore` v4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactStoreV4BuildInputKind {
    CargoLock,
    RegistryChecksum,
    GitDependency,
    Toolchain,
    Compiler,
    Linker,
    TargetConfig,
    CargoConfig,
    Environment,
    BuildScriptInput,
    BuildScriptOutput,
    GeneratedSource,
    FeatureSelection,
    Profile,
    Rustflags,
    Executable,
    DebugMetadata,
}

/// Exact canonical bytes for one compiler-visible Build input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactStoreV4BuildInput {
    pub key: String,
    pub kind: ArtifactStoreV4BuildInputKind,
    pub canonical_bytes: Vec<u8>,
    pub sha256: String,
}

/// Immutable Build object bound to one immutable Source object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactStoreV4BuildSnapshot {
    pub schema_version: u32,
    pub source_identity_sha256: String,
    pub inputs: Vec<ArtifactStoreV4BuildInput>,
    pub identity_sha256: String,
}

impl ArtifactStoreV4BuildSnapshot {
    /// Construct a byte-exact Build snapshot bound to an admitted Source identity.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed source identities, unordered inputs, or
    /// an input digest that does not bind its canonical bytes.
    pub fn new(
        source_identity_sha256: String,
        inputs: Vec<ArtifactStoreV4BuildInput>,
    ) -> Result<Self, GauntletError> {
        let mut snapshot = Self {
            schema_version: ARTIFACTSTORE_V4_BUILD_SNAPSHOT_SCHEMA_VERSION,
            source_identity_sha256,
            inputs,
            identity_sha256: String::new(),
        };
        snapshot.validate_inputs()?;
        snapshot.identity_sha256 = snapshot.computed_identity_sha256()?;
        Ok(snapshot)
    }

    /// Revalidate a decoded Build snapshot and all byte bindings.
    ///
    /// # Errors
    ///
    /// Returns an error for a schema, source binding, input, or identity mismatch.
    pub fn validate(&self) -> Result<(), GauntletError> {
        if self.schema_version != ARTIFACTSTORE_V4_BUILD_SNAPSHOT_SCHEMA_VERSION
            || !is_lower_sha256(&self.source_identity_sha256)
            || !is_lower_sha256(&self.identity_sha256)
        {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "ArtifactStore v4 Build snapshot has an invalid schema or identity"
                    .to_owned(),
            });
        }
        self.validate_inputs()?;
        if self.identity_sha256 != self.computed_identity_sha256()? {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "ArtifactStore v4 Build snapshot identity does not match canonical inputs"
                    .to_owned(),
            });
        }
        Ok(())
    }

    fn validate_inputs(&self) -> Result<(), GauntletError> {
        let mut previous = None;
        for input in &self.inputs {
            if !is_canonical_build_input_key(&input.key)
                || !is_lower_sha256(&input.sha256)
                || input.sha256 != lower_hex(&Sha256::digest(&input.canonical_bytes))
                || previous.is_some_and(|key: &str| key >= input.key.as_str())
            {
                return Err(GauntletError::InvalidPreparedArtifact {
                    reason: "ArtifactStore v4 Build inputs are not canonical and strictly ordered"
                        .to_owned(),
                });
            }
            previous = Some(input.key.as_str());
        }
        Ok(())
    }

    fn computed_identity_sha256(&self) -> Result<String, GauntletError> {
        let bytes = serialize_json_bounded(
            &(
                self.schema_version,
                &self.source_identity_sha256,
                &self.inputs,
            ),
            MAX_ARTIFACTSTORE_V4_BUILD_SNAPSHOT_BYTES,
            "ArtifactStore v4 Build snapshot exceeds its byte budget",
        )?;
        let mut hasher = Sha256::new();
        hasher.update(ARTIFACTSTORE_V4_BUILD_SNAPSHOT_HASH_DOMAIN);
        hasher.update(bytes);
        Ok(lower_hex(&hasher.finalize()))
    }
}

fn is_canonical_build_input_key(key: &str) -> bool {
    !key.is_empty()
        && key.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b'/' | b':')
        })
}

fn collect_tracked_compiler_inputs(
    root: &Path,
) -> Result<BTreeMap<String, Vec<ArtifactStoreV4SourceInclusionReason>>, GauntletError> {
    let mut command = Command::new("git");
    command.arg("-C").arg(root).args(["ls-files", "-z"]);
    for (name, _) in std::env::vars_os() {
        if name.as_encoded_bytes().starts_with(b"GIT_") {
            command.env_remove(name);
        }
    }
    let output = command
        .output()
        .map_err(|error| GauntletError::InvalidPreparedArtifact {
            reason: format!("ArtifactStore v4 cannot enumerate tracked compiler inputs: {error}"),
        })?;
    if !output.status.success() {
        return Err(GauntletError::InvalidPreparedArtifact {
            reason: "ArtifactStore v4 cannot enumerate tracked compiler inputs".to_owned(),
        });
    }

    let mut selected = BTreeMap::new();
    for raw_path in output.stdout.split(|byte| *byte == 0) {
        if raw_path.is_empty() {
            continue;
        }
        let path =
            std::str::from_utf8(raw_path).map_err(|_| GauntletError::InvalidPreparedArtifact {
                reason: "ArtifactStore v4 tracked compiler input is not UTF-8".to_owned(),
            })?;
        if !is_canonical_source_relative_path(path) {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "ArtifactStore v4 tracked compiler input path is ambiguous".to_owned(),
            });
        }
        selected.insert(path.to_owned(), compiler_input_reasons(path));
    }
    if selected.is_empty() || !selected.contains_key("Cargo.lock") {
        return Err(GauntletError::InvalidPreparedArtifact {
            reason: "ArtifactStore v4 tracked compiler input set is missing Cargo.lock".to_owned(),
        });
    }
    Ok(selected)
}

fn collect_observable_workspace_inputs(
    root: &Path,
) -> Result<BTreeMap<String, Vec<ArtifactStoreV4SourceInclusionReason>>, GauntletError> {
    let mut selected = BTreeMap::new();
    collect_observable_workspace_inputs_at(root, Path::new(""), &mut selected)?;
    if selected.is_empty() || !selected.contains_key("Cargo.lock") {
        return Err(GauntletError::InvalidPreparedArtifact {
            reason: "ArtifactStore v4 observable workspace input set is missing Cargo.lock"
                .to_owned(),
        });
    }
    Ok(selected)
}

fn collect_observable_workspace_inputs_at(
    root: &Path,
    relative_directory: &Path,
    selected: &mut BTreeMap<String, Vec<ArtifactStoreV4SourceInclusionReason>>,
) -> Result<(), GauntletError> {
    for child in std::fs::read_dir(root.join(relative_directory))? {
        let child = child?;
        let name = child.file_name();
        if is_non_compiler_workspace_directory(relative_directory, &name) {
            continue;
        }
        let relative_path = relative_directory.join(&name);
        let relative_path =
            relative_path
                .to_str()
                .ok_or_else(|| GauntletError::InvalidPreparedArtifact {
                    reason: "ArtifactStore v4 observable workspace input is not UTF-8".to_owned(),
                })?;
        if !is_canonical_source_relative_path(relative_path) {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "ArtifactStore v4 observable workspace input path is ambiguous".to_owned(),
            });
        }
        let file_type = child.file_type()?;
        if file_type.is_dir() {
            collect_observable_workspace_inputs_at(root, Path::new(relative_path), selected)?;
        } else if file_type.is_file() || file_type.is_symlink() {
            selected.insert(
                relative_path.to_owned(),
                observable_compiler_input_reasons(relative_path),
            );
        } else {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "ArtifactStore v4 observable workspace input has an unsupported file kind"
                    .to_owned(),
            });
        }
    }
    Ok(())
}

fn is_non_compiler_workspace_directory(relative_directory: &Path, name: &OsStr) -> bool {
    let Some(name) = name.to_str() else {
        return false;
    };
    (relative_directory.as_os_str().is_empty() && name.starts_with('.') && name != ".cargo")
        || name == ".git"
        || name == ".beads"
        || name == ".scratch"
        || name == "target"
        || name == "~"
        || name.starts_with(".rch-target")
        || name.starts_with(".worktree")
}

fn compiler_input_reasons(path: &str) -> Vec<ArtifactStoreV4SourceInclusionReason> {
    let mut reasons = BTreeSet::from([ArtifactStoreV4SourceInclusionReason::Tracked]);
    extend_compiler_input_reasons(path, &mut reasons);
    reasons.into_iter().collect()
}

fn observable_compiler_input_reasons(path: &str) -> Vec<ArtifactStoreV4SourceInclusionReason> {
    let mut reasons = BTreeSet::from([ArtifactStoreV4SourceInclusionReason::Untracked]);
    extend_compiler_input_reasons(path, &mut reasons);
    reasons.into_iter().collect()
}

fn extend_compiler_input_reasons(
    path: &str,
    reasons: &mut BTreeSet<ArtifactStoreV4SourceInclusionReason>,
) {
    if path == "Cargo.lock" {
        reasons.insert(ArtifactStoreV4SourceInclusionReason::CargoLock);
    }
    if path == "rust-toolchain.toml" || path == "rust-toolchain" {
        reasons.insert(ArtifactStoreV4SourceInclusionReason::ToolchainConfig);
    }
    if path == ".cargo/config" || path == ".cargo/config.toml" {
        reasons.insert(ArtifactStoreV4SourceInclusionReason::CargoConfig);
        reasons.insert(ArtifactStoreV4SourceInclusionReason::TargetConfig);
    }
    if path == "Cargo.toml" || path.ends_with("/Cargo.toml") {
        reasons.insert(ArtifactStoreV4SourceInclusionReason::WorkspaceMember);
        reasons.insert(ArtifactStoreV4SourceInclusionReason::PathDependency);
    }
    if path == "build.rs" || path.ends_with("/build.rs") {
        reasons.insert(ArtifactStoreV4SourceInclusionReason::BuildScriptInput);
    }
}

fn collect_current_linux_build_inputs(
    root: &Path,
    source: &ArtifactStoreV4SourceSnapshot,
    producer: &GauntletProducerBuildIdentity,
) -> Result<Vec<ArtifactStoreV4BuildInput>, GauntletError> {
    producer.validate_stored_v2()?;
    reject_unbound_runtime_build_overrides()?;

    let mut inputs = BTreeMap::new();
    inputs.insert(
        "cargo-lock".to_owned(),
        build_input(
            ArtifactStoreV4BuildInputKind::CargoLock,
            read_source_file_bound_to_snapshot(root, source, "Cargo.lock")?,
        ),
    );
    for path in ["rust-toolchain.toml", "rust-toolchain"] {
        if source
            .entries
            .iter()
            .any(|entry| entry.relative_path == path)
        {
            inputs.insert(
                format!("toolchain/{path}"),
                build_input(
                    ArtifactStoreV4BuildInputKind::Toolchain,
                    read_source_file_bound_to_snapshot(root, source, path)?,
                ),
            );
        }
    }
    for path in [".cargo/config", ".cargo/config.toml"] {
        if source
            .entries
            .iter()
            .any(|entry| entry.relative_path == path)
        {
            let bytes = read_source_file_bound_to_snapshot(root, source, path)?;
            inputs.insert(
                format!("cargo-config/{path}"),
                build_input(ArtifactStoreV4BuildInputKind::CargoConfig, bytes.clone()),
            );
            inputs.insert(
                format!("target-config/{path}"),
                build_input(ArtifactStoreV4BuildInputKind::TargetConfig, bytes),
            );
        }
    }
    inputs.insert(
        "compiler/rustc-vv".to_owned(),
        build_input(
            ArtifactStoreV4BuildInputKind::Compiler,
            decode_lower_hex(&producer.rustc_version_verbose_hex).ok_or_else(|| {
                GauntletError::InvalidPreparedArtifact {
                    reason: "ArtifactStore v4 producer rustc bytes are malformed".to_owned(),
                }
            })?,
        ),
    );
    inputs.insert(
        "environment/allowlist".to_owned(),
        build_input(
            ArtifactStoreV4BuildInputKind::Environment,
            serde_json::to_vec(&[
                "RUSTFLAGS=unset",
                "CARGO_ENCODED_RUSTFLAGS=unset",
                "RUSTC_LINKER=unset",
            ])?,
        ),
    );
    inputs.insert(
        "features/enabled".to_owned(),
        build_input(
            ArtifactStoreV4BuildInputKind::FeatureSelection,
            producer.enabled_features.join("\n").into_bytes(),
        ),
    );
    inputs.insert(
        "profile/cargo".to_owned(),
        build_input(
            ArtifactStoreV4BuildInputKind::Profile,
            producer.cargo_profile.as_bytes().to_vec(),
        ),
    );
    inputs.insert(
        "rustflags/absent".to_owned(),
        build_input(ArtifactStoreV4BuildInputKind::Rustflags, b"absent".to_vec()),
    );
    inputs.insert(
        "target/triple".to_owned(),
        build_input(
            ArtifactStoreV4BuildInputKind::TargetConfig,
            producer.target_triple.as_bytes().to_vec(),
        ),
    );
    inputs.insert(
        "executable/linux-procfs-running-image".to_owned(),
        linux_running_image_build_input(producer)?,
    );
    inputs.insert(
        "debug/producer-build-identity".to_owned(),
        build_input(
            ArtifactStoreV4BuildInputKind::DebugMetadata,
            serde_json::to_vec(producer)?,
        ),
    );
    inputs.insert(
        "provenance/source-verification".to_owned(),
        build_input(
            ArtifactStoreV4BuildInputKind::DebugMetadata,
            serde_json::to_vec(&serde_json::json!({
                "source_git_dirty": producer.source_git_dirty,
                "source_git_revision": producer.source_git_revision,
                "source_verification": producer.source_verification,
            }))?,
        ),
    );

    Ok(inputs
        .into_iter()
        .map(|(key, mut input)| {
            input.key = key;
            input
        })
        .collect())
}

fn reject_unbound_runtime_build_overrides() -> Result<(), GauntletError> {
    for name in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC_LINKER"] {
        if std::env::var_os(name).is_some_and(|value| !value.is_empty()) {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: format!(
                    "ArtifactStore v4 refuses unbound runtime compiler override {name}"
                ),
            });
        }
    }
    Ok(())
}

fn read_source_file_bound_to_snapshot(
    root: &Path,
    source: &ArtifactStoreV4SourceSnapshot,
    relative_path: &str,
) -> Result<Vec<u8>, GauntletError> {
    let entry = source
        .entries
        .iter()
        .find(|entry| entry.relative_path == relative_path)
        .ok_or_else(|| GauntletError::InvalidPreparedArtifact {
            reason: format!(
                "ArtifactStore v4 source snapshot omits required input {relative_path}"
            ),
        })?;
    if entry.kind != ArtifactStoreV4SourceEntryKind::File {
        return Err(GauntletError::InvalidPreparedArtifact {
            reason: format!(
                "ArtifactStore v4 required input {relative_path} is not a regular file"
            ),
        });
    }
    let bytes = std::fs::read(root.join(relative_path))?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) != entry.byte_len
        || lower_hex(&Sha256::digest(&bytes)) != entry.sha256
    {
        return Err(GauntletError::InvalidPreparedArtifact {
            reason: format!("ArtifactStore v4 source input {relative_path} changed after capture"),
        });
    }
    Ok(bytes)
}

fn build_input(
    kind: ArtifactStoreV4BuildInputKind,
    canonical_bytes: Vec<u8>,
) -> ArtifactStoreV4BuildInput {
    ArtifactStoreV4BuildInput {
        key: String::new(),
        kind,
        sha256: lower_hex(&Sha256::digest(&canonical_bytes)),
        canonical_bytes,
    }
}

fn linux_running_image_build_input(
    producer: &GauntletProducerBuildIdentity,
) -> Result<ArtifactStoreV4BuildInput, GauntletError> {
    let (sha256, byte_len, verification) = current_executable_identity()?;
    if !producer
        .target_triple
        .split('-')
        .any(|component| component == "linux")
        || verification != GauntletExecutableVerification::ProcfsRunningImage
        || producer.executable_verification != GauntletExecutableVerification::ProcfsRunningImage
        || producer.executable_sha256 != sha256
        || producer.executable_byte_len != byte_len
    {
        return Err(GauntletError::InvalidPreparedArtifact {
            reason: "ArtifactStore v4 Linux executable receipt does not bind the running /proc/self/exe image".to_owned(),
        });
    }
    Ok(build_input(
        ArtifactStoreV4BuildInputKind::Executable,
        serde_json::to_vec(&serde_json::json!({
            "path": "/proc/self/exe",
            "sha256": sha256,
            "byte_len": byte_len,
            "verification": verification,
        }))?,
    ))
}

/// Mutable run provenance referencing one immutable object hash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    #[cfg(test)]
    enforce_live_source_checkout: bool,
}

/// Write-last, content-addressed sentinel for a completed campaign.
///
/// This provides relational integrity between the immutable report and the
/// durable completion state. It does not authenticate a filesystem against an
/// adversary capable of coherently rewriting every report, receipt, and
/// object; release publication must add an external CI attestation for that
/// threat model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CampaignCompletionReceipt {
    schema_version: u32,
    run_id: String,
    report_identity_sha256: String,
}

impl CampaignCompletionReceipt {
    fn new(run_id: &str, report_identity_sha256: String) -> Result<Self, GauntletError> {
        let receipt = Self {
            schema_version: CAMPAIGN_COMPLETION_RECEIPT_V1_SCHEMA_VERSION,
            run_id: run_id.to_owned(),
            report_identity_sha256,
        };
        receipt.validate(run_id, &receipt.report_identity_sha256)?;
        Ok(receipt)
    }

    fn validate(
        &self,
        expected_run_id: &str,
        expected_report_identity_sha256: &str,
    ) -> Result<(), GauntletError> {
        validate_run_id(&self.run_id)?;
        if self.schema_version != CAMPAIGN_COMPLETION_RECEIPT_V1_SCHEMA_VERSION
            || self.run_id != expected_run_id
            || !is_lower_hex_len(&self.report_identity_sha256, 64)
            || self.report_identity_sha256 != expected_report_identity_sha256
        {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "campaign completion receipt is malformed or bound to another report"
                    .to_owned(),
            });
        }
        Ok(())
    }

    fn canonical_bytes(&self) -> Result<Vec<u8>, GauntletError> {
        serialize_json_bounded(
            self,
            MAX_CAMPAIGN_COMPLETION_RECEIPT_BYTES,
            "campaign completion receipt exceeds its durable file-size budget",
        )
    }

    fn identity_sha256(&self) -> Result<String, GauntletError> {
        let mut hasher = Sha256::new();
        hasher.update(CAMPAIGN_COMPLETION_RECEIPT_V1_HASH_DOMAIN);
        hasher.update(self.canonical_bytes()?);
        Ok(lower_hex(&hasher.finalize()))
    }

    fn file_name(&self) -> Result<OsString, GauntletError> {
        Ok(OsString::from(format!(
            "completion-{}-{}.json",
            self.report_identity_sha256,
            self.identity_sha256()?
        )))
    }
}

/// Build the exact durable completion marker for a test-owned campaign report.
///
/// This exposes receipt construction only to in-crate tests that need to prove
/// deeper replay validation after a coherent hostile rewrite. It grants no
/// authority and is absent from non-test builds.
#[cfg(all(test, feature = "tantivy-oracle"))]
#[expect(
    clippy::redundant_pub_crate,
    reason = "the authority-bearing fixture is intentionally limited to the parent test boundary"
)]
pub(super) fn campaign_completion_receipt_fixture(
    report: &CampaignReport,
) -> Result<(OsString, Vec<u8>), GauntletError> {
    let receipt =
        CampaignCompletionReceipt::new(&report.run_id, report.report_hash_unchecked_fixture()?)?;
    Ok((receipt.file_name()?, receipt.canonical_bytes()?))
}

/// Self-described execution role preserved by durable integrity checks.
///
/// This value is not an admission capability. It records whether the producer
/// used a diagnostic or typed built-in execution path, but serialized bytes can
/// reproduce either claim. Current objects and reports may self-describe an
/// [`ArtifactTrustCeiling::IntegrityOnly`] ceiling, but only the opaque
/// [`IntegrityCheckedCampaign`] proves that relational validation actually ran.
/// A separate non-serializable, externally authenticated receipt chain is
/// required for authority beyond relational integrity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactExecutionRole {
    /// Decode-only sentinel for campaign schemas that predate role binding.
    #[default]
    LegacyMissing,
    Diagnostic,
    /// Typed built-in execution claim; never admission authority by itself.
    BuiltInExecution,
}

/// Self-described maximum trust permitted by a schema.
///
/// This value never proves that validation occurred and must not be accepted
/// as an achieved-trust capability.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactTrustCeiling {
    /// Historical bytes without the current producer/integrity contract.
    #[default]
    UnauthenticatedLegacy,
    /// Domain-separated relational integrity with no external authentication.
    IntegrityOnly,
}

/// A completed campaign returned only after the store has replayed every
/// durable relational-integrity link.
///
/// The private field prevents raw public DTOs from being promoted by
/// construction or Serde. This wrapper still carries no external
/// authentication or release-admission authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegrityCheckedCampaign {
    report: CampaignReport,
}

impl IntegrityCheckedCampaign {
    /// Borrow the validated report DTO without discarding the wrapper.
    #[must_use]
    pub const fn report(&self) -> &CampaignReport {
        &self.report
    }
}

/// Load the immutable `CampaignReport` V7 replay receipt shipped with this crate.
///
/// This diagnostic receipt is read from compiled-in bytes, rather than
/// regenerated from the caller's checkout. The loader checks both the fixture
/// bytes and domain-separated report identity before exposing the report.
///
/// # Errors
///
/// Returns an error if the embedded bytes, their canonical JSON encoding, or
/// their `CampaignReport` V7 identity do not match the pinned receipt.
pub fn pinned_campaign_report_v7() -> Result<IntegrityCheckedCampaign, GauntletError> {
    let report = crate::runner::load_pinned_campaign_report_v7()?;
    report.validate_contract()?;
    Ok(IntegrityCheckedCampaign { report })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CampaignEvidenceValidation {
    Stored,
    Creation,
}

impl Default for ArtifactStore {
    fn default() -> Self {
        Self::new(".gauntlet")
    }
}

impl ArtifactStore {
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            #[cfg(test)]
            enforce_live_source_checkout: true,
        }
    }

    /// Bind canonical source inputs into an `ArtifactStore` v4 snapshot.
    ///
    /// This only establishes the immutable Source object. It deliberately
    /// grants no execution or release authority until the v4 Build and
    /// supervisor-authentication layers have verified their own bindings.
    ///
    /// # Errors
    ///
    /// Returns an error when an entry is malformed, ambiguously named, or
    /// cannot be bound to the canonical source identity.
    pub fn bind_v4_source_snapshot(
        entries: Vec<ArtifactStoreV4SourceEntry>,
    ) -> Result<ArtifactStoreV4SourceSnapshot, GauntletError> {
        ArtifactStoreV4SourceSnapshot::new(entries)
    }

    /// Bind compiler-visible Build inputs to an `ArtifactStore` v4 Source object.
    ///
    /// # Errors
    ///
    /// Returns an error when the Source identity or any Build input is malformed.
    pub fn bind_v4_build_snapshot(
        source: &ArtifactStoreV4SourceSnapshot,
        inputs: Vec<ArtifactStoreV4BuildInput>,
    ) -> Result<ArtifactStoreV4BuildSnapshot, GauntletError> {
        source.validate()?;
        ArtifactStoreV4BuildSnapshot::new(source.identity_sha256.clone(), inputs)
    }

    /// Persist the immutable v4 Source and Build objects under their complete
    /// domain-separated identities. Existing bytes are accepted only when
    /// exactly equal; neither object is replaceable.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid chain, a malformed existing object, or
    /// a content-address collision.
    pub fn persist_v4_source_build_snapshots(
        &self,
        snapshots: &ArtifactStoreV4SourceBuildSnapshots,
    ) -> Result<ArtifactStoreV4SourceBuildBinding, GauntletError> {
        let binding = snapshots.binding()?;
        let source_bytes = serialize_json_bounded(
            snapshots.source(),
            MAX_ARTIFACTSTORE_V4_SOURCE_SNAPSHOT_BYTES,
            "ArtifactStore v4 Source snapshot exceeds its durable byte budget",
        )?;
        let build_bytes = serialize_json_bounded(
            snapshots.build(),
            MAX_ARTIFACTSTORE_V4_BUILD_SNAPSHOT_BYTES,
            "ArtifactStore v4 Build snapshot exceeds its durable byte budget",
        )?;
        let root = PinnedDirectory::ensure_path(&self.root)?;
        let v4 = root.ensure_child(OsStr::new("v4"))?;
        let sources = v4.ensure_child(OsStr::new("sources"))?;
        let builds = v4.ensure_child(OsStr::new("builds"))?;
        sources.write_once_or_verify(
            OsStr::new(&format!("{}.json", binding.source_identity_sha256)),
            &source_bytes,
            ExistingFileKind::Object,
            MAX_ARTIFACTSTORE_V4_SOURCE_SNAPSHOT_BYTES,
        )?;
        builds.write_once_or_verify(
            OsStr::new(&format!("{}.json", binding.build_identity_sha256)),
            &build_bytes,
            ExistingFileKind::Object,
            MAX_ARTIFACTSTORE_V4_BUILD_SNAPSHOT_BYTES,
        )?;
        Ok(binding)
    }

    /// Reload one persisted v4 Source-to-Build chain through pinned directory
    /// descriptors and revalidate every identity edge.
    ///
    /// # Errors
    ///
    /// Returns an error if either object is missing, noncanonical, malformed,
    /// or not bound to the requested identities.
    pub fn load_v4_source_build_snapshots(
        &self,
        binding: &ArtifactStoreV4SourceBuildBinding,
    ) -> Result<ArtifactStoreV4SourceBuildSnapshots, GauntletError> {
        binding.validate()?;
        let root = PinnedDirectory::open_path(&self.root)?;
        let v4 = root.open_child(OsStr::new("v4"))?;
        let sources = v4.open_child(OsStr::new("sources"))?;
        let builds = v4.open_child(OsStr::new("builds"))?;
        let source_bytes = sources.read_regular_bounded(
            OsStr::new(&format!("{}.json", binding.source_identity_sha256)),
            MAX_ARTIFACTSTORE_V4_SOURCE_SNAPSHOT_BYTES,
        )?;
        let build_bytes = builds.read_regular_bounded(
            OsStr::new(&format!("{}.json", binding.build_identity_sha256)),
            MAX_ARTIFACTSTORE_V4_BUILD_SNAPSHOT_BYTES,
        )?;
        let source: ArtifactStoreV4SourceSnapshot = serde_json::from_slice(&source_bytes)?;
        let build: ArtifactStoreV4BuildSnapshot = serde_json::from_slice(&build_bytes)?;
        if !canonical_json_matches(&source, &source_bytes)?
            || !canonical_json_matches(&build, &build_bytes)?
        {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "persisted ArtifactStore v4 Source or Build object is noncanonical"
                    .to_owned(),
            });
        }
        let snapshots = ArtifactStoreV4SourceBuildSnapshots::new(source, build)?;
        if snapshots.binding()? != *binding {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "persisted ArtifactStore v4 Source/Build chain does not match its binding"
                    .to_owned(),
            });
        }
        Ok(snapshots)
    }

    #[cfg(test)]
    pub(crate) fn with_test_live_source_bypass(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            enforce_live_source_checkout: false,
        }
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    #[allow(
        clippy::unused_self,
        reason = "the receiver carries test-only bypass state that must not exist in production"
    )]
    pub(crate) fn validate_live_source_checkout_for_creation(
        &self,
        producer: &GauntletProducerBuildIdentity,
    ) -> Result<(), GauntletError> {
        #[cfg(test)]
        if !self.enforce_live_source_checkout {
            // Tests using this constructor exercise relational artifact
            // replay without claiming that the remote test executable is the
            // kernel-held producer captured in the fixture. Structural v2
            // identity validation remains mandatory; production constructors
            // can never disable the live-source check.
            producer.validate_stored_v2()?;
            return Ok(());
        }
        producer.validate_live_source_checkout()
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

    /// Encode a built-in-integrity standalone object and run manifest without
    /// writing files. This test-only helper does not mint terminal authority.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid identities/contracts, a dirty or
    /// unverified producer, unsafe run IDs, or serialization failures.
    #[cfg(test)]
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
            ArtifactExecutionRole::BuiltInExecution,
            &compiled_producer,
            object,
            provenance,
        )
    }

    /// Encode a standalone diagnostic object without claiming built-in
    /// integrity. The compiled producer binding remains exact, while dirty or
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
            ArtifactExecutionRole::Diagnostic,
            &compiled_producer,
            object,
            provenance,
        )
    }

    pub(crate) fn prepare_campaign_case(
        &self,
        campaign_run_id: &str,
        ordinal: usize,
        execution_role: ArtifactExecutionRole,
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
        if execution_role == ArtifactExecutionRole::BuiltInExecution {
            self.validate_live_source_checkout_for_creation(&compiled_producer)?;
        }
        self.prepare_at(
            &run_id,
            run_path,
            PreparedRunLocation::Campaign {
                campaign_run_id: campaign_run_id.to_owned(),
                ordinal,
            },
            true,
            execution_role,
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
        execution_role: ArtifactExecutionRole,
        expected_producer: &GauntletProducerBuildIdentity,
        object: &ArtifactObject,
        provenance: BTreeMap<String, String>,
    ) -> Result<PreparedArtifact, GauntletError> {
        validate_run_id(run_id)?;
        if object.execution_role != execution_role {
            return Err(GauntletError::InvalidContract {
                reason: "artifact persistence execution role does not match the object".to_owned(),
            });
        }
        match (execution_role, &object.oracle_dependency) {
            (ArtifactExecutionRole::LegacyMissing, _) => {
                return Err(GauntletError::InvalidContract {
                    reason: "artifact persistence execution role is missing".to_owned(),
                });
            }
            (
                ArtifactExecutionRole::Diagnostic,
                ArtifactOracleDependency::DiagnosticUnspecified,
            ) => {
                object.validate()?;
                object.comparator_config.validate_contract()?;
                if object.object_schema_version != OBJECT_SCHEMA_VERSION {
                    return Err(GauntletError::InvalidContract {
                        reason:
                            "new diagnostic persistence requires the current artifact object schema"
                                .to_owned(),
                    });
                }
            }
            (
                ArtifactExecutionRole::BuiltInExecution,
                ArtifactOracleDependency::BuiltInTantivy { .. },
            ) => object.validate_current_builtin_integrity()?,
            (ArtifactExecutionRole::Diagnostic, _) => {
                return Err(GauntletError::InvalidContract {
                    reason: "diagnostic persistence requires a diagnostic dependency role"
                        .to_owned(),
                });
            }
            (ArtifactExecutionRole::BuiltInExecution, _) => {
                return Err(GauntletError::InvalidContract {
                    reason: "built-in persistence requires a typed built-in dependency role"
                        .to_owned(),
                });
            }
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

    /// Load a completed campaign only after replaying every durable integrity link.
    ///
    /// The report, reservation, case references, immutable objects, comparator
    /// outcomes, divergence classifications, and mismatch aggregates are all
    /// read through pinned directory descriptors and checked before return.
    /// This proves relational integrity only. It does not authenticate the
    /// producer, authorize caller-selected policy, or mint admission authority.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsafe path, incomplete campaign, noncanonical
    /// file, or any provenance/evidence mismatch.
    pub fn load_integrity_checked_campaign(
        &self,
        run_id: &str,
    ) -> Result<IntegrityCheckedCampaign, GauntletError> {
        validate_run_id(run_id)?;
        let (root, campaign) = self.open_pinned_campaign(run_id)?;
        let report_bytes =
            campaign.read_regular_bounded(OsStr::new("report.json"), MAX_CAMPAIGN_REPORT_BYTES)?;
        require_current_campaign_report_schema(&report_bytes)?;
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
        let report_identity_sha256 = report.report_hash()?;
        let expected_completion = CampaignCompletionReceipt::new(run_id, report_identity_sha256)?;
        let completion_name = expected_completion.file_name()?;
        let completion_bytes = campaign
            .read_regular_bounded(&completion_name, MAX_CAMPAIGN_COMPLETION_RECEIPT_BYTES)?;
        let completion: CampaignCompletionReceipt = serde_json::from_slice(&completion_bytes)?;
        completion.validate(run_id, &expected_completion.report_identity_sha256)?;
        if completion != expected_completion
            || !canonical_json_matches(&completion, &completion_bytes)?
        {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "campaign completion receipt is noncanonical or has the wrong identity"
                    .to_owned(),
            });
        }
        Self::validate_completed_campaign_entries(&campaign, &report, &completion_name)?;
        self.verify_campaign_evidence(
            &root,
            &campaign,
            &report,
            CampaignEvidenceValidation::Stored,
        )?;
        Ok(IntegrityCheckedCampaign { report })
    }

    /// Validate all stored case evidence and atomically publish the sole
    /// campaign-completion marker.
    pub(crate) fn complete_campaign(&self, report: &CampaignReport) -> Result<(), GauntletError> {
        report.validate_creation_environment()?;
        if report.execution_role == ArtifactExecutionRole::BuiltInExecution {
            self.validate_live_source_checkout_for_creation(&report.producer_build_identity)?;
        }
        validate_run_id(&report.run_id)?;
        let (root, campaign) = self.open_pinned_campaign(&report.run_id)?;
        self.verify_campaign_evidence(
            &root,
            &campaign,
            report,
            CampaignEvidenceValidation::Creation,
        )?;
        let report_bytes = serialize_json_bounded(
            report,
            MAX_CAMPAIGN_REPORT_BYTES,
            "campaign report exceeds its durable file-size budget",
        )?;
        let completion = CampaignCompletionReceipt::new(&report.run_id, report.report_hash()?)?;
        let completion_bytes = completion.canonical_bytes()?;
        let completion_name = completion.file_name()?;
        campaign.write_once_or_verify(
            OsStr::new("report.json"),
            &report_bytes,
            ExistingFileKind::Run,
            MAX_CAMPAIGN_REPORT_BYTES,
        )?;
        campaign.write_once_or_verify(
            &completion_name,
            &completion_bytes,
            ExistingFileKind::Run,
            MAX_CAMPAIGN_COMPLETION_RECEIPT_BYTES,
        )?;
        Self::validate_completed_campaign_entries(&campaign, report, &completion_name)
    }

    fn validate_completed_campaign_entries(
        campaign: &PinnedDirectory,
        report: &CampaignReport,
        completion_name: &OsStr,
    ) -> Result<(), GauntletError> {
        let mut expected = std::collections::BTreeSet::from([
            OsString::from("reservation.json"),
            OsString::from("report.json"),
            completion_name.to_owned(),
        ]);
        if report
            .cases
            .iter()
            .any(|result| result.artifact_hash.is_some())
        {
            expected.insert(OsString::from("cases"));
        }
        let observed = campaign.entry_names(expected.len().saturating_add(1))?;
        if observed != expected {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "completed campaign directory contains missing or unexpected entries"
                    .to_owned(),
            });
        }
        Ok(())
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
        validation: CampaignEvidenceValidation,
    ) -> Result<(), GauntletError> {
        if let Some(binding) = &report.v4_source_build_binding {
            self.load_v4_source_build_snapshots(binding)?;
        }
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
            let object_schema_version = require_current_artifact_object_schema(&object_bytes)?;
            let object: ArtifactObject = serde_json::from_slice(&object_bytes)?;
            if object.object_schema_version != object_schema_version {
                return Err(GauntletError::InvalidPreparedArtifact {
                    reason: "artifact object schema changed between preflight and full decode"
                        .to_owned(),
                });
            }
            let object_hash = hash_object_bytes(&object_bytes, object_schema_version)?;
            if !canonical_json_matches(&object, &object_bytes)?
                || object_hash != run_manifest.object_hash
            {
                return Err(GauntletError::InvalidPreparedArtifact {
                    reason: "campaign object bytes or content address are inconsistent".to_owned(),
                });
            }
            if object.execution_role != report.execution_role {
                return Err(GauntletError::InvalidPreparedArtifact {
                    reason: "campaign report execution role does not match its case artifact"
                        .to_owned(),
                });
            }
            match (report.execution_role, validation, &object.oracle_dependency) {
                (
                    ArtifactExecutionRole::Diagnostic,
                    _,
                    ArtifactOracleDependency::DiagnosticUnspecified,
                ) => object.validate()?,
                (
                    ArtifactExecutionRole::BuiltInExecution,
                    CampaignEvidenceValidation::Stored,
                    ArtifactOracleDependency::BuiltInTantivy { .. },
                ) => object.validate_stored_builtin_integrity()?,
                (
                    ArtifactExecutionRole::BuiltInExecution,
                    CampaignEvidenceValidation::Creation,
                    ArtifactOracleDependency::BuiltInTantivy { .. },
                ) => object.validate_current_builtin_integrity()?,
                (
                    ArtifactExecutionRole::LegacyMissing
                    | ArtifactExecutionRole::Diagnostic
                    | ArtifactExecutionRole::BuiltInExecution,
                    _,
                    _,
                ) => {
                    return Err(GauntletError::InvalidPreparedArtifact {
                        reason:
                            "campaign report role does not match its case artifact dependency role"
                                .to_owned(),
                    });
                }
            }
            if object.oracle_dependency != report.oracle_dependency {
                return Err(GauntletError::InvalidPreparedArtifact {
                    reason:
                        "campaign case oracle dependency does not match the reservation and report"
                            .to_owned(),
                });
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
        7 => {
            let mut hasher = Sha256::new();
            hasher.update(HASH_DOMAIN_V7);
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

fn validate_stored_object_schema(schema_version: u32) -> Result<(), String> {
    match schema_version {
        ARTIFACT_OBJECT_V7_SCHEMA_VERSION => Ok(()),
        1..=3 => Err(format!(
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
        6 => Err(
            "artifact v6 provides self-consistency integrity only and no durable role/admission authority; rerun with the current contract"
                .to_owned(),
        ),
        _ => Err(format!(
            "artifact object schema version {schema_version} is unsupported and non-admissible; rerun under the complete current evidence contract"
        )),
    }
}

/// Classify a serialized campaign report before any current-DTO decode.
///
/// # Errors
///
/// Returns an error for malformed JSON, a missing/duplicate/non-integer schema
/// key, or an unknown generation.
pub fn classify_campaign_report_schema(
    bytes: &[u8],
) -> Result<SerializedSchemaDisposition, GauntletError> {
    // A narrow derive-based probe ignores unrelated historical fields while
    // still rejecting a missing, non-integer, or duplicate schema key. This
    // routes every shipped generation before the current DTO is decoded.
    let probe: CampaignReportSchemaProbe =
        serde_json::from_slice(bytes).map_err(|error| GauntletError::InvalidPreparedArtifact {
            reason: format!("campaign report schema preflight failed closed: {error}"),
        })?;
    match probe.schema_version {
        schema_version @ (1 | 2 | 3 | 5) => {
            Ok(SerializedSchemaDisposition::UnauthenticatedLegacy { schema_version })
        }
        schema_version @ 4 => Ok(SerializedSchemaDisposition::ReservedRejected { schema_version }),
        schema_version @ 6 => {
            Ok(SerializedSchemaDisposition::LegacyIntegrityCeiling { schema_version })
        }
        schema_version @ crate::runner::CAMPAIGN_REPORT_V7_SCHEMA_VERSION => {
            Ok(SerializedSchemaDisposition::CurrentIntegrityContractCandidate { schema_version })
        }
        schema_version => Err(GauntletError::InvalidPreparedArtifact {
            reason: format!(
                "campaign report schema version {schema_version} is unsupported and non-admissible"
            ),
        }),
    }
}

/// Classify a serialized artifact object before any current-DTO decode.
///
/// # Errors
///
/// Returns an error for malformed JSON, a missing/duplicate/non-integer schema
/// key, or an unknown generation.
pub fn classify_artifact_object_schema(
    bytes: &[u8],
) -> Result<SerializedSchemaDisposition, GauntletError> {
    let probe: ArtifactObjectSchemaProbe =
        serde_json::from_slice(bytes).map_err(|error| GauntletError::InvalidPreparedArtifact {
            reason: format!("artifact object schema preflight failed closed: {error}"),
        })?;
    match probe.object_schema_version {
        schema_version @ (1 | 2 | 3 | 5) => {
            Ok(SerializedSchemaDisposition::UnauthenticatedLegacy { schema_version })
        }
        schema_version @ 4 => Ok(SerializedSchemaDisposition::ReservedRejected { schema_version }),
        schema_version @ 6 => {
            Ok(SerializedSchemaDisposition::LegacyIntegrityCeiling { schema_version })
        }
        schema_version @ ARTIFACT_OBJECT_V7_SCHEMA_VERSION => {
            Ok(SerializedSchemaDisposition::CurrentIntegrityContractCandidate { schema_version })
        }
        schema_version => Err(GauntletError::InvalidPreparedArtifact {
            reason: format!(
                "artifact object schema version {schema_version} is unsupported and non-admissible"
            ),
        }),
    }
}

fn require_current_campaign_report_schema(bytes: &[u8]) -> Result<(), GauntletError> {
    match classify_campaign_report_schema(bytes)? {
        SerializedSchemaDisposition::CurrentIntegrityContractCandidate { .. } => Ok(()),
        SerializedSchemaDisposition::UnauthenticatedLegacy { schema_version } => {
            Err(GauntletError::InvalidPreparedArtifact {
                reason: format!(
                    "unauthenticated legacy campaign report v{schema_version} is decode-only and non-admissible; rerun the campaign"
                ),
            })
        }
        SerializedSchemaDisposition::ReservedRejected { schema_version } => {
            Err(GauntletError::InvalidPreparedArtifact {
                reason: format!(
                    "reserved pre-policy campaign report v{schema_version} is rejected and cannot be promoted; rerun the campaign"
                ),
            })
        }
        SerializedSchemaDisposition::LegacyIntegrityCeiling { schema_version } => {
            Err(GauntletError::InvalidPreparedArtifact {
                reason: format!(
                    "campaign report v{schema_version} provides self-consistency integrity only, not admission authority; rerun under the current contract"
                ),
            })
        }
    }
}

fn require_current_artifact_object_schema(bytes: &[u8]) -> Result<u32, GauntletError> {
    match classify_artifact_object_schema(bytes)? {
        SerializedSchemaDisposition::CurrentIntegrityContractCandidate { schema_version } => {
            Ok(schema_version)
        }
        SerializedSchemaDisposition::UnauthenticatedLegacy { schema_version } => {
            Err(GauntletError::InvalidPreparedArtifact {
                reason: format!(
                    "unauthenticated legacy artifact v{schema_version} is decode-only and non-admissible; rerun the campaign"
                ),
            })
        }
        SerializedSchemaDisposition::ReservedRejected { schema_version } => {
            Err(GauntletError::InvalidPreparedArtifact {
                reason: format!(
                    "reserved pre-policy artifact v{schema_version} is non-admissible; rerun under the complete current evidence contract"
                ),
            })
        }
        SerializedSchemaDisposition::LegacyIntegrityCeiling { schema_version } => {
            Err(GauntletError::InvalidPreparedArtifact {
                reason: format!(
                    "artifact v{schema_version} provides self-consistency integrity only and no durable execution role; rerun with the current contract"
                ),
            })
        }
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

    fn capture_v4_source_entry(
        &self,
        relative_path: &str,
        inclusion_reasons: Vec<ArtifactStoreV4SourceInclusionReason>,
        max_entry_bytes: u64,
    ) -> Result<ArtifactStoreV4SourceEntry, GauntletError> {
        use rustix::fs::{AtFlags, FileType, statat};

        let (parent, name) = relative_path
            .rsplit_once('/')
            .map_or(("", relative_path), |(parent, name)| (parent, name));
        let mut directory = Self {
            file: self.file.try_clone()?,
            display_path: self.display_path.clone(),
        };
        for component in parent.split('/').filter(|component| !component.is_empty()) {
            directory = directory.open_child(OsStr::new(component))?;
        }
        let name = OsStr::new(name);
        let before = statat(&directory.file, name, AtFlags::SYMLINK_NOFOLLOW)
            .map_err(std::io::Error::from)?;
        match FileType::from_raw_mode(before.st_mode) {
            FileType::RegularFile => {
                let (mode, byte_len, sha256) =
                    directory.read_regular_v4_source_entry(name, max_entry_bytes)?;
                let after = statat(&directory.file, name, AtFlags::SYMLINK_NOFOLLOW)
                    .map_err(std::io::Error::from)?;
                if before.st_dev != after.st_dev
                    || before.st_ino != after.st_ino
                    || before.st_mode != after.st_mode
                    || before.st_size != after.st_size
                    || before.st_mtime != after.st_mtime
                    || before.st_mtime_nsec != after.st_mtime_nsec
                    || before.st_ctime != after.st_ctime
                    || before.st_ctime_nsec != after.st_ctime_nsec
                {
                    return Err(GauntletError::InvalidPreparedArtifact {
                        reason:
                            "ArtifactStore v4 source file changed during descriptor-stable capture"
                                .to_owned(),
                    });
                }
                Ok(ArtifactStoreV4SourceEntry {
                    relative_path: relative_path.to_owned(),
                    kind: ArtifactStoreV4SourceEntryKind::File,
                    inclusion_reasons,
                    mode,
                    byte_len,
                    sha256,
                    symlink_target: None,
                    resolved_target_path: None,
                })
            }
            FileType::Symlink => {
                let target_bytes = rustix::fs::readlinkat(&directory.file, name, Vec::new())
                    .map_err(std::io::Error::from)?;
                let target_bytes = target_bytes.as_bytes();
                if u64::try_from(target_bytes.len()).unwrap_or(u64::MAX) > max_entry_bytes {
                    return Err(GauntletError::InvalidPreparedArtifact {
                        reason: "ArtifactStore v4 source symlink exceeds its byte budget"
                            .to_owned(),
                    });
                }
                let target = std::str::from_utf8(target_bytes).map_err(|_| {
                    GauntletError::InvalidPreparedArtifact {
                        reason: "ArtifactStore v4 source symlink target is not UTF-8".to_owned(),
                    }
                })?;
                let after = statat(&directory.file, name, AtFlags::SYMLINK_NOFOLLOW)
                    .map_err(std::io::Error::from)?;
                if before.st_dev != after.st_dev
                    || before.st_ino != after.st_ino
                    || before.st_mode != after.st_mode
                    || before.st_size != after.st_size
                    || before.st_mtime != after.st_mtime
                    || before.st_mtime_nsec != after.st_mtime_nsec
                    || before.st_ctime != after.st_ctime
                    || before.st_ctime_nsec != after.st_ctime_nsec
                {
                    return Err(GauntletError::InvalidPreparedArtifact {
                        reason: "ArtifactStore v4 source symlink changed during descriptor-stable capture"
                            .to_owned(),
                    });
                }
                Ok(ArtifactStoreV4SourceEntry {
                    relative_path: relative_path.to_owned(),
                    kind: ArtifactStoreV4SourceEntryKind::Symlink,
                    inclusion_reasons,
                    mode: before.st_mode,
                    byte_len: u64::try_from(target_bytes.len()).unwrap_or(u64::MAX),
                    sha256: lower_hex(&Sha256::digest(target_bytes)),
                    symlink_target: Some(target.to_owned()),
                    resolved_target_path: Some(resolve_source_symlink_target(
                        relative_path,
                        target,
                    )?),
                })
            }
            _ => Err(GauntletError::InvalidPreparedArtifact {
                reason: "ArtifactStore v4 source capture rejects non-file compiler inputs"
                    .to_owned(),
            }),
        }
    }

    fn read_regular_v4_source_entry(
        &self,
        name: &OsStr,
        max_entry_bytes: u64,
    ) -> Result<(u32, u64, String), GauntletError> {
        use rustix::fs::{FileType, Mode, OFlags, fstat, openat};

        let descriptor = openat(
            &self.file,
            name,
            OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::empty(),
        )
        .map_err(std::io::Error::from)?;
        let before = fstat(&descriptor).map_err(std::io::Error::from)?;
        let byte_len =
            u64::try_from(before.st_size).map_err(|_| GauntletError::InvalidPreparedArtifact {
                reason: "ArtifactStore v4 source file length is invalid".to_owned(),
            })?;
        if FileType::from_raw_mode(before.st_mode) != FileType::RegularFile
            || byte_len > max_entry_bytes
        {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "ArtifactStore v4 source capture rejects oversized or non-regular files"
                    .to_owned(),
            });
        }
        let mut file = File::from(descriptor);
        let capacity =
            usize::try_from(byte_len).map_err(|_| GauntletError::InvalidPreparedArtifact {
                reason: "ArtifactStore v4 source file length does not fit this platform".to_owned(),
            })?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(capacity).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                format!("unable to reserve bounded source read: {error}"),
            )
        })?;
        (&mut file)
            .take(max_entry_bytes.saturating_add(1))
            .read_to_end(&mut bytes)?;
        let after = fstat(&file).map_err(std::io::Error::from)?;
        if before.st_dev != after.st_dev
            || before.st_ino != after.st_ino
            || before.st_mode != after.st_mode
            || before.st_size != after.st_size
            || before.st_mtime != after.st_mtime
            || before.st_mtime_nsec != after.st_mtime_nsec
            || before.st_ctime != after.st_ctime
            || before.st_ctime_nsec != after.st_ctime_nsec
            || u64::try_from(bytes.len()).unwrap_or(u64::MAX) != byte_len
        {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "ArtifactStore v4 source file changed during descriptor-stable capture"
                    .to_owned(),
            });
        }
        Ok((before.st_mode, byte_len, lower_hex(&Sha256::digest(bytes))))
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

    fn capture_v4_source_entry(
        &self,
        _relative_path: &str,
        _inclusion_reasons: Vec<ArtifactStoreV4SourceInclusionReason>,
        _max_entry_bytes: u64,
    ) -> Result<ArtifactStoreV4SourceEntry, GauntletError> {
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
    use crate::engine::{
        BuiltInEngineProfile, BuiltInEngineProfileReceipt, TANTIVY_ORACLE_CONFIG_HASH,
        quill_config_hash,
    };
    use crate::{
        ComparisonMode, CountState, DifferentialCase, EngineDescriptor, EngineFamily,
        EngineObservation, NativeTieKey, RankedHit, compare_observations,
    };

    fn assert_strict_tagged_round_trip<T>(value: &T)
    where
        T: Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        let encoded = serde_json::to_value(value).expect("serialize tagged artifact value");
        let decoded: T =
            serde_json::from_value(encoded.clone()).expect("deserialize tagged artifact value");
        assert_eq!(&decoded, value);

        let mut with_unknown = encoded;
        with_unknown
            .as_object_mut()
            .expect("tagged artifact value must be an object")
            .insert("future_unbound_field".to_owned(), serde_json::json!(true));
        assert!(
            serde_json::from_value::<T>(with_unknown).is_err(),
            "tagged artifact variant accepted an unknown field: {value:?}"
        );
    }

    fn run_fixture_git(repository: &Path, args: &[&str]) -> String {
        let mut command = std::process::Command::new("git");
        command.arg("-C").arg(repository).args(args);
        for (name, _) in std::env::vars_os() {
            if name.as_encoded_bytes().starts_with(b"GIT_") {
                command.env_remove(name);
            }
        }
        let output = command.output().expect("execute fixture Git command");
        assert!(
            output.status.success(),
            "fixture Git command failed: args={args:?} stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        String::from_utf8(output.stdout)
            .expect("fixture Git output is UTF-8")
            .trim()
            .to_owned()
    }

    fn clean_fixture_repository() -> (tempfile::TempDir, String) {
        let repository = tempfile::tempdir().expect("fixture Git repository");
        run_fixture_git(repository.path(), &["init", "--quiet"]);
        std::fs::write(repository.path().join("tracked.txt"), b"sealed source\n")
            .expect("write tracked fixture");
        run_fixture_git(repository.path(), &["add", "tracked.txt"]);
        run_fixture_git(
            repository.path(),
            &[
                "-c",
                "user.name=Gauntlet Test",
                "-c",
                "user.email=gauntlet@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "sealed fixture",
            ],
        );
        let revision = run_fixture_git(repository.path(), &["rev-parse", "HEAD"]);
        (repository, revision)
    }

    #[test]
    fn artifactstore_v4_source_snapshot_binds_sorted_compiler_visible_inputs() {
        let file_hash = "a".repeat(64);
        let link_hash = "b".repeat(64);
        let source_reasons = vec![
            ArtifactStoreV4SourceInclusionReason::Tracked,
            ArtifactStoreV4SourceInclusionReason::Untracked,
            ArtifactStoreV4SourceInclusionReason::IgnoredGenerated,
            ArtifactStoreV4SourceInclusionReason::WorkspaceMember,
            ArtifactStoreV4SourceInclusionReason::PathDependency,
            ArtifactStoreV4SourceInclusionReason::CargoLock,
            ArtifactStoreV4SourceInclusionReason::CargoConfig,
            ArtifactStoreV4SourceInclusionReason::ToolchainConfig,
            ArtifactStoreV4SourceInclusionReason::TargetConfig,
            ArtifactStoreV4SourceInclusionReason::BuildScriptInput,
            ArtifactStoreV4SourceInclusionReason::BuildScriptOutput,
        ];
        let snapshot = ArtifactStore::bind_v4_source_snapshot(vec![
            ArtifactStoreV4SourceEntry {
                relative_path: "Cargo.lock".to_owned(),
                kind: ArtifactStoreV4SourceEntryKind::File,
                inclusion_reasons: source_reasons.clone(),
                mode: 0o100_644,
                byte_len: 42,
                sha256: file_hash,
                symlink_target: None,
                resolved_target_path: None,
            },
            ArtifactStoreV4SourceEntry {
                relative_path: "crates/current".to_owned(),
                kind: ArtifactStoreV4SourceEntryKind::Symlink,
                inclusion_reasons: vec![ArtifactStoreV4SourceInclusionReason::PathDependency],
                mode: 0o120_777,
                byte_len: 18,
                sha256: link_hash,
                symlink_target: Some("../Cargo.lock".to_owned()),
                resolved_target_path: Some("Cargo.lock".to_owned()),
            },
        ])
        .expect("construct canonical source snapshot");

        snapshot.validate().expect("validate constructed snapshot");
        assert!(is_lower_sha256(&snapshot.identity_sha256));

        for index in 0..source_reasons.len() {
            let mut tampered_reasons = snapshot.clone();
            tampered_reasons.entries[0].inclusion_reasons.remove(index);
            assert!(matches!(
                tampered_reasons.validate(),
                Err(GauntletError::InvalidPreparedArtifact { .. })
            ));
        }

        let mut reordered = snapshot.clone();
        reordered.entries.reverse();
        assert!(matches!(
            reordered.validate(),
            Err(GauntletError::InvalidPreparedArtifact { .. })
        ));

        let mut tampered = snapshot;
        tampered.entries[0].byte_len += 1;
        assert!(matches!(
            tampered.validate(),
            Err(GauntletError::InvalidPreparedArtifact { .. })
        ));

        let mut retargeted = reordered;
        retargeted.entries.reverse();
        retargeted.entries[1].symlink_target = Some("Cargo.lock".to_owned());
        assert!(matches!(
            retargeted.validate(),
            Err(GauntletError::InvalidPreparedArtifact { .. })
        ));
    }

    #[test]
    fn artifactstore_v4_collects_every_tracked_input_with_compiler_reasons() {
        let (repository, _) = clean_fixture_repository();
        std::fs::create_dir_all(repository.path().join(".cargo"))
            .expect("create Cargo configuration directory");
        std::fs::create_dir_all(repository.path().join("crates/example"))
            .expect("create workspace member directory");
        std::fs::write(repository.path().join("Cargo.lock"), b"lock\n").expect("write Cargo lock");
        std::fs::write(
            repository.path().join("rust-toolchain.toml"),
            b"[toolchain]\n",
        )
        .expect("write toolchain configuration");
        std::fs::write(repository.path().join(".cargo/config.toml"), b"[build]\n")
            .expect("write Cargo configuration");
        std::fs::write(repository.path().join("build.rs"), b"fn main() {}\n")
            .expect("write build script");
        std::fs::write(
            repository.path().join("crates/example/Cargo.toml"),
            b"[package]\nname = \"example\"\nversion = \"0.1.0\"\n",
        )
        .expect("write workspace member manifest");
        run_fixture_git(repository.path(), &["add", "."]);
        run_fixture_git(
            repository.path(),
            &[
                "-c",
                "user.name=Gauntlet Test",
                "-c",
                "user.email=gauntlet@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "compiler inputs",
            ],
        );

        let selected = collect_tracked_compiler_inputs(repository.path())
            .expect("collect tracked compiler inputs");
        assert!(selected.contains_key("tracked.txt"));
        assert!(selected["Cargo.lock"].contains(&ArtifactStoreV4SourceInclusionReason::CargoLock));
        assert!(
            selected["rust-toolchain.toml"]
                .contains(&ArtifactStoreV4SourceInclusionReason::ToolchainConfig)
        );
        assert!(
            selected[".cargo/config.toml"]
                .contains(&ArtifactStoreV4SourceInclusionReason::CargoConfig)
        );
        assert!(
            selected[".cargo/config.toml"]
                .contains(&ArtifactStoreV4SourceInclusionReason::TargetConfig)
        );
        assert!(
            selected["build.rs"].contains(&ArtifactStoreV4SourceInclusionReason::BuildScriptInput)
        );
        assert!(
            selected["crates/example/Cargo.toml"]
                .contains(&ArtifactStoreV4SourceInclusionReason::WorkspaceMember)
        );

        let snapshot =
            ArtifactStoreV4SourceSnapshot::capture_selected(repository.path(), selected, 1024)
                .expect("descriptor-stable source snapshot");
        assert!(
            snapshot
                .entries
                .iter()
                .any(|entry| entry.relative_path == "Cargo.lock")
        );
    }

    #[test]
    fn artifactstore_v4_collects_observable_inputs_without_git_provenance() {
        let workspace = tempfile::tempdir().expect("temporary observable workspace");
        std::fs::create_dir_all(workspace.path().join(".cargo"))
            .expect("create Cargo config directory");
        std::fs::create_dir_all(workspace.path().join("generated"))
            .expect("create generated source directory");
        std::fs::create_dir_all(workspace.path().join("target"))
            .expect("create excluded target directory");
        std::fs::create_dir_all(workspace.path().join(".scratch"))
            .expect("create excluded diagnostic directory");
        std::fs::create_dir_all(workspace.path().join(".agent-state"))
            .expect("create excluded control directory");
        std::fs::write(workspace.path().join("Cargo.lock"), "lock").expect("write Cargo lock");
        std::fs::write(workspace.path().join("build.rs"), "fn main() {}")
            .expect("write build script");
        std::fs::write(workspace.path().join(".cargo/config.toml"), "[build]")
            .expect("write Cargo config");
        std::fs::write(
            workspace.path().join("generated/input.rs"),
            "pub const INPUT: u8 = 1;",
        )
        .expect("write generated compiler input");
        std::fs::write(workspace.path().join("target/not-source.rs"), "ignored")
            .expect("write generated build output");
        std::fs::write(workspace.path().join(".scratch/receipt.log"), "ignored")
            .expect("write non-compiler diagnostic output");
        std::fs::write(workspace.path().join(".agent-state/state.json"), "ignored")
            .expect("write non-compiler control output");

        let selected = collect_observable_workspace_inputs(workspace.path())
            .expect("collect Git-less observable workspace inputs");
        assert!(selected["Cargo.lock"].contains(&ArtifactStoreV4SourceInclusionReason::Untracked));
        assert!(selected["Cargo.lock"].contains(&ArtifactStoreV4SourceInclusionReason::CargoLock));
        assert!(
            selected["build.rs"].contains(&ArtifactStoreV4SourceInclusionReason::BuildScriptInput)
        );
        assert!(
            selected[".cargo/config.toml"]
                .contains(&ArtifactStoreV4SourceInclusionReason::CargoConfig)
        );
        assert!(
            selected["generated/input.rs"]
                .contains(&ArtifactStoreV4SourceInclusionReason::Untracked)
        );
        assert!(
            !selected.contains_key("target/not-source.rs"),
            "Cargo output directories are not compiler inputs"
        );
        assert!(
            !selected.contains_key(".scratch/receipt.log"),
            "diagnostic output directories are not compiler inputs"
        );
        assert!(
            !selected.contains_key(".agent-state/state.json"),
            "hidden control directories other than .cargo are not compiler inputs"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn artifactstore_v4_linux_executable_receipt_binds_running_procfs_image() {
        let producer = GauntletProducerBuildIdentity::compiled().expect("compiled producer");
        let input = linux_running_image_build_input(&producer)
            .expect("bind kernel-held running executable");
        assert_eq!(input.kind, ArtifactStoreV4BuildInputKind::Executable);
        let receipt: serde_json::Value =
            serde_json::from_slice(&input.canonical_bytes).expect("decode executable receipt");
        assert_eq!(receipt["path"], "/proc/self/exe");
        assert_eq!(receipt["sha256"], producer.executable_sha256);
        assert_eq!(receipt["byte_len"], producer.executable_byte_len);
        assert_eq!(receipt["verification"], "procfs_running_image");

        let mut path_snapshot = producer;
        path_snapshot.executable_verification = GauntletExecutableVerification::PathSnapshot;
        assert!(
            linux_running_image_build_input(&path_snapshot).is_err(),
            "a replaceable executable path must not satisfy the Linux receipt"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "requires strict remote Linux execution"]
    fn artifactstore_v4_current_linux_collector_binds_procfs_image_to_source_build_chain() {
        let producer = GauntletProducerBuildIdentity::compiled().expect("compiled producer");
        producer
            .validate_stored_v2()
            .expect("compiled producer must retain a well-formed typed source provenance");
        let snapshots = ArtifactStoreV4SourceBuildSnapshots::collect_current_linux()
            .expect("collect a current Linux source/build chain");
        snapshots
            .source()
            .validate()
            .expect("validate source snapshot");
        snapshots
            .build()
            .validate()
            .expect("validate build snapshot");
        let executable = snapshots
            .build()
            .inputs
            .iter()
            .find(|input| input.kind == ArtifactStoreV4BuildInputKind::Executable)
            .expect("Build snapshot must retain a Linux executable receipt");
        let receipt: serde_json::Value =
            serde_json::from_slice(&executable.canonical_bytes).expect("decode executable receipt");
        assert_eq!(receipt["path"], "/proc/self/exe");
        assert_eq!(receipt["verification"], "procfs_running_image");
        assert_eq!(receipt["sha256"], producer.executable_sha256);
        let source_provenance = snapshots
            .build()
            .inputs
            .iter()
            .find(|input| input.key == "provenance/source-verification")
            .expect("Build snapshot must retain typed source provenance");
        let provenance: serde_json::Value =
            serde_json::from_slice(&source_provenance.canonical_bytes)
                .expect("decode source provenance receipt");
        assert_eq!(
            provenance["source_verification"],
            serde_json::to_value(producer.source_verification)
                .expect("serialize source-verification mode")
        );
        assert_eq!(
            provenance["source_git_revision"],
            producer.source_git_revision
        );
        assert_eq!(provenance["source_git_dirty"], producer.source_git_dirty);
    }

    #[test]
    fn artifactstore_v4_source_snapshot_rejects_ambiguous_paths_and_kinds() {
        let hash = "c".repeat(64);
        assert!(matches!(
            ArtifactStoreV4SourceSnapshot::new(vec![ArtifactStoreV4SourceEntry {
                relative_path: "../Cargo.toml".to_owned(),
                kind: ArtifactStoreV4SourceEntryKind::File,
                inclusion_reasons: vec![ArtifactStoreV4SourceInclusionReason::Tracked],
                mode: 0o100_644,
                byte_len: 1,
                sha256: hash.clone(),
                symlink_target: None,
                resolved_target_path: None,
            }]),
            Err(GauntletError::InvalidPreparedArtifact { .. })
        ));
        assert!(matches!(
            ArtifactStoreV4SourceSnapshot::new(vec![ArtifactStoreV4SourceEntry {
                relative_path: "current".to_owned(),
                kind: ArtifactStoreV4SourceEntryKind::Symlink,
                inclusion_reasons: vec![ArtifactStoreV4SourceInclusionReason::PathDependency],
                mode: 0o120_777,
                byte_len: 0,
                sha256: hash,
                symlink_target: None,
                resolved_target_path: None,
            }]),
            Err(GauntletError::InvalidPreparedArtifact { .. })
        ));
    }

    #[cfg(all(target_family = "unix", not(target_os = "wasi")))]
    #[test]
    fn artifactstore_v4_source_snapshot_capture_binds_file_and_symlink_bytes() {
        use std::os::unix::fs::PermissionsExt as _;
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().expect("temporary source root");
        std::fs::write(root.path().join("Cargo.lock"), b"locked dependency graph\n")
            .expect("write source lockfile");
        std::fs::create_dir(root.path().join("crates")).expect("create source subdirectory");
        symlink("../Cargo.lock", root.path().join("crates/current"))
            .expect("create source symlink");

        let selected = BTreeMap::from([
            (
                "Cargo.lock".to_owned(),
                vec![
                    ArtifactStoreV4SourceInclusionReason::Tracked,
                    ArtifactStoreV4SourceInclusionReason::CargoLock,
                ],
            ),
            (
                "crates/current".to_owned(),
                vec![ArtifactStoreV4SourceInclusionReason::PathDependency],
            ),
        ]);
        let snapshot =
            ArtifactStoreV4SourceSnapshot::capture_selected(root.path(), selected.clone(), 1024)
                .expect("capture descriptor-stable source snapshot");

        snapshot.validate().expect("validate captured snapshot");
        assert_eq!(snapshot.entries[0].byte_len, 24);
        assert_eq!(
            snapshot.entries[1].symlink_target.as_deref(),
            Some("../Cargo.lock")
        );
        assert_eq!(
            snapshot.entries[1].resolved_target_path.as_deref(),
            Some("Cargo.lock")
        );

        std::fs::write(root.path().join("Cargo.lock"), b"sealed dependency graph\n")
            .expect("rewrite source lockfile at the same length");
        let content_mutated =
            ArtifactStoreV4SourceSnapshot::capture_selected(root.path(), selected.clone(), 1024)
                .expect("recapture same-length content mutation");
        assert_ne!(
            snapshot.identity_sha256, content_mutated.identity_sha256,
            "same-length in-place content mutation must change the Source identity"
        );

        std::fs::set_permissions(
            root.path().join("Cargo.lock"),
            std::fs::Permissions::from_mode(0o100_600),
        )
        .expect("change source-file mode");
        let mode_mutated =
            ArtifactStoreV4SourceSnapshot::capture_selected(root.path(), selected, 1024)
                .expect("recapture mode mutation");
        assert_ne!(
            content_mutated.identity_sha256, mode_mutated.identity_sha256,
            "mode mutation must change the Source identity"
        );
    }

    #[cfg(all(target_family = "unix", not(target_os = "wasi")))]
    #[test]
    fn artifactstore_v4_source_snapshot_capture_rejects_incomplete_and_unsafe_inputs() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().expect("temporary source root");
        std::fs::write(root.path().join("Cargo.lock"), b"locked dependency graph\n")
            .expect("write source lockfile");
        std::fs::create_dir(root.path().join("crates")).expect("create source subdirectory");
        symlink("../Cargo.lock", root.path().join("crates/current"))
            .expect("create source symlink");
        symlink("/outside-source-root", root.path().join("crates/escape"))
            .expect("create escaping source symlink");

        assert!(
            ArtifactStoreV4SourceSnapshot::capture_selected(
                root.path(),
                BTreeMap::from([(
                    "crates/current".to_owned(),
                    vec![ArtifactStoreV4SourceInclusionReason::PathDependency],
                )]),
                1024,
            )
            .is_err(),
            "a selected symlink must also select its regular-file target"
        );
        assert!(
            ArtifactStoreV4SourceSnapshot::capture_selected(
                root.path(),
                BTreeMap::from([(
                    "crates/escape".to_owned(),
                    vec![ArtifactStoreV4SourceInclusionReason::PathDependency],
                )]),
                1024,
            )
            .is_err(),
            "a source symlink escaping the selected root must fail closed"
        );
        assert!(
            ArtifactStoreV4SourceSnapshot::capture_selected(
                root.path(),
                BTreeMap::from([(
                    "crates".to_owned(),
                    vec![ArtifactStoreV4SourceInclusionReason::WorkspaceMember],
                )]),
                1024,
            )
            .is_err(),
            "a directory is not a compiler-visible file input"
        );
        assert!(
            ArtifactStoreV4SourceSnapshot::capture_selected(
                root.path(),
                BTreeMap::from([(
                    "Cargo.lock".to_owned(),
                    vec![ArtifactStoreV4SourceInclusionReason::CargoLock],
                )]),
                1,
            )
            .is_err(),
            "an oversized selected input must fail closed"
        );
    }

    #[test]
    fn artifactstore_v4_build_snapshot_binds_exact_build_input_bytes() {
        let kinds = [
            ArtifactStoreV4BuildInputKind::CargoLock,
            ArtifactStoreV4BuildInputKind::RegistryChecksum,
            ArtifactStoreV4BuildInputKind::GitDependency,
            ArtifactStoreV4BuildInputKind::Toolchain,
            ArtifactStoreV4BuildInputKind::Compiler,
            ArtifactStoreV4BuildInputKind::Linker,
            ArtifactStoreV4BuildInputKind::TargetConfig,
            ArtifactStoreV4BuildInputKind::CargoConfig,
            ArtifactStoreV4BuildInputKind::Environment,
            ArtifactStoreV4BuildInputKind::BuildScriptInput,
            ArtifactStoreV4BuildInputKind::BuildScriptOutput,
            ArtifactStoreV4BuildInputKind::GeneratedSource,
            ArtifactStoreV4BuildInputKind::FeatureSelection,
            ArtifactStoreV4BuildInputKind::Profile,
            ArtifactStoreV4BuildInputKind::Rustflags,
            ArtifactStoreV4BuildInputKind::Executable,
            ArtifactStoreV4BuildInputKind::DebugMetadata,
        ];
        let inputs: Vec<ArtifactStoreV4BuildInput> = kinds
            .into_iter()
            .enumerate()
            .map(|(index, kind)| {
                let canonical_bytes =
                    format!("exact compiler-visible input {index:02}").into_bytes();
                ArtifactStoreV4BuildInput {
                    key: format!("build-input-{index:02}"),
                    kind,
                    sha256: lower_hex(&Sha256::digest(&canonical_bytes)),
                    canonical_bytes,
                }
            })
            .collect();
        let snapshot = ArtifactStoreV4BuildSnapshot::new("d".repeat(64), inputs.clone())
            .expect("construct canonical Build snapshot");
        snapshot
            .validate()
            .expect("validate constructed Build snapshot");

        for index in 0..inputs.len() {
            let mut tampered_inputs = inputs.clone();
            tampered_inputs[index].canonical_bytes.push(b'!');
            assert!(matches!(
                ArtifactStoreV4BuildSnapshot::new("d".repeat(64), tampered_inputs),
                Err(GauntletError::InvalidPreparedArtifact { .. })
            ));
        }
    }

    #[test]
    fn artifactstore_v4_source_build_chain_persists_and_reloads_exact_bytes() {
        let source = ArtifactStoreV4SourceSnapshot::new(vec![ArtifactStoreV4SourceEntry {
            relative_path: "Cargo.lock".to_owned(),
            kind: ArtifactStoreV4SourceEntryKind::File,
            inclusion_reasons: vec![ArtifactStoreV4SourceInclusionReason::CargoLock],
            mode: 0o100_644,
            byte_len: 5,
            sha256: lower_hex(&Sha256::digest(b"lock\n")),
            symlink_target: None,
            resolved_target_path: None,
        }])
        .expect("construct source snapshot");
        let build = ArtifactStoreV4BuildSnapshot::new(
            source.identity_sha256.clone(),
            vec![ArtifactStoreV4BuildInput {
                key: "Cargo.lock".to_owned(),
                kind: ArtifactStoreV4BuildInputKind::CargoLock,
                canonical_bytes: b"lock\n".to_vec(),
                sha256: lower_hex(&Sha256::digest(b"lock\n")),
            }],
        )
        .expect("construct build snapshot");
        let snapshots =
            ArtifactStoreV4SourceBuildSnapshots::new(source, build).expect("bind Source to Build");
        let root = tempfile::tempdir().expect("temporary ArtifactStore root");
        let store = ArtifactStore::new(root.path());

        let binding = store
            .persist_v4_source_build_snapshots(&snapshots)
            .expect("persist immutable v4 chain");
        let reloaded = store
            .load_v4_source_build_snapshots(&binding)
            .expect("reload exact immutable v4 chain");

        assert_eq!(reloaded, snapshots);
        assert_eq!(
            store
                .persist_v4_source_build_snapshots(&snapshots)
                .expect("idempotent exact re-publish"),
            binding
        );

        let unrelated_source =
            ArtifactStoreV4SourceSnapshot::new(vec![ArtifactStoreV4SourceEntry {
                relative_path: "Cargo.toml".to_owned(),
                kind: ArtifactStoreV4SourceEntryKind::File,
                inclusion_reasons: vec![ArtifactStoreV4SourceInclusionReason::CargoConfig],
                mode: 0o100_644,
                byte_len: 9,
                sha256: lower_hex(&Sha256::digest(b"[package]")),
                symlink_target: None,
                resolved_target_path: None,
            }])
            .expect("construct unrelated source snapshot");
        assert!(matches!(
            ArtifactStoreV4SourceBuildSnapshots::new(unrelated_source, reloaded.build().clone()),
            Err(GauntletError::InvalidPreparedArtifact { .. })
        ));
    }

    #[test]
    fn live_git_preflight_rejects_every_hidden_or_drifting_checkout_state() {
        let (clean, clean_revision) = clean_fixture_repository();
        validate_live_git_checkout(clean.path(), clean.path(), &clean_revision)
            .expect("clean exact fixture checkout is admissible");

        let (untracked, untracked_revision) = clean_fixture_repository();
        std::fs::write(untracked.path().join("untracked.txt"), b"not sealed\n")
            .expect("write untracked fixture");
        assert!(
            validate_live_git_checkout(untracked.path(), untracked.path(), &untracked_revision)
                .is_err(),
            "untracked source must fail closed"
        );

        let (dirty, dirty_revision) = clean_fixture_repository();
        std::fs::write(dirty.path().join("tracked.txt"), b"dirty source\n")
            .expect("dirty tracked fixture");
        assert!(
            validate_live_git_checkout(dirty.path(), dirty.path(), &dirty_revision).is_err(),
            "dirty tracked source must fail closed"
        );

        let (assumed, assumed_revision) = clean_fixture_repository();
        run_fixture_git(
            assumed.path(),
            &["update-index", "--assume-unchanged", "tracked.txt"],
        );
        assert!(
            validate_live_git_checkout(assumed.path(), assumed.path(), &assumed_revision).is_err(),
            "assume-unchanged index state must fail closed"
        );

        let (skipped, skipped_revision) = clean_fixture_repository();
        run_fixture_git(
            skipped.path(),
            &["update-index", "--skip-worktree", "tracked.txt"],
        );
        assert!(
            validate_live_git_checkout(skipped.path(), skipped.path(), &skipped_revision).is_err(),
            "skip-worktree index state must fail closed"
        );

        let (advanced, sealed_revision) = clean_fixture_repository();
        std::fs::write(
            advanced.path().join("second.txt"),
            b"new committed source\n",
        )
        .expect("write revision-drift fixture");
        run_fixture_git(advanced.path(), &["add", "second.txt"]);
        run_fixture_git(
            advanced.path(),
            &[
                "-c",
                "user.name=Gauntlet Test",
                "-c",
                "user.email=gauntlet@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "advance fixture",
            ],
        );
        assert!(
            validate_live_git_checkout(advanced.path(), advanced.path(), &sealed_revision).is_err(),
            "revision drift must fail closed"
        );

        let (other, _) = clean_fixture_repository();
        assert!(
            validate_live_git_checkout(clean.path(), other.path(), &clean_revision).is_err(),
            "a clean checkout at the wrong canonical root must fail closed"
        );
    }

    #[test]
    fn every_shipped_schema_routes_through_the_typed_raw_byte_classifier() {
        // Campaign reports never had committed golden-byte fixtures. These
        // deliberately minimal probes exercise the duplicate-safe outer schema
        // envelope without pretending to be complete historical reports.
        let historical_reports: [(&[u8], SerializedSchemaDisposition); 6] = [
            (
            br#"{"schema_version":1,"run_id":"historical-v1","engines":{}}"#,
                SerializedSchemaDisposition::UnauthenticatedLegacy { schema_version: 1 },
            ),
            (
            br#"{"schema_version":2,"run_id":"historical-v2","provenance":null}"#,
                SerializedSchemaDisposition::UnauthenticatedLegacy { schema_version: 2 },
            ),
            (
            br#"{"schema_version":3,"run_id":"historical-v3","lexical_coverage":[]}"#,
                SerializedSchemaDisposition::UnauthenticatedLegacy { schema_version: 3 },
            ),
            (
            br#"{"schema_version":4,"run_id":"reserved-v4","pre_policy":true}"#,
                SerializedSchemaDisposition::ReservedRejected { schema_version: 4 },
            ),
            (
            br#"{"schema_version":5,"run_id":"historical-v5","producer_build_identity":{}}"#,
                SerializedSchemaDisposition::UnauthenticatedLegacy { schema_version: 5 },
            ),
            (
            br#"{"schema_version":6,"run_id":"integrity-only-v6","admission":"built_in_evidence"}"#,
                SerializedSchemaDisposition::LegacyIntegrityCeiling { schema_version: 6 },
            ),
        ];
        for (bytes, expected) in historical_reports {
            let version = expected.schema_version();
            assert_eq!(
                classify_campaign_report_schema(bytes).expect("typed report disposition"),
                expected,
            );
            let error = require_current_campaign_report_schema(bytes)
                .expect_err("historical reports must route before current DTO decode");
            let GauntletError::InvalidPreparedArtifact { reason } = error else {
                panic!("historical report v{version} returned the wrong error class");
            };
            match version {
                1 | 2 | 3 | 5 => assert!(reason.contains("unauthenticated legacy")),
                4 => assert!(reason.contains("reserved pre-policy")),
                6 => assert!(reason.contains("integrity only")),
                _ => unreachable!(),
            }
        }

        // v1/v3/v4/v5 are the exact committed historical golden bytes. v2 was
        // only ever a synthetic compatibility generation and v6 shipped
        // without a committed golden, so those two remain explicit probes.
        let historical_objects: [(&[u8], SerializedSchemaDisposition); 6] = [
            (
                include_bytes!("../fixtures/artifact-object-v1.json"),
                SerializedSchemaDisposition::UnauthenticatedLegacy { schema_version: 1 },
            ),
            (
                br#"{"object_schema_version":2,"engines":{},"oracle_version":{}}"#,
                SerializedSchemaDisposition::UnauthenticatedLegacy { schema_version: 2 },
            ),
            (
                include_bytes!("../fixtures/artifact-object-v3.json"),
                SerializedSchemaDisposition::UnauthenticatedLegacy { schema_version: 3 },
            ),
            (
                include_bytes!("../fixtures/artifact-object-v4.json"),
                SerializedSchemaDisposition::ReservedRejected { schema_version: 4 },
            ),
            (
                include_bytes!("../fixtures/artifact-object-v5.json"),
                SerializedSchemaDisposition::UnauthenticatedLegacy { schema_version: 5 },
            ),
            (
                br#"{"object_schema_version":6,"oracle_dependency":{"kind":"built_in_tantivy"}}"#,
                SerializedSchemaDisposition::LegacyIntegrityCeiling { schema_version: 6 },
            ),
        ];
        for (bytes, expected) in historical_objects {
            let version = expected.schema_version();
            assert_eq!(
                classify_artifact_object_schema(bytes).expect("typed object disposition"),
                expected,
            );
            let error = require_current_artifact_object_schema(bytes)
                .expect_err("historical objects must route before current DTO decode");
            let GauntletError::InvalidPreparedArtifact { reason } = error else {
                panic!("historical object v{version} returned the wrong error class");
            };
            match version {
                1..=3 => assert!(reason.contains("legacy artifact")),
                4 => assert!(reason.contains("reserved pre-policy")),
                5 => assert!(reason.contains("unauthenticated legacy")),
                6 => assert!(reason.contains("integrity only")),
                _ => unreachable!(),
            }
        }

        assert!(
            classify_campaign_report_schema(br#"{"schema_version":7,"schema_version":1}"#).is_err(),
            "duplicate report schema keys must fail closed"
        );
        assert!(
            classify_artifact_object_schema(
                br#"{"object_schema_version":7,"object_schema_version":1}"#
            )
            .is_err(),
            "duplicate object schema keys must fail closed"
        );
        assert_eq!(
            crate::runner::CAMPAIGN_REPORT_SCHEMA_VERSION,
            crate::runner::CAMPAIGN_REPORT_V7_SCHEMA_VERSION,
            "creation alias must not drift from the explicitly routed v7 generation",
        );
        assert_eq!(
            classify_campaign_report_schema(br#"{"schema_version":7}"#)
                .expect("current report candidate"),
            SerializedSchemaDisposition::CurrentIntegrityContractCandidate { schema_version: 7 },
        );
        assert_eq!(
            classify_artifact_object_schema(br#"{"object_schema_version":7}"#)
                .expect("current object candidate"),
            SerializedSchemaDisposition::CurrentIntegrityContractCandidate { schema_version: 7 },
        );

        let invalid_reports: [&[u8]; 11] = [
            b"",
            br"{",
            br"{}",
            br#"{"schema_version":null}"#,
            br#"{"schema_version":"7"}"#,
            br#"{"schema_version":7.0}"#,
            br#"{"schema_version":-1}"#,
            br#"{"schema_version":4294967296}"#,
            br#"{"schema_version":0}"#,
            br#"{"schema_version":8}"#,
            br#"{"schema_version":1,"schema_version":7}"#,
        ];
        for bytes in invalid_reports {
            assert!(
                classify_campaign_report_schema(bytes).is_err(),
                "invalid campaign-report schema envelope must fail: {}",
                String::from_utf8_lossy(bytes),
            );
        }

        let invalid_objects: [&[u8]; 11] = [
            b"",
            br"{",
            br"{}",
            br#"{"object_schema_version":null}"#,
            br#"{"object_schema_version":"7"}"#,
            br#"{"object_schema_version":7.0}"#,
            br#"{"object_schema_version":-1}"#,
            br#"{"object_schema_version":4294967296}"#,
            br#"{"object_schema_version":0}"#,
            br#"{"object_schema_version":8}"#,
            br#"{"object_schema_version":1,"object_schema_version":7}"#,
        ];
        for bytes in invalid_objects {
            assert!(
                classify_artifact_object_schema(bytes).is_err(),
                "invalid artifact-object schema envelope must fail: {}",
                String::from_utf8_lossy(bytes),
            );
        }
    }

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

    fn sample_lexical_contract_comparison() -> LexicalContractComparison {
        use crate::comparator::{
            LEXICAL_CONTRACT_BUNDLE_SCHEMA_VERSION, LEXICAL_CONTRACT_COMPARISON_SCHEMA_VERSION,
            LexicalBackendIdentity, LexicalBoundary, LexicalContractBundle,
            LexicalContractCoverage, LexicalCountState, LexicalEmptyShape, LexicalEngineRole,
            LexicalHydrationExecution, LexicalHydrationNotRunReason, LexicalHydrationSelection,
            LexicalHydrationTransition, LexicalObservation, LexicalObservationContext,
            LexicalObservationOutcome, LexicalProbeCoverage, LexicalSideCoverage,
        };

        let observation = |engine: &str, boundary| LexicalObservation {
            context: LexicalObservationContext::new(
                boundary,
                LexicalBackendIdentity {
                    engine: engine.to_owned(),
                    revision: format!("{engine}-test-revision"),
                    index_identity: "strict-wire-index".to_owned(),
                },
                "a".repeat(64),
                "b".repeat(64),
                "strict wire",
                7,
                10,
                LexicalExposureContract::CORE_LEXICAL_SEARCH,
            )
            .expect("strict lexical context"),
            outcome: LexicalObservationOutcome::Success {
                hits: Vec::new(),
                returned_count: 0,
                empty_shape: LexicalEmptyShape::Empty,
                total_count: LexicalCountState::NotExposed,
            },
        };
        let transition = || LexicalHydrationTransition {
            selection: LexicalHydrationSelection::AllLexicalWinners,
            execution: LexicalHydrationExecution::NotRun {
                reason: LexicalHydrationNotRunReason::CandidateSearchFailed,
            },
        };
        let bundle = |engine: &str, engine_role| LexicalContractBundle {
            schema_version: LEXICAL_CONTRACT_BUNDLE_SCHEMA_VERSION.to_owned(),
            engine_role,
            snapshot_sha256: "c".repeat(64),
            fusion_metadata_deferred: false,
            full_search: observation(engine, LexicalBoundary::FullSearch),
            fusion_candidates: observation(engine, LexicalBoundary::FusionCandidates),
            all_lexical_winners_hydration: transition(),
            strict_hybrid_winners_hydration: transition(),
            semantic_only_hydration: transition(),
            mixed_winners_hydration: transition(),
        };
        let coverage = || LexicalSideCoverage {
            full_search: LexicalProbeCoverage::ExercisedEmpty,
            fusion_candidates: LexicalProbeCoverage::ExercisedEmpty,
            all_lexical_winners_hydration: LexicalProbeCoverage::NotRun {
                reason: LexicalHydrationNotRunReason::CandidateSearchFailed,
            },
            strict_hybrid_winners_hydration: LexicalProbeCoverage::NotRun {
                reason: LexicalHydrationNotRunReason::CandidateSearchFailed,
            },
            semantic_only_hydration: LexicalProbeCoverage::NotRun {
                reason: LexicalHydrationNotRunReason::CandidateSearchFailed,
            },
            mixed_winners_hydration: LexicalProbeCoverage::NotRun {
                reason: LexicalHydrationNotRunReason::CandidateSearchFailed,
            },
        };

        LexicalContractComparison {
            schema_version: LEXICAL_CONTRACT_COMPARISON_SCHEMA_VERSION.to_owned(),
            status: LexicalComparisonStatus::Equivalent,
            applied_laws: Vec::new(),
            coverage: LexicalContractCoverage {
                subject: coverage(),
                oracle: coverage(),
            },
            waived_differences: Vec::new(),
            mismatches: Vec::new(),
            first_mismatch: None,
            subject: bundle("quill", LexicalEngineRole::Subject),
            oracle: bundle("tantivy", LexicalEngineRole::Oracle),
        }
    }

    fn sample_object() -> ArtifactObject {
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
            crate_version: oracle_version_contract()
                .expect("version contract")
                .lexical_package_version,
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
            trust_ceiling: ArtifactTrustCeiling::IntegrityOnly,
            execution_role: ArtifactExecutionRole::Diagnostic,
            legacy_oracle_version: None,
            oracle_dependency: ArtifactOracleDependency::DiagnosticUnspecified,
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

    fn bind_builtin_scalar_profile(object: &mut ArtifactObject) {
        let config = frankensearch_quill::QuillConfig::default();
        object.execution_role = ArtifactExecutionRole::BuiltInExecution;
        object.engines.subject.implementation = "frankensearch-quill/scalar-index".to_owned();
        object.engines.subject.crate_version =
            frankensearch_quill::FRANKENSEARCH_QUILL_CRATE_VERSION.to_owned();
        object.engines.subject.config_hash = quill_config_hash(&config);
        object.oracle_dependency = ArtifactOracleDependency::BuiltInTantivy {
            contract: oracle_version_contract().expect("version contract"),
        };
        if object.engines.semantic_contract.is_none() {
            object
                .engines
                .bind_semantic_contract(SemanticContract::shipping_default())
                .expect("bind built-in semantic contract");
        }
        object
            .engines
            .bind_builtin_profile(BuiltInEngineProfileReceipt::new(
                BuiltInEngineProfile::ScalarShipping,
                &config,
            ))
            .expect("bind built-in profile receipt");
    }

    fn sample_builtin_object() -> ArtifactObject {
        let mut object = sample_object();
        bind_builtin_scalar_profile(&mut object);
        object
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
            query_generator_schema_version: crate::QUERY_MANIFEST_SCHEMA_VERSION,
            query_generator_id: crate::GENERATOR_ID.to_owned(),
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

    fn sample_builtin_campaign_object() -> ArtifactObject {
        let mut object = sample_campaign_object();
        bind_builtin_scalar_profile(&mut object);
        object
    }

    fn golden_producer_build_identity() -> GauntletProducerBuildIdentity {
        let mut feature_hasher = Sha256::new();
        feature_hasher.update([]);
        GauntletProducerBuildIdentity {
            schema_version: PRODUCER_BUILD_IDENTITY_SCHEMA_VERSION,
            producer_contract_version: PRODUCER_CONTRACT_VERSION_V5.to_owned(),
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
            executable_verification: GauntletExecutableVerification::ProcfsRunningImage,
        }
    }

    fn golden_admissible_producer_build_identity() -> GauntletProducerBuildIdentity {
        let mut identity = golden_producer_build_identity();
        identity.enabled_features = vec!["tantivy_oracle".to_owned()];
        let mut feature_hasher = Sha256::new();
        feature_hasher.update(identity.enabled_features.join("\n"));
        identity.enabled_features_sha256 = lower_hex(&feature_hasher.finalize());
        identity
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
        let mut object = sample_builtin_object();
        object.engines.oracle.source_revision = "f".repeat(40);
        assert!(
            ArtifactStore::default()
                .prepare("bad-oracle-producer", &object, BTreeMap::new())
                .is_err()
        );
    }

    #[test]
    fn prepare_rejects_two_descriptors_that_agree_on_a_fabricated_build_identity() {
        let mut object = sample_builtin_object();
        bind_producer_identity(&mut object, golden_admissible_producer_build_identity());
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
        let mut dirty = sample_builtin_object();
        dirty.producer_build_identity.source_git_dirty = true;
        dirty.engines.subject.source_dirty = true;
        dirty.engines.oracle.source_dirty = true;
        dirty
            .validate()
            .expect("truthful dirty producer identity remains structurally recordable");
        assert!(matches!(
            dirty.validate_current_builtin_integrity(),
            Err(GauntletError::InvalidContract { ref reason })
                if reason.contains("clean Git-verified producer")
        ));

        let mut unverified = sample_builtin_object();
        unverified.producer_build_identity.source_git_revision = "d".repeat(40);
        unverified.engines.subject.source_revision = "d".repeat(40);
        unverified.engines.oracle.source_revision = "d".repeat(40);
        unverified.producer_build_identity.source_verification =
            GauntletProducerSourceVerification::ExplicitUnverified;
        unverified
            .validate()
            .expect("truthful unverified producer identity remains structurally recordable");
        assert!(matches!(
            unverified.validate_current_builtin_integrity(),
            Err(GauntletError::InvalidContract { ref reason })
                if reason.contains("clean Git-verified producer")
        ));
    }

    #[test]
    fn admissible_preparation_accepts_one_clean_verified_sealed_identity() {
        let store = ArtifactStore::default();
        let mut object = sample_builtin_object();
        let producer_identity = golden_admissible_producer_build_identity();
        bind_producer_identity(&mut object, producer_identity.clone());
        let prepared = store
            .prepare_at(
                "admissible-production",
                store.root().join("runs/admissible-production.json"),
                PreparedRunLocation::Standalone,
                false,
                ArtifactExecutionRole::BuiltInExecution,
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
        let mut object = sample_builtin_object();
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

        let mut object = sample_object();
        bind_producer_identity(&mut object, diagnostic_identity.clone());
        let run_path = store.root().join("runs/unverified-diagnostic.json");
        let prepared = store
            .prepare_at(
                "unverified-diagnostic",
                run_path,
                PreparedRunLocation::Standalone,
                false,
                ArtifactExecutionRole::Diagnostic,
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
        let mut object = sample_builtin_object();
        let ArtifactOracleDependency::BuiltInTantivy { contract } = &mut object.oracle_dependency
        else {
            panic!("sample evidence object lost its built-in dependency role")
        };
        contract.lexical_contract_audit_revision = "f".repeat(40);
        let error = object
            .validate()
            .expect_err("stored dependency mutation must fail structurally");
        assert!(matches!(
            error,
            GauntletError::InvalidContract { ref reason }
                if reason.contains("stored oracle dependency contract")
        ));
    }

    #[test]
    fn prepare_rejects_unpinned_oracle_configuration() {
        let mut object = sample_builtin_object();
        object.engines.oracle.config_hash = "different-schema".to_owned();
        assert!(
            ArtifactStore::default()
                .prepare("bad-oracle-config", &object, BTreeMap::new())
                .is_err()
        );
    }

    #[test]
    fn prepare_rejects_fabricated_exact_report() {
        let mut object = sample_builtin_object();
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
        assert_eq!(
            object
                .schema_trust_ceiling()
                .expect("legacy trust classification"),
            ArtifactTrustCeiling::UnauthenticatedLegacy
        );

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
    fn committed_v5_object_decodes_but_requires_a_current_role_bound_rerun() {
        const LEGACY_V5_BYTES: &[u8] = include_bytes!("../fixtures/artifact-object-v5.json");
        let object: ArtifactObject =
            serde_json::from_slice(LEGACY_V5_BYTES).expect("decode committed v5 object");
        assert_eq!(object.object_schema_version, 5);
        assert!(object.legacy_oracle_version.is_some());
        assert!(matches!(
            object.validate(),
            Err(GauntletError::InvalidContract { ref reason })
                if reason.contains("pre-run-identity artifact v5")
                    && reason.contains("non-admissible")
        ));
        assert_eq!(
            hash_object_bytes(LEGACY_V5_BYTES, 5)
                .expect("registered v5 diagnostic address")
                .len(),
            64
        );
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
        object.case.metadata.generator_id = Some(crate::GENERATOR_ID.to_owned());
        object.case.metadata.generator_seed = Some(42);
        object.case.metadata.corpus_hash = Some("0".repeat(64));

        assert!(
            ArtifactStore::default()
                .prepare_diagnostic("missing-campaign-context", &object, BTreeMap::new())
                .is_err()
        );
    }

    #[test]
    fn stored_generator_identity_campaign_context_is_explicit_and_fail_closed() {
        let object = sample_campaign_object();
        object.validate().expect("current manifest identity");

        let mut unsupported_schema = object.clone();
        unsupported_schema
            .campaign
            .as_mut()
            .expect("campaign context")
            .query_generator_schema_version =
            crate::QUERY_MANIFEST_SCHEMA_VERSION.saturating_add(1);
        assert!(unsupported_schema.validate().is_err());

        let mut unsupported_generator = object;
        unsupported_generator
            .campaign
            .as_mut()
            .expect("campaign context")
            .query_generator_id = "frankensearch-quill-gauntlet/generator-v3".to_owned();
        assert!(unsupported_generator.validate().is_err());
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
        let golden_with_newline = include_bytes!("../fixtures/artifact-object-v7.json");
        let golden = golden_with_newline
            .strip_suffix(b"\n")
            .expect("golden fixture must end in exactly one LF");
        assert_eq!(
            object.object_hash().unwrap(),
            "3ba1751438f70da4dfb41bdce755906602a4b7d3d3fcf499803919c70473c800"
        );
        assert_eq!(
            std::str::from_utf8(&canonical).expect("canonical object UTF-8"),
            std::str::from_utf8(golden).expect("golden object UTF-8")
        );
    }

    #[test]
    fn current_object_rejects_unknown_fields_across_representative_hashed_nested_dtos() {
        let object = sample_builtin_campaign_object();
        let paths = [
            "/oracle_dependency",
            "/producer_build_identity",
            "/engines",
            "/engines/subject",
            "/engines/oracle",
            "/engines/semantic_contract",
            "/case",
            "/case/metadata",
            "/comparator_config",
            "/comparison",
            "/comparison/subject",
            "/comparison/subject/hits/0",
            "/comparison/subject/hits/0/native_tie_key",
            "/comparison/oracle",
            "/comparison/oracle/hits/0",
            "/comparison/divergences/0",
            "/lexical_contract",
            "/campaign",
            "/campaign/semantic_contract",
            "/campaign/query",
            "/campaign/query/query_kind",
            "/campaign/query/filters",
            "/campaign/query/filters/source_filter",
        ];
        for path in paths {
            let mut encoded = serde_json::to_value(&object).expect("serialize current object");
            encoded
                .pointer_mut(path)
                .and_then(serde_json::Value::as_object_mut)
                .unwrap_or_else(|| panic!("strict-Serde path must name an object: {path}"))
                .insert("future_unbound_field".to_owned(), serde_json::json!(true));
            assert!(
                serde_json::from_value::<ArtifactObject>(encoded).is_err(),
                "unknown field at {path} must fail current object decoding",
            );
        }

        let mut diagnostic =
            serde_json::to_value(sample_object()).expect("serialize current diagnostic object");
        diagnostic
            .pointer_mut("/oracle_dependency")
            .and_then(serde_json::Value::as_object_mut)
            .expect("diagnostic dependency is an object")
            .insert("future_unbound_field".to_owned(), serde_json::json!(true));
        assert!(
            serde_json::from_value::<ArtifactObject>(diagnostic).is_err(),
            "unit diagnostic dependency variants must reject unknown fields",
        );

        let manifest = RunManifest {
            schema_version: RUN_MANIFEST_SCHEMA_VERSION,
            run_id: "strict-manifest".to_owned(),
            object_hash: "a".repeat(64),
            provenance: BTreeMap::new(),
        };
        let mut encoded = serde_json::to_value(&manifest).expect("serialize run manifest");
        encoded
            .as_object_mut()
            .expect("run manifest object")
            .insert("future_unbound_field".to_owned(), serde_json::json!(true));
        assert!(
            serde_json::from_value::<RunManifest>(encoded).is_err(),
            "run manifests must reject unknown fields",
        );
    }

    #[test]
    fn artifact_tagged_variants_reject_unknown_and_duplicate_tags_exhaustively() {
        for evidence in [
            ArtifactLexicalContractEvidence::LegacyPreV3Missing,
            ArtifactLexicalContractEvidence::RankEnvelopeOnly,
            ArtifactLexicalContractEvidence::CoreLexicalV3 {
                comparison: Box::new(sample_lexical_contract_comparison()),
            },
        ] {
            assert_strict_tagged_round_trip(&evidence);
        }

        for dependency in [
            ArtifactOracleDependency::LegacyMissing,
            ArtifactOracleDependency::DiagnosticUnspecified,
            ArtifactOracleDependency::BuiltInTantivy {
                contract: oracle_version_contract().expect("current oracle contract"),
            },
        ] {
            assert_strict_tagged_round_trip(&dependency);
        }

        assert!(
            serde_json::from_str::<ArtifactLexicalContractEvidence>(
                r#"{"scope":"rank_envelope_only","scope":"legacy_pre_v3_missing"}"#,
            )
            .is_err(),
            "duplicate lexical-contract scope tags must fail closed"
        );
        assert!(
            serde_json::from_str::<ArtifactOracleDependency>(
                r#"{"kind":"diagnostic_unspecified","kind":"legacy_missing"}"#,
            )
            .is_err(),
            "duplicate oracle-dependency kind tags must fail closed"
        );
    }

    #[test]
    fn object_embeds_both_engines_and_the_exact_version_contract() {
        let object = sample_builtin_object();
        assert_eq!(
            object
                .schema_trust_ceiling()
                .expect("current trust ceiling"),
            ArtifactTrustCeiling::IntegrityOnly,
            "typed built-in bytes are not admission authority"
        );
        assert_eq!(object.engines.subject.family, EngineFamily::Quill);
        assert_eq!(object.engines.oracle.family, EngineFamily::Tantivy);
        assert_eq!(
            object.oracle_dependency,
            ArtifactOracleDependency::BuiltInTantivy {
                contract: oracle_version_contract().unwrap(),
            }
        );
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
    fn stored_evidence_validation_does_not_consult_the_current_executable() {
        let mut object = sample_builtin_campaign_object();
        let archived_producer = golden_admissible_producer_build_identity();
        assert_ne!(
            archived_producer.executable_sha256,
            GauntletProducerBuildIdentity::compiled()
                .expect("compiled producer")
                .executable_sha256
        );
        let archived_producer_hash = archived_producer
            .identity_hash()
            .expect("archived producer hash");
        bind_producer_identity(&mut object, archived_producer);

        object
            .validate_stored_builtin_integrity()
            .expect("self-contained historical producer identity remains replayable");
        let binding = object
            .divergence_binding()
            .expect("archived evidence remains a valid divergence witness");
        assert_eq!(binding.producer_identity_sha256, archived_producer_hash);
        assert_eq!(binding.corpus_manifest_sha256, "a".repeat(64));
        assert_eq!(binding.query_manifest_sha256, "b".repeat(64));
        assert_eq!(binding.query_suite_source, QuerySuiteSource::ExplicitCases);
        assert_eq!(binding.query_source_identity_sha256, "c".repeat(64));
        assert_eq!(binding.fixture_id, object.case.fixture_id);
        assert_eq!(binding.rank_class, object.comparison.rank_class);
        assert_eq!(binding.divergences, object.comparison.divergences);
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
                .prepare_diagnostic("../escape", &object, BTreeMap::new())
                .is_err()
        );
        assert!(
            store
                .prepare_diagnostic("", &object, BTreeMap::new())
                .is_err()
        );
    }

    #[test]
    fn serialized_builtin_role_cannot_enter_the_diagnostic_namespace() {
        let encoded = serde_json::to_vec(&sample_builtin_object()).expect("encode built-in role");
        let decoded: ArtifactObject =
            serde_json::from_slice(&encoded).expect("decode public artifact DTO");
        let error = ArtifactStore::default()
            .prepare_diagnostic("role-laundering", &decoded, BTreeMap::new())
            .expect_err("diagnostic API must reject a deserialized built-in role");
        assert!(matches!(
            error,
            GauntletError::InvalidContract { ref reason }
                if reason.contains("artifact persistence execution role does not match the object")
        ));
    }

    #[test]
    fn diagnostic_role_cannot_enter_the_admissible_namespace() {
        let mut object = sample_object();
        let producer = golden_admissible_producer_build_identity();
        bind_producer_identity(&mut object, producer.clone());
        let error = ArtifactStore::default()
            .prepare_at(
                "diagnostic-promotion",
                PathBuf::from(".gauntlet/runs/diagnostic-promotion.json"),
                PreparedRunLocation::Standalone,
                false,
                ArtifactExecutionRole::BuiltInExecution,
                &producer,
                &object,
                BTreeMap::new(),
            )
            .expect_err("diagnostic object must not be promoted by persistence policy");
        assert!(matches!(
            error,
            GauntletError::InvalidContract { ref reason }
                if reason.contains("artifact persistence execution role does not match the object")
        ));
    }

    #[test]
    fn admissible_object_requires_the_tantivy_oracle_feature_receipt() {
        let mut object = sample_builtin_object();
        bind_producer_identity(&mut object, golden_producer_build_identity());
        let error = object
            .validate_current_builtin_integrity()
            .expect_err("featureless producer must not authenticate a built-in oracle");
        assert!(matches!(
            error,
            GauntletError::InvalidContract { ref reason }
                if reason.contains("tantivy_oracle")
        ));
    }

    #[test]
    fn admissible_object_rejects_subject_implementation_version_and_config_drift() {
        let producer = golden_admissible_producer_build_identity();

        let mut wrong_implementation = sample_builtin_object();
        bind_producer_identity(&mut wrong_implementation, producer.clone());
        wrong_implementation.engines.subject.implementation = "quill-lookalike".to_owned();
        assert!(wrong_implementation.validate().is_err());

        let mut wrong_version = sample_builtin_object();
        bind_producer_identity(&mut wrong_version, producer.clone());
        wrong_version.engines.subject.crate_version = "999.0.0".to_owned();
        assert!(wrong_version.validate_current_builtin_integrity().is_err());

        let mut wrong_config = sample_builtin_object();
        bind_producer_identity(&mut wrong_config, producer);
        wrong_config.engines.subject.config_hash = "0000000000000000".to_owned();
        assert!(wrong_config.validate().is_err());
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
            .prepare_campaign_case(
                "foo",
                0,
                ArtifactExecutionRole::Diagnostic,
                &sample_campaign_object(),
                BTreeMap::new(),
            )
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
