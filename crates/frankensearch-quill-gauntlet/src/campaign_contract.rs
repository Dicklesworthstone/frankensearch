//! Frozen replacement-completeness vocabulary and structural validator.
//!
//! This module deliberately has no dependency on the campaign runner or its
//! artifact types. It answers one narrow question: does a collection contain
//! exactly the 48 built-in evidence cells frozen by policy v1, with the exact
//! machine, topology, night, seed, corpus, profile, and replay bindings?
//!
//! Passing [`validate_replacement_completeness`] is only a structural fact. It
//! does not verify producer identities, admit an artifact, activate a gate,
//! authorize a default-backend flip, or make any terminal campaign decision.

use std::collections::BTreeSet;

use serde::{Deserialize, Deserializer, Serialize, de};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Wire schema and SHA-256 domain for the frozen replacement policy.
pub const REPLACEMENT_COMPLETENESS_POLICY_SCHEMA_VERSION: &str =
    "replacement-completeness-policy/v1";
/// Domain separator used before the canonical policy JSON preimage.
pub const REPLACEMENT_COMPLETENESS_POLICY_HASH_DOMAIN: &[u8] =
    b"replacement-completeness-policy/v1\0";
/// First UTC night admitted by the v1 replacement policy.
pub const REPLACEMENT_COMPLETENESS_EPOCH_NIGHT: &str = "2026-08-01";
/// Exact number of cells in the frozen v1 replacement policy.
pub const REPLACEMENT_COMPLETENESS_EXPECTED_CELL_COUNT: usize = 48;

/// Frozen UTC night for seed slot zero.
pub const REPLACEMENT_COMPLETENESS_SLOT_0_NIGHT: &str = "2026-08-01";
/// Frozen corpus seed for slot zero.
pub const REPLACEMENT_COMPLETENESS_SLOT_0_CORPUS_SEED: u64 = 0xE609;
/// Frozen query seed for slot zero.
pub const REPLACEMENT_COMPLETENESS_SLOT_0_QUERY_SEED: u64 = 0x9602;
/// Frozen repository seed for slot zero.
pub const REPLACEMENT_COMPLETENESS_SLOT_0_REPOSITORY_SEED: u64 = 0x9603;

/// Frozen UTC night for seed slot one.
pub const REPLACEMENT_COMPLETENESS_SLOT_1_NIGHT: &str = "2026-08-02";
/// Frozen corpus seed for slot one.
pub const REPLACEMENT_COMPLETENESS_SLOT_1_CORPUS_SEED: u64 = 0xE60A;
/// Frozen query seed for slot one.
pub const REPLACEMENT_COMPLETENESS_SLOT_1_QUERY_SEED: u64 = 0x9604;
/// Frozen repository seed for slot one.
pub const REPLACEMENT_COMPLETENESS_SLOT_1_REPOSITORY_SEED: u64 = 0x9605;

/// Frozen UTC night for seed slot two.
pub const REPLACEMENT_COMPLETENESS_SLOT_2_NIGHT: &str = "2026-08-03";
/// Frozen corpus seed for slot two.
pub const REPLACEMENT_COMPLETENESS_SLOT_2_CORPUS_SEED: u64 = 0xE60B;
/// Frozen query seed for slot two.
pub const REPLACEMENT_COMPLETENESS_SLOT_2_QUERY_SEED: u64 = 0x9606;
/// Frozen repository seed for slot two.
pub const REPLACEMENT_COMPLETENESS_SLOT_2_REPOSITORY_SEED: u64 = 0x9607;

/// The two replacement profiles whose complete evidence is required.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CampaignProfileV1 {
    /// Shipping facade/default behavior under the full core-v3 contract.
    #[serde(rename = "ShippingDefaultCoreV3")]
    ShippingDefaultCoreV3,
    /// The complete CASS-visible lexical contract.
    #[serde(rename = "CassTotalV1")]
    CassTotalV1,
}

/// Corpus provenance required independently for every replacement profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CampaignCorpusV1 {
    /// Deterministically generated conformance corpus.
    #[serde(rename = "Generated")]
    Generated,
    /// Frozen repository-snapshot corpus.
    #[serde(rename = "Repository")]
    Repository,
}

/// Closed hardware vocabulary for the two policy-v1 machines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CampaignHardwareClassV1 {
    /// AMD Threadripper PRO 5995WX machine class.
    #[serde(rename = "trj-zen3-5995wx")]
    TrjZen35995wx,
    /// Apple M4 macOS machine class.
    #[serde(rename = "m4-macos")]
    M4Macos,
}

/// Closed execution-profile vocabulary for policy v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CampaignExecutionProfileV1 {
    /// One admitted worker per physical AMD core.
    #[serde(rename = "physical-64")]
    Physical64,
    /// Scheduler-managed ten-worker Apple capacity.
    #[serde(rename = "scheduler-10")]
    Scheduler10,
}

/// Exact topology semantics bound into each machine key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CampaignTopologyV1 {
    /// Sixty-four physical cores, one admitted hardware thread per core.
    #[serde(rename = "physical-cores-64")]
    PhysicalCores64,
    /// Ten scheduler-managed workers without an affinity or P/E claim.
    #[serde(rename = "scheduler-workers-10")]
    SchedulerWorkers10,
}

/// Typed hardware, execution-profile, and topology tuple.
///
/// Deserialization accepts only the closed component vocabulary. The
/// completeness validator additionally requires one of the two exact frozen
/// tuples and rejects every cross-component substitution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignMachineProfileV1 {
    hardware_class_id: CampaignHardwareClassV1,
    execution_profile_id: CampaignExecutionProfileV1,
    topology: CampaignTopologyV1,
}

impl CampaignMachineProfileV1 {
    /// Build a typed tuple without claiming that policy admits the combination.
    #[must_use]
    pub const fn from_parts(
        hardware_class_id: CampaignHardwareClassV1,
        execution_profile_id: CampaignExecutionProfileV1,
        topology: CampaignTopologyV1,
    ) -> Self {
        Self {
            hardware_class_id,
            execution_profile_id,
            topology,
        }
    }

    /// Typed hardware identity.
    #[must_use]
    pub const fn hardware_class_id(self) -> CampaignHardwareClassV1 {
        self.hardware_class_id
    }

    /// Typed execution-profile identity.
    #[must_use]
    pub const fn execution_profile_id(self) -> CampaignExecutionProfileV1 {
        self.execution_profile_id
    }

    /// Typed capacity/topology semantics.
    #[must_use]
    pub const fn topology(self) -> CampaignTopologyV1 {
        self.topology
    }
}

/// Exact AMD `physical-64` machine key admitted by policy v1.
pub const REPLACEMENT_COMPLETENESS_AMD_PHYSICAL_64: CampaignMachineProfileV1 =
    CampaignMachineProfileV1::from_parts(
        CampaignHardwareClassV1::TrjZen35995wx,
        CampaignExecutionProfileV1::Physical64,
        CampaignTopologyV1::PhysicalCores64,
    );

/// Exact Apple `scheduler-10` machine key admitted by policy v1.
pub const REPLACEMENT_COMPLETENESS_APPLE_SCHEDULER_10: CampaignMachineProfileV1 =
    CampaignMachineProfileV1::from_parts(
        CampaignHardwareClassV1::M4Macos,
        CampaignExecutionProfileV1::Scheduler10,
        CampaignTopologyV1::SchedulerWorkers10,
    );

/// Strict Gregorian `YYYY-MM-DD` UTC-night value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct CampaignNightV1(String);

impl CampaignNightV1 {
    /// Parse one exact Gregorian calendar date.
    ///
    /// # Errors
    ///
    /// Returns [`CampaignContractValueError::InvalidNight`] for malformed or
    /// impossible dates.
    pub fn parse(value: &str) -> Result<Self, CampaignContractValueError> {
        if valid_yyyy_mm_dd(value) {
            Ok(Self(value.to_owned()))
        } else {
            Err(CampaignContractValueError::InvalidNight {
                value: value.to_owned(),
            })
        }
    }

    /// Exact wire spelling.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for CampaignNightV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Self::parse(&raw).map_err(de::Error::custom)
    }
}

/// Bounded seed-slot ordinal (`0..=2`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct CampaignSeedSlotV1(u8);

impl CampaignSeedSlotV1 {
    /// Frozen slot zero.
    pub const SLOT_0: Self = Self(0);
    /// Frozen slot one.
    pub const SLOT_1: Self = Self(1);
    /// Frozen slot two.
    pub const SLOT_2: Self = Self(2);

    /// Construct a bounded slot.
    ///
    /// # Errors
    ///
    /// Returns [`CampaignContractValueError::InvalidSeedSlot`] for values above
    /// two.
    pub const fn new(value: u8) -> Result<Self, CampaignContractValueError> {
        if value <= 2 {
            Ok(Self(value))
        } else {
            Err(CampaignContractValueError::InvalidSeedSlot { value })
        }
    }

    /// Numeric slot ordinal.
    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }
}

impl<'de> Deserialize<'de> for CampaignSeedSlotV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = u8::deserialize(deserializer)?;
        Self::new(raw).map_err(de::Error::custom)
    }
}

/// Bounded replay ordinal (`0` or `1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct CampaignReplayV1(u8);

impl CampaignReplayV1 {
    /// First frozen replay.
    pub const REPLAY_0: Self = Self(0);
    /// Second frozen replay.
    pub const REPLAY_1: Self = Self(1);

    /// Construct a bounded replay ordinal.
    ///
    /// # Errors
    ///
    /// Returns [`CampaignContractValueError::InvalidReplay`] for any value
    /// other than zero or one.
    pub const fn new(value: u8) -> Result<Self, CampaignContractValueError> {
        if value <= 1 {
            Ok(Self(value))
        } else {
            Err(CampaignContractValueError::InvalidReplay { value })
        }
    }

    /// Numeric replay ordinal.
    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }
}

impl<'de> Deserialize<'de> for CampaignReplayV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = u8::deserialize(deserializer)?;
        Self::new(raw).map_err(de::Error::custom)
    }
}

/// Exact three-seed bundle bound to one night/slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignSeedBundleV1 {
    #[serde(rename = "corpus_seed")]
    corpus: u64,
    #[serde(rename = "query_seed")]
    query: u64,
    #[serde(rename = "repository_seed")]
    repository: u64,
}

impl CampaignSeedBundleV1 {
    /// Construct a declarative seed bundle.
    #[must_use]
    pub const fn new(corpus_seed: u64, query_seed: u64, repository_seed: u64) -> Self {
        Self {
            corpus: corpus_seed,
            query: query_seed,
            repository: repository_seed,
        }
    }

    /// Corpus-generation seed.
    #[must_use]
    pub const fn corpus_seed(self) -> u64 {
        self.corpus
    }

    /// Query-generation seed.
    #[must_use]
    pub const fn query_seed(self) -> u64 {
        self.query
    }

    /// Repository-selection seed.
    #[must_use]
    pub const fn repository_seed(self) -> u64 {
        self.repository
    }
}

/// One frozen night, slot, and three-seed schedule entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignSeedScheduleV1 {
    night: CampaignNightV1,
    seed_slot: CampaignSeedSlotV1,
    seeds: CampaignSeedBundleV1,
}

impl CampaignSeedScheduleV1 {
    /// Scheduled UTC night.
    #[must_use]
    pub const fn night(&self) -> &CampaignNightV1 {
        &self.night
    }

    /// Scheduled seed-slot ordinal.
    #[must_use]
    pub const fn seed_slot(&self) -> CampaignSeedSlotV1 {
        self.seed_slot
    }

    /// All three frozen seeds for the slot.
    #[must_use]
    pub const fn seeds(&self) -> CampaignSeedBundleV1 {
        self.seeds
    }
}

/// SHA-256 digest spelled as exactly 64 lower-case hexadecimal characters.
///
/// This is an integrity value, not a verified-identity or admission token.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct CampaignSha256V1(String);

impl CampaignSha256V1 {
    /// Parse one strict lower-case SHA-256 hexadecimal spelling.
    ///
    /// # Errors
    ///
    /// Returns [`CampaignContractValueError::InvalidSha256`] unless `value` is
    /// exactly 64 lower-case hexadecimal characters.
    pub fn parse(value: &str) -> Result<Self, CampaignContractValueError> {
        if value.len() == 64
            && value
                .as_bytes()
                .iter()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        {
            Ok(Self(value.to_owned()))
        } else {
            Err(CampaignContractValueError::InvalidSha256 {
                value: value.to_owned(),
            })
        }
    }

    /// Exact lower-case hexadecimal spelling.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for CampaignSha256V1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Self::parse(&raw).map_err(de::Error::custom)
    }
}

/// Claimant-supplied producer/dependency identities for built-in evidence.
///
/// Parsing this binding proves only that two well-formed digests were supplied.
/// The identities must still be resolved and authenticated by the artifact and
/// runner layers before they can support any decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuiltInEvidenceBindingV1 {
    producer_build_identity_sha256: CampaignSha256V1,
    oracle_dependency_contract_sha256: CampaignSha256V1,
}

impl BuiltInEvidenceBindingV1 {
    /// Construct a declarative built-in evidence binding.
    #[must_use]
    pub const fn new(
        producer_build_identity_sha256: CampaignSha256V1,
        oracle_dependency_contract_sha256: CampaignSha256V1,
    ) -> Self {
        Self {
            producer_build_identity_sha256,
            oracle_dependency_contract_sha256,
        }
    }

    /// Claimed producer build identity.
    #[must_use]
    pub const fn producer_build_identity_sha256(&self) -> &CampaignSha256V1 {
        &self.producer_build_identity_sha256
    }

    /// Claimed oracle dependency-contract identity.
    #[must_use]
    pub const fn oracle_dependency_contract_sha256(&self) -> &CampaignSha256V1 {
        &self.oracle_dependency_contract_sha256
    }
}

/// Whether a cell is diagnostic or claims the built-in evidence path.
///
/// This serialized declaration is intentionally caller-constructible and is
/// therefore not an admission, verification, or flip-authority type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "binding", deny_unknown_fields)]
pub enum CampaignEvidenceRole {
    /// Diagnostic evidence; never replacement-complete.
    #[serde(rename = "Diagnostic")]
    Diagnostic,
    /// Evidence claiming the built-in path, bound to producer and dependency
    /// identities for later authentication by the owning layers.
    #[serde(rename = "BuiltInEvidence")]
    BuiltInEvidence(
        /// Declarative content-identity binding.
        BuiltInEvidenceBindingV1,
    ),
}

/// Contract surface actually exercised by one cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CampaignContractModeV1 {
    /// Rank-only coverage, explicitly insufficient for replacement.
    #[serde(rename = "RankEnvelopeOnly")]
    RankEnvelopeOnly,
    /// Full shipping default/core-v3 coverage.
    #[serde(rename = "CoreLexicalV3")]
    CoreLexicalV3,
    /// Full CASS-visible total contract.
    #[serde(rename = "CassTotalV1")]
    CassTotalV1,
}

/// Collision-free identity for one frozen policy cell.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignCellKeyV1 {
    campaign_profile: CampaignProfileV1,
    corpus: CampaignCorpusV1,
    machine_profile: CampaignMachineProfileV1,
    night: CampaignNightV1,
    seed_slot: CampaignSeedSlotV1,
    replay: CampaignReplayV1,
}

impl CampaignCellKeyV1 {
    /// Construct a declarative key. Frozen-policy membership is checked only by
    /// [`validate_replacement_completeness`].
    #[must_use]
    pub const fn new(
        campaign_profile: CampaignProfileV1,
        corpus: CampaignCorpusV1,
        machine_profile: CampaignMachineProfileV1,
        night: CampaignNightV1,
        seed_slot: CampaignSeedSlotV1,
        replay: CampaignReplayV1,
    ) -> Self {
        Self {
            campaign_profile,
            corpus,
            machine_profile,
            night,
            seed_slot,
            replay,
        }
    }

    /// Replacement profile axis.
    #[must_use]
    pub const fn campaign_profile(&self) -> CampaignProfileV1 {
        self.campaign_profile
    }

    /// Corpus-provenance axis.
    #[must_use]
    pub const fn corpus(&self) -> CampaignCorpusV1 {
        self.corpus
    }

    /// Exact typed machine/profile/topology tuple.
    #[must_use]
    pub const fn machine_profile(&self) -> CampaignMachineProfileV1 {
        self.machine_profile
    }

    /// UTC-night axis.
    #[must_use]
    pub const fn night(&self) -> &CampaignNightV1 {
        &self.night
    }

    /// Seed-slot axis.
    #[must_use]
    pub const fn seed_slot(&self) -> CampaignSeedSlotV1 {
        self.seed_slot
    }

    /// Replay axis.
    #[must_use]
    pub const fn replay(&self) -> CampaignReplayV1 {
        self.replay
    }
}

/// Evidence payload associated with one frozen cell key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignCellEvidenceV1 {
    key: CampaignCellKeyV1,
    evidence_role: CampaignEvidenceRole,
    contract_mode: CampaignContractModeV1,
    seeds: CampaignSeedBundleV1,
}

impl CampaignCellEvidenceV1 {
    /// Construct a declarative cell. This does not authenticate identities or
    /// prove frozen-policy completeness.
    #[must_use]
    pub const fn new(
        key: CampaignCellKeyV1,
        evidence_role: CampaignEvidenceRole,
        contract_mode: CampaignContractModeV1,
        seeds: CampaignSeedBundleV1,
    ) -> Self {
        Self {
            key,
            evidence_role,
            contract_mode,
            seeds,
        }
    }

    /// Cell identity.
    #[must_use]
    pub const fn key(&self) -> &CampaignCellKeyV1 {
        &self.key
    }

    /// Declarative evidence role.
    #[must_use]
    pub const fn evidence_role(&self) -> &CampaignEvidenceRole {
        &self.evidence_role
    }

    /// Exercised contract mode.
    #[must_use]
    pub const fn contract_mode(&self) -> CampaignContractModeV1 {
        self.contract_mode
    }

    /// Recorded seed bundle.
    #[must_use]
    pub const fn seeds(&self) -> CampaignSeedBundleV1 {
        self.seeds
    }
}

/// Canonical hash preimage for replacement-completeness policy v1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplacementCompletenessPolicyV1 {
    schema_version: String,
    epoch_night: CampaignNightV1,
    campaign_profiles: [CampaignProfileV1; 2],
    corpora: [CampaignCorpusV1; 2],
    machine_profiles: [CampaignMachineProfileV1; 2],
    seed_schedule: [CampaignSeedScheduleV1; 3],
    replays: [CampaignReplayV1; 2],
    expected_cells: Vec<CampaignCellKeyV1>,
}

impl ReplacementCompletenessPolicyV1 {
    /// Policy wire schema.
    #[must_use]
    pub fn schema_version(&self) -> &str {
        &self.schema_version
    }

    /// First UTC night in the frozen schedule.
    #[must_use]
    pub const fn epoch_night(&self) -> &CampaignNightV1 {
        &self.epoch_night
    }

    /// Ordered replacement profiles.
    #[must_use]
    pub const fn campaign_profiles(&self) -> &[CampaignProfileV1; 2] {
        &self.campaign_profiles
    }

    /// Ordered corpus-provenance values.
    #[must_use]
    pub const fn corpora(&self) -> &[CampaignCorpusV1; 2] {
        &self.corpora
    }

    /// Ordered exact machine keys.
    #[must_use]
    pub const fn machine_profiles(&self) -> &[CampaignMachineProfileV1; 2] {
        &self.machine_profiles
    }

    /// Ordered night/slot/seed schedule.
    #[must_use]
    pub const fn seed_schedule(&self) -> &[CampaignSeedScheduleV1; 3] {
        &self.seed_schedule
    }

    /// Ordered replay ordinals.
    #[must_use]
    pub const fn replays(&self) -> &[CampaignReplayV1; 2] {
        &self.replays
    }

    /// Canonically ordered complete cell set.
    #[must_use]
    pub fn expected_cells(&self) -> &[CampaignCellKeyV1] {
        &self.expected_cells
    }
}

/// Scalar-validation failures for strict campaign DTOs.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CampaignContractValueError {
    /// UTC night was not one exact, valid Gregorian date.
    #[error("campaign night must be an exact Gregorian YYYY-MM-DD date, found {value:?}")]
    InvalidNight {
        /// Rejected spelling.
        value: String,
    },
    /// Seed slot was outside `0..=2`.
    #[error("campaign seed slot must be 0, 1, or 2; found {value}")]
    InvalidSeedSlot {
        /// Rejected ordinal.
        value: u8,
    },
    /// Replay ordinal was outside `0..=1`.
    #[error("campaign replay must be 0 or 1; found {value}")]
    InvalidReplay {
        /// Rejected ordinal.
        value: u8,
    },
    /// SHA-256 spelling was not strict lower-case hexadecimal.
    #[error(
        "campaign SHA-256 must contain exactly 64 lower-case hexadecimal characters, found {value:?}"
    )]
    InvalidSha256 {
        /// Rejected spelling.
        value: String,
    },
}

/// Why a key is outside the frozen 48-cell universe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnexpectedCampaignCellReasonV1 {
    /// The night is not the one assigned to its seed slot.
    #[serde(rename = "NightSeedSlotMismatch")]
    NightSeedSlotMismatch,
    /// Hardware, execution profile, or topology is not one exact frozen tuple.
    #[serde(rename = "MachineProfileNotFrozen")]
    MachineProfileNotFrozen,
    /// A closed-axis combination was not present in the canonical policy set.
    #[serde(rename = "CellNotFrozen")]
    CellNotFrozen,
}

/// Fail-closed structural replacement-completeness errors.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ReplacementCompletenessError {
    /// A key outside the frozen policy was supplied.
    #[error("replacement evidence contains an extra cell {key:?}: {reason:?}")]
    ExtraCell {
        /// Rejected cell key.
        key: CampaignCellKeyV1,
        /// Stable classification of the mismatch.
        reason: UnexpectedCampaignCellReasonV1,
    },
    /// The same frozen key appeared more than once.
    #[error("replacement evidence contains duplicate cell {key:?}")]
    DuplicateCell {
        /// Duplicated key.
        key: CampaignCellKeyV1,
    },
    /// One frozen key was absent.
    #[error("replacement evidence is missing cell {key:?}")]
    MissingCell {
        /// First absent key in canonical order.
        key: CampaignCellKeyV1,
    },
    /// Diagnostic evidence cannot satisfy replacement completeness.
    #[error("replacement cell {key:?} is diagnostic rather than built-in evidence")]
    DiagnosticEvidence {
        /// Diagnostic cell key.
        key: CampaignCellKeyV1,
    },
    /// Rank-only evidence cannot satisfy replacement completeness.
    #[error("replacement cell {key:?} exercised RankEnvelopeOnly")]
    RankEnvelopeOnly {
        /// Rank-only cell key.
        key: CampaignCellKeyV1,
    },
    /// Full contract mode did not match the key's replacement profile.
    #[error("replacement cell {key:?} requires {expected:?} but recorded {actual:?}")]
    ContractModeMismatch {
        /// Mismatched cell key.
        key: CampaignCellKeyV1,
        /// Required full contract.
        expected: CampaignContractModeV1,
        /// Recorded contract.
        actual: CampaignContractModeV1,
    },
    /// Any one of the three seeds differed from the slot's frozen bundle.
    #[error("replacement cell {key:?} has seed bundle {actual:?}, expected {expected:?}")]
    SeedBundleMismatch {
        /// Mismatched cell key.
        key: CampaignCellKeyV1,
        /// Frozen slot bundle.
        expected: CampaignSeedBundleV1,
        /// Recorded bundle.
        actual: CampaignSeedBundleV1,
    },
}

/// Return the three ordered night/slot/seed schedule entries frozen by v1.
#[must_use]
pub fn frozen_replacement_seed_schedule() -> [CampaignSeedScheduleV1; 3] {
    [
        CampaignSeedScheduleV1 {
            night: CampaignNightV1(REPLACEMENT_COMPLETENESS_SLOT_0_NIGHT.to_owned()),
            seed_slot: CampaignSeedSlotV1::SLOT_0,
            seeds: CampaignSeedBundleV1::new(
                REPLACEMENT_COMPLETENESS_SLOT_0_CORPUS_SEED,
                REPLACEMENT_COMPLETENESS_SLOT_0_QUERY_SEED,
                REPLACEMENT_COMPLETENESS_SLOT_0_REPOSITORY_SEED,
            ),
        },
        CampaignSeedScheduleV1 {
            night: CampaignNightV1(REPLACEMENT_COMPLETENESS_SLOT_1_NIGHT.to_owned()),
            seed_slot: CampaignSeedSlotV1::SLOT_1,
            seeds: CampaignSeedBundleV1::new(
                REPLACEMENT_COMPLETENESS_SLOT_1_CORPUS_SEED,
                REPLACEMENT_COMPLETENESS_SLOT_1_QUERY_SEED,
                REPLACEMENT_COMPLETENESS_SLOT_1_REPOSITORY_SEED,
            ),
        },
        CampaignSeedScheduleV1 {
            night: CampaignNightV1(REPLACEMENT_COMPLETENESS_SLOT_2_NIGHT.to_owned()),
            seed_slot: CampaignSeedSlotV1::SLOT_2,
            seeds: CampaignSeedBundleV1::new(
                REPLACEMENT_COMPLETENESS_SLOT_2_CORPUS_SEED,
                REPLACEMENT_COMPLETENESS_SLOT_2_QUERY_SEED,
                REPLACEMENT_COMPLETENESS_SLOT_2_REPOSITORY_SEED,
            ),
        },
    ]
}

/// Return the exact seed bundle assigned to one bounded slot.
#[must_use]
pub fn frozen_replacement_seed_bundle(slot: CampaignSeedSlotV1) -> CampaignSeedBundleV1 {
    match slot.get() {
        0 => CampaignSeedBundleV1::new(
            REPLACEMENT_COMPLETENESS_SLOT_0_CORPUS_SEED,
            REPLACEMENT_COMPLETENESS_SLOT_0_QUERY_SEED,
            REPLACEMENT_COMPLETENESS_SLOT_0_REPOSITORY_SEED,
        ),
        1 => CampaignSeedBundleV1::new(
            REPLACEMENT_COMPLETENESS_SLOT_1_CORPUS_SEED,
            REPLACEMENT_COMPLETENESS_SLOT_1_QUERY_SEED,
            REPLACEMENT_COMPLETENESS_SLOT_1_REPOSITORY_SEED,
        ),
        2 => CampaignSeedBundleV1::new(
            REPLACEMENT_COMPLETENESS_SLOT_2_CORPUS_SEED,
            REPLACEMENT_COMPLETENESS_SLOT_2_QUERY_SEED,
            REPLACEMENT_COMPLETENESS_SLOT_2_REPOSITORY_SEED,
        ),
        _ => unreachable!("CampaignSeedSlotV1 maintains the private 0..=2 invariant"),
    }
}

/// Generate the canonical 48 frozen keys in policy order.
#[must_use]
pub fn frozen_replacement_cell_keys() -> Vec<CampaignCellKeyV1> {
    let campaign_profiles = [
        CampaignProfileV1::ShippingDefaultCoreV3,
        CampaignProfileV1::CassTotalV1,
    ];
    let corpora = [CampaignCorpusV1::Generated, CampaignCorpusV1::Repository];
    let machine_profiles = [
        REPLACEMENT_COMPLETENESS_AMD_PHYSICAL_64,
        REPLACEMENT_COMPLETENESS_APPLE_SCHEDULER_10,
    ];
    let schedule = frozen_replacement_seed_schedule();
    let replays = [CampaignReplayV1::REPLAY_0, CampaignReplayV1::REPLAY_1];

    let mut cells = Vec::with_capacity(REPLACEMENT_COMPLETENESS_EXPECTED_CELL_COUNT);
    for campaign_profile in campaign_profiles {
        for corpus in corpora {
            for machine_profile in machine_profiles {
                for slot in &schedule {
                    for replay in replays {
                        cells.push(CampaignCellKeyV1::new(
                            campaign_profile,
                            corpus,
                            machine_profile,
                            slot.night.clone(),
                            slot.seed_slot,
                            replay,
                        ));
                    }
                }
            }
        }
    }
    cells
}

/// Materialize the canonical policy-v1 hash preimage.
#[must_use]
pub fn frozen_replacement_completeness_policy() -> ReplacementCompletenessPolicyV1 {
    ReplacementCompletenessPolicyV1 {
        schema_version: REPLACEMENT_COMPLETENESS_POLICY_SCHEMA_VERSION.to_owned(),
        epoch_night: CampaignNightV1(REPLACEMENT_COMPLETENESS_EPOCH_NIGHT.to_owned()),
        campaign_profiles: [
            CampaignProfileV1::ShippingDefaultCoreV3,
            CampaignProfileV1::CassTotalV1,
        ],
        corpora: [CampaignCorpusV1::Generated, CampaignCorpusV1::Repository],
        machine_profiles: [
            REPLACEMENT_COMPLETENESS_AMD_PHYSICAL_64,
            REPLACEMENT_COMPLETENESS_APPLE_SCHEDULER_10,
        ],
        seed_schedule: frozen_replacement_seed_schedule(),
        replays: [CampaignReplayV1::REPLAY_0, CampaignReplayV1::REPLAY_1],
        expected_cells: frozen_replacement_cell_keys(),
    }
}

/// SHA-256 of the domain-separated canonical JSON policy preimage.
///
/// # Errors
///
/// Propagates serialization failure. The frozen policy currently contains no
/// fallible map keys or non-finite numeric values.
pub fn replacement_completeness_policy_sha256() -> Result<String, serde_json::Error> {
    hash_replacement_policy(
        REPLACEMENT_COMPLETENESS_POLICY_HASH_DOMAIN,
        &frozen_replacement_completeness_policy(),
    )
}

/// Require exact structural coverage of the frozen 48-cell replacement set.
///
/// The check is order-independent and fail-closed for extra, duplicate, and
/// missing keys. Every admitted key must also carry a built-in evidence
/// declaration, the profile's full contract mode, and all three exact slot
/// seeds.
///
/// Success is `()` on purpose: it cannot be retained, serialized, or passed as
/// a caller-mintable verification/admission token. The artifact and runner
/// layers remain responsible for authenticating the declarative digest
/// bindings and for every campaign decision.
///
/// # Errors
///
/// Returns the first deterministic structural mismatch.
pub fn validate_replacement_completeness(
    cells: &[CampaignCellEvidenceV1],
) -> Result<(), ReplacementCompletenessError> {
    let expected = frozen_replacement_cell_keys();
    let expected_set: BTreeSet<_> = expected.iter().cloned().collect();
    let mut seen = BTreeSet::new();

    for cell in cells {
        if !expected_set.contains(&cell.key) {
            return Err(ReplacementCompletenessError::ExtraCell {
                key: cell.key.clone(),
                reason: classify_unexpected_key(&cell.key),
            });
        }
        if !seen.insert(cell.key.clone()) {
            return Err(ReplacementCompletenessError::DuplicateCell {
                key: cell.key.clone(),
            });
        }
        if matches!(cell.evidence_role, CampaignEvidenceRole::Diagnostic) {
            return Err(ReplacementCompletenessError::DiagnosticEvidence {
                key: cell.key.clone(),
            });
        }
        if cell.contract_mode == CampaignContractModeV1::RankEnvelopeOnly {
            return Err(ReplacementCompletenessError::RankEnvelopeOnly {
                key: cell.key.clone(),
            });
        }

        let expected_mode = required_contract_mode(cell.key.campaign_profile);
        if cell.contract_mode != expected_mode {
            return Err(ReplacementCompletenessError::ContractModeMismatch {
                key: cell.key.clone(),
                expected: expected_mode,
                actual: cell.contract_mode,
            });
        }

        let expected_seeds = frozen_replacement_seed_bundle(cell.key.seed_slot);
        if cell.seeds != expected_seeds {
            return Err(ReplacementCompletenessError::SeedBundleMismatch {
                key: cell.key.clone(),
                expected: expected_seeds,
                actual: cell.seeds,
            });
        }
    }

    for key in expected {
        if !seen.contains(&key) {
            return Err(ReplacementCompletenessError::MissingCell { key });
        }
    }
    Ok(())
}

const fn required_contract_mode(profile: CampaignProfileV1) -> CampaignContractModeV1 {
    match profile {
        CampaignProfileV1::ShippingDefaultCoreV3 => CampaignContractModeV1::CoreLexicalV3,
        CampaignProfileV1::CassTotalV1 => CampaignContractModeV1::CassTotalV1,
    }
}

fn classify_unexpected_key(key: &CampaignCellKeyV1) -> UnexpectedCampaignCellReasonV1 {
    let schedule = frozen_replacement_seed_schedule();
    let scheduled_night = &schedule[usize::from(key.seed_slot.get())].night;
    if key.night != *scheduled_night {
        return UnexpectedCampaignCellReasonV1::NightSeedSlotMismatch;
    }
    if !matches!(
        key.machine_profile,
        REPLACEMENT_COMPLETENESS_AMD_PHYSICAL_64 | REPLACEMENT_COMPLETENESS_APPLE_SCHEDULER_10
    ) {
        return UnexpectedCampaignCellReasonV1::MachineProfileNotFrozen;
    }
    UnexpectedCampaignCellReasonV1::CellNotFrozen
}

fn hash_replacement_policy(
    domain: &[u8],
    policy: &ReplacementCompletenessPolicyV1,
) -> Result<String, serde_json::Error> {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(serde_json::to_vec(policy)?);
    Ok(lower_hex(&hasher.finalize()))
}

fn lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

fn valid_yyyy_mm_dd(value: &str) -> bool {
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

    let year = digits_to_u16(&bytes[0..4]);
    let month = digits_to_u16(&bytes[5..7]);
    let day = digits_to_u16(&bytes[8..10]);
    if year == 0 || !(1..=12).contains(&month) {
        return false;
    }
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => return false,
    };
    (1..=days_in_month).contains(&day)
}

fn digits_to_u16(bytes: &[u8]) -> u16 {
    bytes.iter().fold(0, |value, byte| {
        value * 10 + u16::from(byte.saturating_sub(b'0'))
    })
}

const fn is_leap_year(year: u16) -> bool {
    year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400))
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use serde_json::{Value, json};

    use super::*;

    fn digest(byte: char) -> CampaignSha256V1 {
        CampaignSha256V1::parse(&byte.to_string().repeat(64))
            .expect("test digest must be strict lower-case hex")
    }

    fn built_in_role() -> CampaignEvidenceRole {
        CampaignEvidenceRole::BuiltInEvidence(BuiltInEvidenceBindingV1::new(
            digest('a'),
            digest('b'),
        ))
    }

    fn valid_cells() -> Vec<CampaignCellEvidenceV1> {
        frozen_replacement_cell_keys()
            .into_iter()
            .map(|key| {
                let mode = required_contract_mode(key.campaign_profile);
                let seeds = frozen_replacement_seed_bundle(key.seed_slot);
                CampaignCellEvidenceV1::new(key, built_in_role(), mode, seeds)
            })
            .collect()
    }

    #[test]
    fn frozen_schedule_matches_all_preagreed_literals() {
        let schedule = frozen_replacement_seed_schedule();
        assert_eq!(schedule[0].night().as_str(), "2026-08-01");
        assert_eq!(schedule[0].seed_slot().get(), 0);
        assert_eq!(
            schedule[0].seeds(),
            CampaignSeedBundleV1::new(0xE609, 0x9602, 0x9603)
        );
        assert_eq!(schedule[1].night().as_str(), "2026-08-02");
        assert_eq!(schedule[1].seed_slot().get(), 1);
        assert_eq!(
            schedule[1].seeds(),
            CampaignSeedBundleV1::new(0xE60A, 0x9604, 0x9605)
        );
        assert_eq!(schedule[2].night().as_str(), "2026-08-03");
        assert_eq!(schedule[2].seed_slot().get(), 2);
        assert_eq!(
            schedule[2].seeds(),
            CampaignSeedBundleV1::new(0xE60B, 0x9606, 0x9607)
        );
    }

    #[test]
    fn frozen_cell_product_is_exactly_48_unique_keys() {
        let cells = frozen_replacement_cell_keys();
        let unique: BTreeSet<_> = cells.iter().cloned().collect();
        assert_eq!(cells.len(), REPLACEMENT_COMPLETENESS_EXPECTED_CELL_COUNT);
        assert_eq!(unique.len(), REPLACEMENT_COMPLETENESS_EXPECTED_CELL_COUNT);

        let mut profiles = BTreeMap::new();
        let mut corpora = BTreeMap::new();
        let mut machines = BTreeMap::new();
        let mut slots = BTreeMap::new();
        let mut replays = BTreeMap::new();
        for key in &cells {
            *profiles.entry(key.campaign_profile()).or_insert(0) += 1;
            *corpora.entry(key.corpus()).or_insert(0) += 1;
            *machines.entry(key.machine_profile()).or_insert(0) += 1;
            *slots.entry(key.seed_slot()).or_insert(0) += 1;
            *replays.entry(key.replay()).or_insert(0) += 1;
        }
        assert_eq!(profiles.values().copied().collect::<Vec<_>>(), vec![24, 24]);
        assert_eq!(corpora.values().copied().collect::<Vec<_>>(), vec![24, 24]);
        assert_eq!(machines.values().copied().collect::<Vec<_>>(), vec![24, 24]);
        assert_eq!(
            slots.values().copied().collect::<Vec<_>>(),
            vec![16, 16, 16]
        );
        assert_eq!(replays.values().copied().collect::<Vec<_>>(), vec![24, 24]);
    }

    #[test]
    fn frozen_cell_order_is_canonical_and_stable() {
        let cells = frozen_replacement_cell_keys();
        let first = cells.first().expect("frozen cells cannot be empty");
        assert_eq!(
            first.campaign_profile(),
            CampaignProfileV1::ShippingDefaultCoreV3
        );
        assert_eq!(first.corpus(), CampaignCorpusV1::Generated);
        assert_eq!(
            first.machine_profile(),
            REPLACEMENT_COMPLETENESS_AMD_PHYSICAL_64
        );
        assert_eq!(first.night().as_str(), "2026-08-01");
        assert_eq!(first.seed_slot().get(), 0);
        assert_eq!(first.replay().get(), 0);

        let last = cells.last().expect("frozen cells cannot be empty");
        assert_eq!(last.campaign_profile(), CampaignProfileV1::CassTotalV1);
        assert_eq!(last.corpus(), CampaignCorpusV1::Repository);
        assert_eq!(
            last.machine_profile(),
            REPLACEMENT_COMPLETENESS_APPLE_SCHEDULER_10
        );
        assert_eq!(last.night().as_str(), "2026-08-03");
        assert_eq!(last.seed_slot().get(), 2);
        assert_eq!(last.replay().get(), 1);

        assert_eq!(cells, frozen_replacement_cell_keys());
    }

    #[test]
    fn exact_machine_keys_have_stable_wire_forms() {
        assert_eq!(
            serde_json::to_value(REPLACEMENT_COMPLETENESS_AMD_PHYSICAL_64)
                .expect("AMD key must serialize"),
            json!({
                "hardware_class_id": "trj-zen3-5995wx",
                "execution_profile_id": "physical-64",
                "topology": "physical-cores-64"
            })
        );
        assert_eq!(
            serde_json::to_value(REPLACEMENT_COMPLETENESS_APPLE_SCHEDULER_10)
                .expect("Apple key must serialize"),
            json!({
                "hardware_class_id": "m4-macos",
                "execution_profile_id": "scheduler-10",
                "topology": "scheduler-workers-10"
            })
        );
    }

    #[test]
    fn strict_scalars_reject_bad_dates_slots_replays_and_hashes() {
        for invalid in [
            "2026-8-01",
            "20260801",
            "2026-02-29",
            "2024-02-30",
            "0000-01-01",
            "2026-13-01",
            "2026-08-00",
        ] {
            assert!(
                CampaignNightV1::parse(invalid).is_err(),
                "accepted {invalid}"
            );
        }
        assert!(CampaignNightV1::parse("2024-02-29").is_ok());
        assert!(CampaignSeedSlotV1::new(3).is_err());
        assert!(CampaignReplayV1::new(2).is_err());
        assert!(CampaignSha256V1::parse(&"a".repeat(63)).is_err());
        assert!(CampaignSha256V1::parse(&"A".repeat(64)).is_err());
        assert!(CampaignSha256V1::parse(&"g".repeat(64)).is_err());
        assert!(CampaignSha256V1::parse(&"0".repeat(64)).is_ok());

        assert!(serde_json::from_str::<CampaignNightV1>("\"2026-02-29\"").is_err());
        assert!(serde_json::from_str::<CampaignSeedSlotV1>("3").is_err());
        assert!(serde_json::from_str::<CampaignReplayV1>("2").is_err());
    }

    #[test]
    fn strict_dtos_reject_unknown_and_missing_fields_without_defaults() {
        let key = frozen_replacement_cell_keys().remove(0);
        let mut key_json = serde_json::to_value(&key).expect("key must serialize");
        key_json
            .as_object_mut()
            .expect("key must be an object")
            .insert("unknown".to_owned(), Value::Bool(true));
        assert!(serde_json::from_value::<CampaignCellKeyV1>(key_json).is_err());

        let mut missing = serde_json::to_value(&key).expect("key must serialize");
        missing
            .as_object_mut()
            .expect("key must be an object")
            .remove("replay");
        assert!(serde_json::from_value::<CampaignCellKeyV1>(missing).is_err());

        let cell = &valid_cells()[0];
        let mut cell_json = serde_json::to_value(cell).expect("cell must serialize");
        cell_json
            .as_object_mut()
            .expect("cell must be an object")
            .insert("terminal".to_owned(), Value::Bool(true));
        assert!(serde_json::from_value::<CampaignCellEvidenceV1>(cell_json).is_err());

        let policy = frozen_replacement_completeness_policy();
        let mut policy_json = serde_json::to_value(&policy).expect("policy must serialize");
        policy_json
            .as_object_mut()
            .expect("policy must be an object")
            .insert("default_flip".to_owned(), Value::Bool(true));
        assert!(serde_json::from_value::<ReplacementCompletenessPolicyV1>(policy_json).is_err());

        let mut missing_policy = serde_json::to_value(policy).expect("policy must serialize");
        missing_policy
            .as_object_mut()
            .expect("policy must be an object")
            .remove("expected_cells");
        assert!(serde_json::from_value::<ReplacementCompletenessPolicyV1>(missing_policy).is_err());
    }

    #[test]
    fn evidence_role_is_strict_and_binds_two_identities() {
        let role = built_in_role();
        let value = serde_json::to_value(&role).expect("role must serialize");
        assert_eq!(value["kind"], "BuiltInEvidence");
        assert_eq!(
            value["binding"]["producer_build_identity_sha256"],
            "a".repeat(64)
        );
        assert_eq!(
            value["binding"]["oracle_dependency_contract_sha256"],
            "b".repeat(64)
        );
        assert_eq!(
            serde_json::from_value::<CampaignEvidenceRole>(value)
                .expect("strict role must round-trip"),
            role
        );

        let extra = json!({"kind":"Diagnostic", "binding": {}});
        assert!(serde_json::from_value::<CampaignEvidenceRole>(extra).is_err());
        let missing = json!({"kind":"BuiltInEvidence"});
        assert!(serde_json::from_value::<CampaignEvidenceRole>(missing).is_err());
        let uppercase = json!({
            "kind": "BuiltInEvidence",
            "binding": {
                "producer_build_identity_sha256": "A".repeat(64),
                "oracle_dependency_contract_sha256": "b".repeat(64)
            }
        });
        assert!(serde_json::from_value::<CampaignEvidenceRole>(uppercase).is_err());
    }

    #[test]
    fn complete_built_in_full_contract_set_passes_in_any_order() {
        let cells = valid_cells();
        assert_eq!(validate_replacement_completeness(&cells), Ok(()));

        let mut reversed = cells;
        reversed.reverse();
        assert_eq!(validate_replacement_completeness(&reversed), Ok(()));
    }

    #[test]
    fn missing_duplicate_and_extra_cells_are_distinct_failures() {
        let mut missing = valid_cells();
        let absent = missing.remove(17).key;
        assert_eq!(
            validate_replacement_completeness(&missing),
            Err(ReplacementCompletenessError::MissingCell { key: absent })
        );

        let mut duplicate = valid_cells();
        let duplicated = duplicate[0].key.clone();
        duplicate[1].key = duplicated.clone();
        assert_eq!(
            validate_replacement_completeness(&duplicate),
            Err(ReplacementCompletenessError::DuplicateCell { key: duplicated })
        );

        let mut extra = valid_cells();
        extra[0].key.night = CampaignNightV1::parse("2026-08-04").expect("valid mutation date");
        let extra_key = extra[0].key.clone();
        assert_eq!(
            validate_replacement_completeness(&extra),
            Err(ReplacementCompletenessError::ExtraCell {
                key: extra_key,
                reason: UnexpectedCampaignCellReasonV1::NightSeedSlotMismatch,
            })
        );
    }

    #[test]
    fn diagnostic_rank_only_and_wrong_full_contract_are_rejected() {
        let mut diagnostic = valid_cells();
        diagnostic[7].evidence_role = CampaignEvidenceRole::Diagnostic;
        let key = diagnostic[7].key.clone();
        assert_eq!(
            validate_replacement_completeness(&diagnostic),
            Err(ReplacementCompletenessError::DiagnosticEvidence { key })
        );

        let mut rank_only = valid_cells();
        rank_only[8].contract_mode = CampaignContractModeV1::RankEnvelopeOnly;
        let key = rank_only[8].key.clone();
        assert_eq!(
            validate_replacement_completeness(&rank_only),
            Err(ReplacementCompletenessError::RankEnvelopeOnly { key })
        );

        let mut wrong_full = valid_cells();
        wrong_full[0].contract_mode = CampaignContractModeV1::CassTotalV1;
        let key = wrong_full[0].key.clone();
        assert_eq!(
            validate_replacement_completeness(&wrong_full),
            Err(ReplacementCompletenessError::ContractModeMismatch {
                key,
                expected: CampaignContractModeV1::CoreLexicalV3,
                actual: CampaignContractModeV1::CassTotalV1,
            })
        );
    }

    #[test]
    fn cross_night_slot_substitution_is_rejected() {
        let mut cells = valid_cells();
        cells[0].key.seed_slot = CampaignSeedSlotV1::SLOT_1;
        let key = cells[0].key.clone();
        assert_eq!(
            validate_replacement_completeness(&cells),
            Err(ReplacementCompletenessError::ExtraCell {
                key,
                reason: UnexpectedCampaignCellReasonV1::NightSeedSlotMismatch,
            })
        );
    }

    #[test]
    fn hardware_profile_and_topology_substitutions_are_each_rejected() {
        let mutations = [
            CampaignMachineProfileV1::from_parts(
                CampaignHardwareClassV1::M4Macos,
                CampaignExecutionProfileV1::Physical64,
                CampaignTopologyV1::PhysicalCores64,
            ),
            CampaignMachineProfileV1::from_parts(
                CampaignHardwareClassV1::TrjZen35995wx,
                CampaignExecutionProfileV1::Scheduler10,
                CampaignTopologyV1::PhysicalCores64,
            ),
            CampaignMachineProfileV1::from_parts(
                CampaignHardwareClassV1::TrjZen35995wx,
                CampaignExecutionProfileV1::Physical64,
                CampaignTopologyV1::SchedulerWorkers10,
            ),
        ];

        for mutation in mutations {
            let mut cells = valid_cells();
            cells[0].key.machine_profile = mutation;
            let key = cells[0].key.clone();
            assert_eq!(
                validate_replacement_completeness(&cells),
                Err(ReplacementCompletenessError::ExtraCell {
                    key,
                    reason: UnexpectedCampaignCellReasonV1::MachineProfileNotFrozen,
                })
            );
        }
    }

    #[test]
    fn every_individual_seed_mutation_is_rejected() {
        for seed_index in 0..3 {
            let mut cells = valid_cells();
            match seed_index {
                0 => cells[0].seeds.corpus += 1,
                1 => cells[0].seeds.query += 1,
                2 => cells[0].seeds.repository += 1,
                _ => unreachable!("bounded test seed index"),
            }
            let key = cells[0].key.clone();
            let actual = cells[0].seeds;
            assert_eq!(
                validate_replacement_completeness(&cells),
                Err(ReplacementCompletenessError::SeedBundleMismatch {
                    key,
                    expected: CampaignSeedBundleV1::new(0xE609, 0x9602, 0x9603),
                    actual,
                })
            );
        }
    }

    #[test]
    fn policy_round_trip_order_and_domain_hash_are_frozen() {
        let policy = frozen_replacement_completeness_policy();
        let bytes = serde_json::to_vec(&policy).expect("frozen policy must serialize");
        let decoded: ReplacementCompletenessPolicyV1 =
            serde_json::from_slice(&bytes).expect("frozen policy must deserialize");
        assert_eq!(decoded, policy);
        assert_eq!(
            policy.schema_version(),
            "replacement-completeness-policy/v1"
        );
        assert_eq!(policy.expected_cells().len(), 48);

        let hash = replacement_completeness_policy_sha256().expect("frozen policy must hash");
        assert_eq!(
            hash,
            "e174b05cdca40bef0418b298713b02196390bb0a41abbac1d74c6a7ecf5d3036"
        );
        assert_eq!(
            replacement_completeness_policy_sha256().expect("repeat hash must succeed"),
            hash
        );

        let mut seed_mutation = policy.clone();
        seed_mutation.seed_schedule[0].seeds.corpus += 1;
        assert_ne!(
            hash_replacement_policy(REPLACEMENT_COMPLETENESS_POLICY_HASH_DOMAIN, &seed_mutation)
                .expect("mutated policy must hash"),
            hash
        );

        let mut order_mutation = policy.clone();
        order_mutation.expected_cells.reverse();
        assert_ne!(
            hash_replacement_policy(REPLACEMENT_COMPLETENESS_POLICY_HASH_DOMAIN, &order_mutation)
                .expect("reordered policy must hash"),
            hash
        );

        assert_ne!(
            hash_replacement_policy(b"replacement-completeness-policy/v2\0", &policy)
                .expect("domain-separated policy must hash"),
            hash
        );
    }
}
