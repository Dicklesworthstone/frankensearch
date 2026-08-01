//! Central typed semantic [`RecoveryPlan`] and truthful readiness planner
//! (bd-vmv7).
//!
//! One versioned, machine-readable contract that maps the current semantic
//! readiness state plus the requested mode and policy to a truthful next
//! action. Products (facade, fsfs, CASS) consume this shared type instead of
//! parsing error strings; the terminal integration bead wires state
//! producers and executors around it.
//!
//! # Truthfulness invariants
//!
//! - Installing a model never claims semantic-searchable: the acquire
//!   action's postcondition is model-acquired-unverified, and readiness
//!   additionally requires a load self-test and a compatible non-empty
//!   index. This is enforced by test, not convention.
//! - Explicit semantic requests fail closed for every unavailable state.
//!   Typed partial-quality coverage remains semantically available at its
//!   measured coverage; hybrid requests may otherwise proceed lexical-only,
//!   but only with a [`SemanticResponseContract`] that names the requested
//!   and realized topologies, reports zero coverage, and admits zero semantic
//!   scores.
//! - The planner is pure and exhaustive: every enumerated state is matched
//!   without wildcards, so a new readiness state fails compilation until it
//!   is planned for.
//! - Model acquisition is always scoped to one logical model, tier, complete
//!   mathematical embedding-space identity, frozen manifest, revision,
//!   license assertion, source, byte budget, path-free destination identity,
//!   document census, and caller-computed reindex estimate. An authorization
//!   for any other scope is not interchangeable.
//! - Serialized plans are deliberately untrusted and non-executable. A caller
//!   must validate one against independently obtained readiness, request,
//!   policy, and frozen-target inputs; success returns a newly planned
//!   [`RecoveryPlan`] rather than promoting payload fields.
//! - A recovery action exposes argv only when the current product command
//!   enforces the complete authorization and semantic identity contract.
//!   Model acquisition, daemon/ANN repair, and all semantic index mutations
//!   are capability-blocked in this core-only tranche because current generic
//!   commands do not provide that binding.
//!
//! # Why schema v4
//!
//! The original v1 foundation represented offline recovery as a blocked
//! network download, represented request mode without retrieval topology,
//! and had no producer provenance, response-admission contract, or scoped
//! acquisition authorization. Correcting those facts changes required wire
//! fields and reverses the meaning of the offline transition, so decoding the
//! v2 contract as v1 would be unsafe. V3 additionally bound acquisition
//! consent to the exact model ID, tier, and mathematical space plus the
//! caller-supplied document count and estimated reindex duration, and refuses
//! to emit acquisition argv until an executor can consume that entire scope.
//! It also separated untrusted wire payloads from executable plans and pinned
//! recovery-local tier values to lowercase `fast` / `quality` spellings
//! instead of inheriting the Rust enum's incidental serde representation.
//! V4 makes every acquisition authorization short-lived, nonce-bound, and
//! evaluated only against caller-supplied trusted time. A v3 client cannot
//! safely present or validate those required anti-replay facts, so v4
//! deliberately fails closed on older payloads instead of installing a
//! compatibility shim. Every v1/v2/v3 stable
//! state/action/postcondition/policy code remains unchanged.
//!
//! # Stable codes
//!
//! Every state, action, postcondition, and policy-prerequisite code is a
//! three-segment lowercase dotted identifier (the same format the
//! observability lint enforces for reason codes) and is append-only within
//! a schema version. All codes live in this module so the full table is
//! auditable in one place; a test validates format and uniqueness against
//! [`crate::decision_plane::ReasonCode`] rules.

use serde::{Deserialize, Serialize};

use crate::{
    config::ZeroSignalReason,
    generation::{EmbeddingSpaceIdentityV1, EmbeddingSpaceKindV1},
    traits::ModelTier,
    types::{RetrievalTopology, retrieval_topology_fits_request},
};

mod recovery_model_tier_wire {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use crate::traits::ModelTier;

    #[derive(Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum WireModelTier {
        Fast,
        Quality,
    }

    // Serde's `with` module contract passes the field by reference even
    // though ModelTier is Copy.
    #[allow(clippy::trivially_copy_pass_by_ref)]
    pub fn serialize<S>(tier: &ModelTier, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match tier {
            ModelTier::Fast => WireModelTier::Fast,
            ModelTier::Quality => WireModelTier::Quality,
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<ModelTier, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(match WireModelTier::deserialize(deserializer)? {
            WireModelTier::Fast => ModelTier::Fast,
            WireModelTier::Quality => ModelTier::Quality,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
enum WireField<T> {
    #[default]
    Missing,
    Present(T),
}

impl<'de, T> Deserialize<'de> for WireField<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        T::deserialize(deserializer).map(Self::Present)
    }
}

impl<T> WireField<Option<T>> {
    fn required_option_ref(&self) -> Result<Option<&T>, ()> {
        match self {
            Self::Missing => Err(()),
            Self::Present(value) => Ok(value.as_ref()),
        }
    }
}

/// Schema version for serialized [`RecoveryPlan`] payloads.
pub const RECOVERY_PLAN_SCHEMA_VERSION: &str = "frankensearch.recovery_plan.v4";

/// Schema version for a scoped [`ModelAcquisitionAuthorization`].
pub const MODEL_ACQUISITION_AUTHORIZATION_SCHEMA_VERSION: &str =
    "frankensearch.model_acquisition_authorization.v3";

/// Maximum lifetime of one exact model-acquisition authorization.
///
/// The planner is clock-free: callers freeze issuance, expiry, and trusted
/// evaluation time. This bound limits replay exposure but does not claim
/// single-use semantics; an executor promising single use must additionally
/// consume nonces atomically.
pub const MAX_MODEL_ACQUISITION_AUTHORIZATION_LIFETIME_SECONDS: u64 = 15 * 60;

/// One million parts per million: complete semantic document coverage.
pub const COMPLETE_COVERAGE_PPM: u32 = 1_000_000;

/// Placeholder token integrators substitute with the resolved index
/// directory.
///
/// The pure planner never sees real user paths (they are redacted from
/// telemetry); rendering a runnable command is the integrator's job.
pub const ARG_INDEX_DIR: &str = "<index-dir>";

/// Placeholder token integrators substitute with the corpus source
/// directory to (re-)ingest.
pub const ARG_SOURCE_DIR: &str = "<source-dir>";

/// Reserved placeholder for a future operator-supplied local model bundle.
///
/// The current fsfs parser does not implement offline model import, so the
/// planner never emits this token in executable argv. It remains a public
/// schema vocabulary item for the future capability rather than pretending a
/// fictional command is runnable today.
pub const ARG_MODEL_BUNDLE: &str = "ARG_MODEL_BUNDLE";

/// Wire discriminator for [`RecoveryPlan`].
///
/// Using a closed enum rather than an arbitrary string makes serde reject
/// older or unknown schemas before a caller can accidentally execute their
/// actions with v4 semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryPlanSchemaVersion {
    #[serde(rename = "frankensearch.recovery_plan.v4")]
    V4,
}

/// Wire discriminator for [`ModelAcquisitionAuthorization`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelAcquisitionAuthorizationSchemaVersion {
    #[serde(rename = "frankensearch.model_acquisition_authorization.v3")]
    V3,
}

/// Verified producer provenance for a semantic-ready lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerifiedSemanticProvenance {
    /// Frozen local artifacts passed manifest verification and load self-test.
    Local,
    /// A remote producer passed the pinned response-attestation contract.
    Remote,
    /// A daemon producer passed the pinned daemon-attestation contract.
    Daemon,
}

/// Trust classification exposed with every recovery plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticProvenance {
    VerifiedLocal,
    VerifiedRemote,
    VerifiedDaemon,
    /// Explicit remote intent exists, but its producer space is not attested.
    UnverifiedRemote,
    /// Explicit non-semantic hash test/control lane.
    HashControl,
    /// No producer is currently admissible.
    Unavailable,
}

impl From<VerifiedSemanticProvenance> for SemanticProvenance {
    fn from(value: VerifiedSemanticProvenance) -> Self {
        match value {
            VerifiedSemanticProvenance::Local => Self::VerifiedLocal,
            VerifiedSemanticProvenance::Remote => Self::VerifiedRemote,
            VerifiedSemanticProvenance::Daemon => Self::VerifiedDaemon,
        }
    }
}

/// Why the caller's request cannot be served semantically right now.
///
/// This is the planner's input state, produced by readiness probes
/// (model manifest checks, index census, generation binding). Variants
/// mirror the states bd-vmv7 enumerates; [`ZeroSignalReason`] carries the
/// finer classification for empty-index states so the two vocabularies
/// stay aligned rather than diverging.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "state", content = "detail")]
pub enum SemanticReadiness {
    /// The semantic lane is fully usable: verified model, loadable, and a
    /// compatible index with usable live vectors.
    Ready {
        provenance: VerifiedSemanticProvenance,
    },
    /// No usable model artifact exists in the configured cache for this exact
    /// progressive tier.
    ModelMissing {
        #[serde(with = "recovery_model_tier_wire")]
        tier: ModelTier,
    },
    /// A model artifact exists for this exact progressive tier but failed
    /// verification or load self-test.
    ModelUnloadable {
        #[serde(with = "recovery_model_tier_wire")]
        tier: ModelTier,
    },
    /// A verified, loadable model exists but no vector index does.
    IndexAbsent,
    /// The index exists but its embedding-space identity does not match the
    /// configured model (legacy generation or intentional identity change).
    IdentityMismatch,
    /// A daemon serves a different embedding space than the local
    /// configuration expects.
    DaemonMismatch,
    /// The index exists and is readable but produced zero signal; the typed
    /// reason distinguishes benign emptiness from availability failures.
    IndexEmpty(ZeroSignalReason),
    /// The index or model manifest is corrupt or fails safety validation
    /// and must not be trusted.
    ManifestUnsafe,
    /// The ANN sidecar belongs to an older generation than the vector
    /// index; exact search works but ANN must not serve.
    AnnStale,
    /// An index generation was interrupted before publication.
    GenerationIncomplete,
    /// Fast-tier search works but some records lack quality-tier
    /// embeddings, so refinement coverage is partial.
    PartialQualityCoverage {
        provenance: VerifiedSemanticProvenance,
        /// Fraction of live documents with quality-tier embeddings.
        coverage_ppm: u32,
    },
    /// Explicit remote intent exists, but the producer cannot be attested.
    /// This state is durable, non-indexable, and never becomes local/hash
    /// fallback inside the planner.
    RemoteUnverified,
    /// Explicit non-semantic hash test/control lane.
    HashControl,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadyDetailWire {
    provenance: VerifiedSemanticProvenance,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelTierDetailWire {
    #[serde(with = "recovery_model_tier_wire")]
    tier: ModelTier,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PartialQualityCoverageDetailWire {
    provenance: VerifiedSemanticProvenance,
    coverage_ppm: u32,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum SemanticReadinessTagWire {
    Ready,
    ModelMissing,
    ModelUnloadable,
    IndexAbsent,
    IdentityMismatch,
    DaemonMismatch,
    IndexEmpty,
    ManifestUnsafe,
    AnnStale,
    GenerationIncomplete,
    PartialQualityCoverage,
    RemoteUnverified,
    HashControl,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum SemanticReadinessDetailWire {
    Ready(ReadyDetailWire),
    ModelTier(ModelTierDetailWire),
    PartialQualityCoverage(PartialQualityCoverageDetailWire),
    ZeroSignal(ZeroSignalReason),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SemanticReadinessWire {
    state: SemanticReadinessTagWire,
    #[serde(default)]
    detail: WireField<SemanticReadinessDetailWire>,
}

impl<'de> Deserialize<'de> for SemanticReadiness {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = SemanticReadinessWire::deserialize(deserializer)?;
        match (wire.state, wire.detail) {
            (
                SemanticReadinessTagWire::Ready,
                WireField::Present(SemanticReadinessDetailWire::Ready(ReadyDetailWire {
                    provenance,
                })),
            ) => Ok(Self::Ready { provenance }),
            (
                SemanticReadinessTagWire::ModelMissing,
                WireField::Present(SemanticReadinessDetailWire::ModelTier(ModelTierDetailWire {
                    tier,
                })),
            ) => Ok(Self::ModelMissing { tier }),
            (
                SemanticReadinessTagWire::ModelUnloadable,
                WireField::Present(SemanticReadinessDetailWire::ModelTier(ModelTierDetailWire {
                    tier,
                })),
            ) => Ok(Self::ModelUnloadable { tier }),
            (SemanticReadinessTagWire::IndexAbsent, WireField::Missing) => Ok(Self::IndexAbsent),
            (SemanticReadinessTagWire::IdentityMismatch, WireField::Missing) => {
                Ok(Self::IdentityMismatch)
            }
            (SemanticReadinessTagWire::DaemonMismatch, WireField::Missing) => {
                Ok(Self::DaemonMismatch)
            }
            (
                SemanticReadinessTagWire::IndexEmpty,
                WireField::Present(SemanticReadinessDetailWire::ZeroSignal(reason)),
            ) => Ok(Self::IndexEmpty(reason)),
            (SemanticReadinessTagWire::ManifestUnsafe, WireField::Missing) => {
                Ok(Self::ManifestUnsafe)
            }
            (SemanticReadinessTagWire::AnnStale, WireField::Missing) => Ok(Self::AnnStale),
            (SemanticReadinessTagWire::GenerationIncomplete, WireField::Missing) => {
                Ok(Self::GenerationIncomplete)
            }
            (
                SemanticReadinessTagWire::PartialQualityCoverage,
                WireField::Present(SemanticReadinessDetailWire::PartialQualityCoverage(
                    PartialQualityCoverageDetailWire {
                        provenance,
                        coverage_ppm,
                    },
                )),
            ) => Ok(Self::PartialQualityCoverage {
                provenance,
                coverage_ppm,
            }),
            (SemanticReadinessTagWire::RemoteUnverified, WireField::Missing) => {
                Ok(Self::RemoteUnverified)
            }
            (SemanticReadinessTagWire::HashControl, WireField::Missing) => Ok(Self::HashControl),
            _ => Err(serde::de::Error::custom(
                "readiness detail is missing, forbidden, or inconsistent with state",
            )),
        }
    }
}

impl SemanticReadiness {
    /// Stable three-segment state code.
    #[must_use]
    pub const fn state_code(&self) -> &'static str {
        match self {
            Self::Ready { .. } => "recovery.state.ready",
            Self::ModelMissing { .. } => "recovery.state.model_missing",
            Self::ModelUnloadable { .. } => "recovery.state.model_unloadable",
            Self::IndexAbsent => "recovery.state.index_absent",
            Self::IdentityMismatch => "recovery.state.identity_mismatch",
            Self::DaemonMismatch => "recovery.state.daemon_mismatch",
            Self::IndexEmpty(_) => "recovery.state.index_empty",
            Self::ManifestUnsafe => "recovery.state.manifest_unsafe",
            Self::AnnStale => "recovery.state.ann_stale",
            Self::GenerationIncomplete => "recovery.state.generation_incomplete",
            Self::PartialQualityCoverage { .. } => "recovery.state.partial_quality_coverage",
            Self::RemoteUnverified => "recovery.state.remote_unverified",
            Self::HashControl => "recovery.state.hash_control",
        }
    }

    /// True when semantic results can be served (possibly with reduced
    /// refinement quality). Only [`Self::Ready`] and
    /// [`Self::PartialQualityCoverage`] qualify: partial coverage degrades
    /// refinement, not availability.
    #[must_use]
    pub const fn semantic_available(&self) -> bool {
        matches!(
            self,
            Self::Ready { .. } | Self::PartialQualityCoverage { .. }
        )
    }

    /// Producer trust associated with the current readiness state.
    #[must_use]
    pub const fn provenance(&self) -> SemanticProvenance {
        match self {
            Self::Ready { provenance } | Self::PartialQualityCoverage { provenance, .. } => {
                match provenance {
                    VerifiedSemanticProvenance::Local => SemanticProvenance::VerifiedLocal,
                    VerifiedSemanticProvenance::Remote => SemanticProvenance::VerifiedRemote,
                    VerifiedSemanticProvenance::Daemon => SemanticProvenance::VerifiedDaemon,
                }
            }
            Self::RemoteUnverified => SemanticProvenance::UnverifiedRemote,
            Self::HashControl => SemanticProvenance::HashControl,
            Self::ModelMissing { .. }
            | Self::ModelUnloadable { .. }
            | Self::IndexAbsent
            | Self::IdentityMismatch
            | Self::DaemonMismatch
            | Self::IndexEmpty(_)
            | Self::ManifestUnsafe
            | Self::AnnStale
            | Self::GenerationIncomplete => SemanticProvenance::Unavailable,
        }
    }
}

/// What the caller asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestMode {
    /// The caller explicitly requires semantic results: fail closed when
    /// the lane is unavailable.
    ExplicitSemantic,
    /// The caller accepts hybrid results: lexical fallback is permitted,
    /// but only with explicit degradation metadata.
    Hybrid,
    /// Explicit non-semantic hash test/control request.
    HashControl,
}

/// Requested operation, including the exact retrieval topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct RecoveryRequest {
    pub mode: RequestMode,
    pub requested_topology: RetrievalTopology,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecoveryRequestWire {
    mode: RequestMode,
    requested_topology: RetrievalTopology,
}

impl<'de> Deserialize<'de> for RecoveryRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = RecoveryRequestWire::deserialize(deserializer)?;
        Self {
            mode: wire.mode,
            requested_topology: wire.requested_topology,
        }
        .validate()
        .map_err(serde::de::Error::custom)
    }
}

impl RecoveryRequest {
    /// Validate that mode and topology describe one unambiguous request.
    ///
    /// `PartialQuality` and `LexicalOnly` are realized topologies, never
    /// semantic request targets. Hash is legal only through the explicit
    /// `HashControl` mode.
    ///
    /// # Errors
    ///
    /// Returns [`RecoveryContractError::InvalidRequestTopology`] for an
    /// ambiguous or silently degrading combination.
    pub fn validate(self) -> Result<Self, RecoveryContractError> {
        let valid = match self.mode {
            RequestMode::ExplicitSemantic | RequestMode::Hybrid => matches!(
                self.requested_topology,
                RetrievalTopology::FastOnly
                    | RetrievalTopology::QualityOnly
                    | RetrievalTopology::FullProgressive
            ),
            RequestMode::HashControl => {
                matches!(self.requested_topology, RetrievalTopology::HashControl)
            }
        };
        if valid {
            Ok(self)
        } else {
            Err(RecoveryContractError::InvalidRequestTopology {
                mode: self.mode,
                topology: self.requested_topology,
            })
        }
    }
}

/// Whether a human can be asked for consent right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractionPolicy {
    Interactive,
    NonInteractive,
}

/// Whether network access is permitted for recovery actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NetworkPolicy {
    Allowed,
    Offline,
}

/// The caller's environment policy, combined.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RecoveryPolicy {
    pub interaction: InteractionPolicy,
    pub network: NetworkPolicy,
    /// Exact non-TTY/programmatic model-acquisition authorization, when
    /// already granted. It satisfies consent only when byte-for-byte equal
    /// to the action's required authorization.
    pub acquisition_authorization: Option<ModelAcquisitionAuthorization>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecoveryPolicyWire {
    interaction: InteractionPolicy,
    network: NetworkPolicy,
    #[serde(default)]
    acquisition_authorization: WireField<Option<ModelAcquisitionAuthorization>>,
}

impl<'de> Deserialize<'de> for RecoveryPolicy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = RecoveryPolicyWire::deserialize(deserializer)?;
        let WireField::Present(acquisition_authorization) = wire.acquisition_authorization else {
            return Err(serde::de::Error::missing_field("acquisition_authorization"));
        };
        Ok(Self {
            interaction: wire.interaction,
            network: wire.network,
            acquisition_authorization,
        })
    }
}

/// Path-free class of destination bound by model-acquisition consent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelDestinationClass {
    /// Product-managed model cache.
    ManagedCache,
    /// Caller-selected model directory outside the managed cache.
    ExplicitDirectory,
}

/// Machine-distinguishable byte source authorized for acquisition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ModelAcquisitionSource {
    /// Immutable HTTPS sources named by credential-free host.
    Network { source_hosts: Vec<String> },
    /// Complete operator-supplied artifact tree.
    LocalBundle,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum ModelAcquisitionSourceTagWire {
    Network,
    LocalBundle,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelAcquisitionSourceWire {
    kind: ModelAcquisitionSourceTagWire,
    #[serde(default)]
    source_hosts: WireField<Vec<String>>,
}

impl<'de> Deserialize<'de> for ModelAcquisitionSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = ModelAcquisitionSourceWire::deserialize(deserializer)?;
        match (wire.kind, wire.source_hosts) {
            (ModelAcquisitionSourceTagWire::Network, WireField::Present(source_hosts)) => {
                Ok(Self::Network { source_hosts })
            }
            (ModelAcquisitionSourceTagWire::Network, WireField::Missing) => {
                Err(serde::de::Error::missing_field("source_hosts"))
            }
            (ModelAcquisitionSourceTagWire::LocalBundle, WireField::Missing) => {
                Ok(Self::LocalBundle)
            }
            (ModelAcquisitionSourceTagWire::LocalBundle, WireField::Present(_)) => Err(
                serde::de::Error::custom("source_hosts is forbidden for local_bundle source"),
            ),
        }
    }
}

/// Exact, path-free authorization required before model bytes are acquired.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelAcquisitionAuthorization {
    pub schema_version: ModelAcquisitionAuthorizationSchemaVersion,
    /// Stable logical ID passed to the exact-model acquisition command.
    pub model_id: String,
    /// Progressive tier this artifact will serve.
    #[serde(with = "recovery_model_tier_wire")]
    pub model_tier: ModelTier,
    /// Complete mathematical identity of the vectors this model produces.
    ///
    /// A model name, revision, or dimension alone never establishes space
    /// compatibility.
    pub embedding_space: EmbeddingSpaceIdentityV1,
    pub manifest_fingerprint: String,
    pub upstream_revision: String,
    pub license_spdx: String,
    pub source: ModelAcquisitionSource,
    pub byte_budget: u64,
    pub destination_class: ModelDestinationClass,
    /// Bounded hash of the canonical destination, never the raw path.
    pub destination_fingerprint: String,
    /// Exact corpus size shown when consent is requested.
    pub document_count: u64,
    /// Caller-supplied estimate of the reindex wall-clock cost.
    ///
    /// The unit is explicit so renderers never guess. The pure planner does
    /// not derive this value from ambient telemetry or filesystem state.
    pub estimated_reindex_duration_ms: u64,
    /// Caller-frozen issuance time for this exact authorization.
    pub issued_at_unix_seconds: u64,
    /// Caller-frozen exclusive expiry time for this exact authorization.
    pub expires_at_unix_seconds: u64,
    /// Caller-generated 128-bit nonce encoded as exactly 32 lowercase
    /// hexadecimal characters.
    pub nonce: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelAcquisitionAuthorizationWire {
    schema_version: ModelAcquisitionAuthorizationSchemaVersion,
    model_id: String,
    #[serde(with = "recovery_model_tier_wire")]
    model_tier: ModelTier,
    embedding_space: EmbeddingSpaceIdentityV1,
    manifest_fingerprint: String,
    upstream_revision: String,
    license_spdx: String,
    source: ModelAcquisitionSource,
    byte_budget: u64,
    destination_class: ModelDestinationClass,
    destination_fingerprint: String,
    document_count: u64,
    estimated_reindex_duration_ms: u64,
    issued_at_unix_seconds: u64,
    expires_at_unix_seconds: u64,
    nonce: String,
}

impl<'de> Deserialize<'de> for ModelAcquisitionAuthorization {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = ModelAcquisitionAuthorizationWire::deserialize(deserializer)?;
        let authorization = Self {
            schema_version: wire.schema_version,
            model_id: wire.model_id,
            model_tier: wire.model_tier,
            embedding_space: wire.embedding_space,
            manifest_fingerprint: wire.manifest_fingerprint,
            upstream_revision: wire.upstream_revision,
            license_spdx: wire.license_spdx,
            source: wire.source,
            byte_budget: wire.byte_budget,
            destination_class: wire.destination_class,
            destination_fingerprint: wire.destination_fingerprint,
            document_count: wire.document_count,
            estimated_reindex_duration_ms: wire.estimated_reindex_duration_ms,
            issued_at_unix_seconds: wire.issued_at_unix_seconds,
            expires_at_unix_seconds: wire.expires_at_unix_seconds,
            nonce: wire.nonce,
        };
        authorization.validate().map_err(serde::de::Error::custom)?;
        Ok(authorization)
    }
}

impl ModelAcquisitionAuthorization {
    /// Validate that every exact authorization scope is present and usable.
    ///
    /// # Errors
    ///
    /// Rejects blank or control-bearing scope fields, zero byte budgets,
    /// network acquisition without at least one source host, and source hosts
    /// that are not credential-free DNS names, IPv4 addresses, or bracketed
    /// IPv6 addresses (each optionally followed by an explicit non-zero port).
    pub fn validate(&self) -> Result<(), RecoveryContractError> {
        validate_scope_text("model_id", &self.model_id)?;
        self.embedding_space.validate().map_err(|error| {
            RecoveryContractError::InvalidAcquisitionSpaceIdentity {
                reason: error.to_string(),
            }
        })?;
        if self.embedding_space.kind != EmbeddingSpaceKindV1::Semantic {
            return Err(RecoveryContractError::NonSemanticAcquisitionSpace);
        }
        if self.model_id != self.embedding_space.logical_model_id {
            return Err(RecoveryContractError::InconsistentAcquisitionIdentity {
                field: "model_id",
            });
        }
        validate_scope_text("manifest_fingerprint", &self.manifest_fingerprint)?;
        validate_scope_text("upstream_revision", &self.upstream_revision)?;
        if self.upstream_revision != self.embedding_space.immutable_revision {
            return Err(RecoveryContractError::InconsistentAcquisitionIdentity {
                field: "upstream_revision",
            });
        }
        validate_scope_text("license_spdx", &self.license_spdx)?;
        validate_scope_text("destination_fingerprint", &self.destination_fingerprint)?;
        if self.byte_budget == 0 {
            return Err(RecoveryContractError::ZeroAcquisitionByteBudget);
        }
        if let ModelAcquisitionSource::Network { source_hosts } = &self.source {
            if source_hosts.is_empty() {
                return Err(RecoveryContractError::MissingNetworkSourceHosts);
            }
            for host in source_hosts {
                validate_network_source_host(host)?;
            }
        }
        validate_acquisition_authorization_window(
            self.issued_at_unix_seconds,
            self.expires_at_unix_seconds,
        )?;
        validate_acquisition_authorization_nonce(&self.nonce)?;
        Ok(())
    }

    /// Revalidate this authorization against caller-supplied trusted time.
    ///
    /// Executors must call this immediately before the first acquisition side
    /// effect. A prior planning or wire-promotion check does not keep an
    /// authorization valid after its exclusive expiry boundary.
    ///
    /// # Errors
    ///
    /// Returns the same structural validation errors as [`Self::validate`],
    /// [`RecoveryContractError::AcquisitionAuthorizationNotYetValid`] before
    /// issuance, or [`RecoveryContractError::AcquisitionAuthorizationExpired`]
    /// at and after expiry.
    pub fn validate_at(
        &self,
        evaluation_time_unix_seconds: u64,
    ) -> Result<(), RecoveryContractError> {
        self.validate()?;
        if evaluation_time_unix_seconds < self.issued_at_unix_seconds {
            return Err(RecoveryContractError::AcquisitionAuthorizationNotYetValid {
                issued_at_unix_seconds: self.issued_at_unix_seconds,
                evaluation_time_unix_seconds,
            });
        }
        if evaluation_time_unix_seconds >= self.expires_at_unix_seconds {
            return Err(RecoveryContractError::AcquisitionAuthorizationExpired {
                expires_at_unix_seconds: self.expires_at_unix_seconds,
                evaluation_time_unix_seconds,
            });
        }
        Ok(())
    }
}

/// Source-independent target from which the planner derives the exact
/// network or local-bundle authorization required by policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelAcquisitionTarget {
    /// Stable logical ID selected by the caller's frozen manifest.
    pub model_id: String,
    /// Progressive tier selected by the caller's requested topology.
    pub model_tier: ModelTier,
    /// Complete mathematical identity selected by the caller's frozen
    /// manifest and readiness probe.
    pub embedding_space: EmbeddingSpaceIdentityV1,
    pub manifest_fingerprint: String,
    pub upstream_revision: String,
    pub license_spdx: String,
    pub network_source_hosts: Vec<String>,
    pub byte_budget: u64,
    pub destination_class: ModelDestinationClass,
    pub destination_fingerprint: String,
    /// Corpus census supplied by the caller for consent presentation.
    pub document_count: u64,
    /// Caller-computed reindex estimate, explicitly expressed in
    /// milliseconds. The planner never derives or adjusts it.
    pub estimated_reindex_duration_ms: u64,
    /// Caller-frozen issuance time copied into the exact authorization.
    pub issued_at_unix_seconds: u64,
    /// Caller-frozen exclusive expiry time copied into the exact
    /// authorization.
    pub expires_at_unix_seconds: u64,
    /// Caller-generated 128-bit lowercase-hex nonce copied into the exact
    /// authorization.
    pub nonce: String,
}

impl ModelAcquisitionTarget {
    fn authorization_for(
        &self,
        network: NetworkPolicy,
    ) -> Result<ModelAcquisitionAuthorization, RecoveryContractError> {
        let authorization = ModelAcquisitionAuthorization {
            schema_version: ModelAcquisitionAuthorizationSchemaVersion::V3,
            model_id: self.model_id.clone(),
            model_tier: self.model_tier,
            embedding_space: self.embedding_space.clone(),
            manifest_fingerprint: self.manifest_fingerprint.clone(),
            upstream_revision: self.upstream_revision.clone(),
            license_spdx: self.license_spdx.clone(),
            source: match network {
                NetworkPolicy::Allowed => ModelAcquisitionSource::Network {
                    source_hosts: self.network_source_hosts.clone(),
                },
                NetworkPolicy::Offline => ModelAcquisitionSource::LocalBundle,
            },
            byte_budget: self.byte_budget,
            destination_class: self.destination_class,
            destination_fingerprint: self.destination_fingerprint.clone(),
            document_count: self.document_count,
            estimated_reindex_duration_ms: self.estimated_reindex_duration_ms,
            issued_at_unix_seconds: self.issued_at_unix_seconds,
            expires_at_unix_seconds: self.expires_at_unix_seconds,
            nonce: self.nonce.clone(),
        };
        authorization.validate()?;
        Ok(authorization)
    }
}

fn validate_acquisition_authorization_window(
    issued_at_unix_seconds: u64,
    expires_at_unix_seconds: u64,
) -> Result<(), RecoveryContractError> {
    if expires_at_unix_seconds <= issued_at_unix_seconds {
        return Err(
            RecoveryContractError::InvalidAcquisitionAuthorizationWindow {
                issued_at_unix_seconds,
                expires_at_unix_seconds,
            },
        );
    }
    let lifetime_seconds = expires_at_unix_seconds - issued_at_unix_seconds;
    if lifetime_seconds > MAX_MODEL_ACQUISITION_AUTHORIZATION_LIFETIME_SECONDS {
        return Err(
            RecoveryContractError::AcquisitionAuthorizationLifetimeExceeded {
                lifetime_seconds,
                max_lifetime_seconds: MAX_MODEL_ACQUISITION_AUTHORIZATION_LIFETIME_SECONDS,
            },
        );
    }
    Ok(())
}

fn validate_acquisition_authorization_nonce(nonce: &str) -> Result<(), RecoveryContractError> {
    let valid_shape = nonce.len() == 32
        && nonce
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    let nonzero = nonce.bytes().any(|byte| byte != b'0');
    if !valid_shape || !nonzero {
        return Err(RecoveryContractError::InvalidAcquisitionAuthorizationNonce);
    }
    Ok(())
}

fn validate_scope_text(field: &'static str, value: &str) -> Result<(), RecoveryContractError> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(RecoveryContractError::InvalidAcquisitionScopeField { field });
    }
    Ok(())
}

fn validate_network_source_host(host: &str) -> Result<(), RecoveryContractError> {
    let invalid = || RecoveryContractError::InvalidNetworkSourceHost;
    if host.is_empty()
        || host.chars().any(|character| {
            character.is_whitespace()
                || character.is_control()
                || matches!(character, '@' | '/' | '?' | '#' | '\\')
        })
        || host.contains("://")
    {
        return Err(invalid());
    }

    if let Some(bracketed) = host.strip_prefix('[') {
        let Some(closing_bracket) = bracketed.find(']') else {
            return Err(invalid());
        };
        let address = &bracketed[..closing_bracket];
        let suffix = &bracketed[closing_bracket + 1..];
        address
            .parse::<std::net::Ipv6Addr>()
            .map_err(|_| invalid())?;
        if !suffix.is_empty() {
            let port = suffix.strip_prefix(':').ok_or_else(invalid)?;
            validate_network_source_port(port)?;
        }
        return Ok(());
    }

    if host.contains(['[', ']']) || host.bytes().filter(|byte| *byte == b':').count() > 1 {
        return Err(invalid());
    }
    let (address, port) = match host.rsplit_once(':') {
        Some((address, port)) => (address, Some(port)),
        None => (host, None),
    };
    if let Some(port) = port {
        validate_network_source_port(port)?;
    }
    if address.parse::<std::net::Ipv4Addr>().is_ok() {
        return Ok(());
    }
    if address
        .bytes()
        .all(|byte| byte.is_ascii_digit() || byte == b'.')
        || address.len() > 253
        || address.ends_with('.')
    {
        return Err(invalid());
    }
    for label in address.split('.') {
        if label.is_empty()
            || label.len() > 63
            || !label
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
            || !label
                .as_bytes()
                .first()
                .is_some_and(u8::is_ascii_alphanumeric)
            || !label
                .as_bytes()
                .last()
                .is_some_and(u8::is_ascii_alphanumeric)
        {
            return Err(invalid());
        }
    }
    Ok(())
}

fn validate_network_source_port(port: &str) -> Result<(), RecoveryContractError> {
    if port.is_empty() || !port.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(RecoveryContractError::InvalidNetworkSourceHost);
    }
    let parsed = port
        .parse::<u16>()
        .map_err(|_| RecoveryContractError::InvalidNetworkSourceHost)?;
    if parsed == 0 {
        return Err(RecoveryContractError::InvalidNetworkSourceHost);
    }
    Ok(())
}

/// Typed validation failure for a recovery request or response contract.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RecoveryContractError {
    #[error("request mode {mode:?} cannot request topology {topology:?}")]
    InvalidRequestTopology {
        mode: RequestMode,
        topology: RetrievalTopology,
    },
    #[error("hash-control request mode and readiness must be selected together")]
    HashControlModeReadinessMismatch,
    #[error("acquisition scope field `{field}` is empty or contains control characters")]
    InvalidAcquisitionScopeField { field: &'static str },
    #[error("model acquisition space identity is invalid: {reason}")]
    InvalidAcquisitionSpaceIdentity { reason: String },
    #[error("model acquisition requires a semantic embedding space")]
    NonSemanticAcquisitionSpace,
    #[error("acquisition identity field `{field}` conflicts with the complete space identity")]
    InconsistentAcquisitionIdentity { field: &'static str },
    #[error("readiness tier {tier} cannot satisfy requested topology {requested_topology:?}")]
    UnavailableTierTopologyMismatch {
        tier: ModelTier,
        requested_topology: RetrievalTopology,
    },
    #[error("acquisition target tier {target_tier} does not match readiness tier {readiness_tier}")]
    AcquisitionTargetTierMismatch {
        readiness_tier: ModelTier,
        target_tier: ModelTier,
    },
    #[error("model acquisition byte budget must be non-zero")]
    ZeroAcquisitionByteBudget,
    #[error(
        "model acquisition authorization window is invalid: issued_at={issued_at_unix_seconds}, \
         expires_at={expires_at_unix_seconds}"
    )]
    InvalidAcquisitionAuthorizationWindow {
        issued_at_unix_seconds: u64,
        expires_at_unix_seconds: u64,
    },
    #[error(
        "model acquisition authorization lifetime {lifetime_seconds}s exceeds the \
         {max_lifetime_seconds}s maximum"
    )]
    AcquisitionAuthorizationLifetimeExceeded {
        lifetime_seconds: u64,
        max_lifetime_seconds: u64,
    },
    #[error(
        "model acquisition authorization nonce must be a nonzero 128-bit value encoded as \
         exactly 32 lowercase hexadecimal characters"
    )]
    InvalidAcquisitionAuthorizationNonce,
    #[error(
        "model acquisition authorization is not yet valid: issued_at={issued_at_unix_seconds}, \
         evaluated_at={evaluation_time_unix_seconds}"
    )]
    AcquisitionAuthorizationNotYetValid {
        issued_at_unix_seconds: u64,
        evaluation_time_unix_seconds: u64,
    },
    #[error(
        "model acquisition authorization expired: expires_at={expires_at_unix_seconds}, \
         evaluated_at={evaluation_time_unix_seconds}"
    )]
    AcquisitionAuthorizationExpired {
        expires_at_unix_seconds: u64,
        evaluation_time_unix_seconds: u64,
    },
    #[error("model acquisition requires an exact supplied authorization before execution")]
    MissingAcquisitionAuthorization,
    #[error("model acquisition authorization was supplied when no exact acquisition requires it")]
    SurplusAcquisitionAuthorization,
    #[error("model acquisition authorization field `{field}` does not match the required scope")]
    MismatchedAcquisitionAuthorization { field: &'static str },
    #[error("network model acquisition requires at least one credential-free source host")]
    MissingNetworkSourceHosts,
    #[error(
        "network model acquisition source host must be a credential-free DNS name, IPv4 address, \
         or bracketed IPv6 address with an optional non-zero port"
    )]
    InvalidNetworkSourceHost,
    #[error("coverage_ppm {coverage_ppm} is outside the valid range for {topology:?}")]
    InvalidCoverage {
        topology: RetrievalTopology,
        coverage_ppm: u32,
    },
    #[error("requested topology {requested:?} cannot realize as {realized:?}")]
    IncompatibleResponseTopology {
        requested: RetrievalTopology,
        realized: RetrievalTopology,
    },
    #[error("non-semantic topology {topology:?} cannot admit {admitted} semantic scores")]
    NonSemanticScoresAdmitted {
        topology: RetrievalTopology,
        admitted: u64,
    },
    #[error("lexical-only response requires exactly one degradation reason code")]
    MissingDegradationReason,
    #[error("non-lexical response cannot carry a degradation reason code")]
    UnexpectedDegradationReason,
    #[error("recovery plan field `{field}` is inconsistent with the typed decision")]
    InconsistentRecoveryPlan { field: &'static str },
}

/// Whether retrying the original request can succeed, and under what
/// condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Retryability {
    /// The lane is usable; no recovery action is needed.
    NotNeeded,
    /// Retry after executing the recommended action.
    AfterAction,
    /// The current request itself produced no signal; retry only after
    /// changing its zero-k, filter, or vector input.
    AfterRequestChange,
    /// The recommended action cannot run under the current policy; the
    /// listed prerequisites must be granted first.
    BlockedByPolicy,
    /// The recommended action has no executable implementation in the
    /// current runtime. A listed capability must land before retrying.
    BlockedByCapability,
}

/// One truthful next action.
///
/// The four booleans are independent schema-mandated facts about the
/// action (bd-vmv7's field list), not an encoded state machine, so a
/// bitflag or enum representation would obscure the serialized contract.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RecoveryAction {
    /// Stable action code, append-only within a schema version.
    code: String,
    /// Human-readable explanation of what the action does and why.
    explanation: String,
    /// Command as an argv array. Placeholder tokens ([`ARG_INDEX_DIR`],
    /// [`ARG_SOURCE_DIR`]) are substituted by integrators; the pure
    /// planner never handles real paths.
    argv: Vec<String>,
    /// The action needs network access (model downloads).
    network_required: bool,
    /// The action needs explicit human consent (it replaces existing
    /// artifacts).
    consent_required: bool,
    /// The pre-existing data this action is intended to repair survives. For
    /// index recovery this means user documents and index contents; for model
    /// recovery it includes the cached model artifact itself. Reacquisition
    /// is therefore `false` even though corpus and index data remain intact.
    preserves_old_data: bool,
    /// The action replaces or rewrites existing artifacts.
    potentially_destructive: bool,
    /// Stable codes of conditions that must hold before the action can
    /// run (policy grants).
    prerequisites: Vec<String>,
    /// Stable code of the state expected after the action succeeds. Never
    /// `recovery.state.ready` for acquisition actions: readiness
    /// additionally requires the load self-test and a compatible
    /// non-empty index.
    expected_postcondition: String,
    /// Exact acquisition authorization this action requires. `None` for
    /// non-acquisition actions and when the caller failed to bind a frozen
    /// model target (which blocks the plan through a prerequisite).
    required_authorization: Option<ModelAcquisitionAuthorization>,
}

impl RecoveryAction {
    /// Stable action code.
    #[must_use]
    pub fn code(&self) -> &str {
        &self.code
    }

    /// Human-readable reason and effect.
    #[must_use]
    pub fn explanation(&self) -> &str {
        &self.explanation
    }

    /// Parser-executable argv, or an empty slice when prerequisites make the
    /// action unavailable.
    #[must_use]
    pub fn argv(&self) -> &[String] {
        &self.argv
    }

    /// Whether execution requires network access.
    #[must_use]
    pub const fn network_required(&self) -> bool {
        self.network_required
    }

    /// Whether execution requires explicit consent.
    #[must_use]
    pub const fn consent_required(&self) -> bool {
        self.consent_required
    }

    /// Whether the pre-existing data this action is intended to repair
    /// survives.
    #[must_use]
    pub const fn preserves_old_data(&self) -> bool {
        self.preserves_old_data
    }

    /// Whether the action rewrites or replaces artifacts.
    #[must_use]
    pub const fn potentially_destructive(&self) -> bool {
        self.potentially_destructive
    }

    /// Stable prerequisite codes that currently block execution.
    #[must_use]
    pub fn prerequisites(&self) -> &[String] {
        &self.prerequisites
    }

    /// Stable postcondition code expected after successful execution.
    #[must_use]
    pub fn expected_postcondition(&self) -> &str {
        &self.expected_postcondition
    }

    /// Exact scoped authorization required for acquisition.
    #[must_use]
    pub const fn required_authorization(&self) -> Option<&ModelAcquisitionAuthorization> {
        self.required_authorization.as_ref()
    }

    /// Render the argv for a POSIX shell, quoting every argument that
    /// contains characters beyond `[A-Za-z0-9_./:=-]` and the placeholder
    /// tokens (which are documentation, not shell input).
    #[must_use]
    pub fn shell_command(&self) -> String {
        self.argv
            .iter()
            .map(|arg| shell_quote(arg))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

// This raw shape mirrors the same independent schema-mandated facts as
// RecoveryAction; collapsing them would make wire validation less explicit.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecoveryActionWire {
    code: String,
    explanation: String,
    argv: Vec<String>,
    network_required: bool,
    consent_required: bool,
    preserves_old_data: bool,
    potentially_destructive: bool,
    prerequisites: Vec<String>,
    expected_postcondition: String,
    #[serde(default)]
    required_authorization: WireField<Option<ModelAcquisitionAuthorization>>,
}

fn shell_quote(arg: &str) -> String {
    let safe = !arg.is_empty()
        && arg.chars().all(|c| {
            c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '/' | ':' | '=' | '-' | '<' | '>')
        });
    if safe {
        arg.to_owned()
    } else {
        // POSIX single-quote escaping: close, escaped quote, reopen.
        format!("'{}'", arg.replace('\'', "'\\''"))
    }
}

/// Truthful semantic contribution metadata for one response.
///
/// This contract is reusable by product output schemas after query
/// execution. The planner initializes `admitted_semantic_scores` to zero;
/// a search path must replace it with the actual admitted count before
/// emitting a completed semantic response. Non-semantic realized topologies
/// are permanently constrained to zero. The trusted type is Serialize-only:
/// wire data inside an [`UntrustedRecoveryPlan`] is decoded into a private raw
/// shape and compared field-for-field with a freshly planned contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SemanticResponseContract {
    requested_topology: RetrievalTopology,
    realized_topology: RetrievalTopology,
    coverage_ppm: u32,
    admitted_semantic_scores: u64,
    /// Present exactly for a lexical-only hybrid degradation.
    degradation_reason_code: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct SemanticResponseContractWire {
    requested_topology: RetrievalTopology,
    realized_topology: RetrievalTopology,
    coverage_ppm: u32,
    admitted_semantic_scores: u64,
    #[serde(default)]
    degradation_reason_code: WireField<Option<String>>,
}

impl SemanticResponseContract {
    /// Construct and validate one response contract.
    ///
    /// # Errors
    ///
    /// Rejects impossible requested/realized topology pairs, out-of-range
    /// coverage, semantic-score admission by lexical/hash lanes, and
    /// missing or spurious degradation reasons.
    pub fn new(
        requested_topology: RetrievalTopology,
        realized_topology: RetrievalTopology,
        coverage_ppm: u32,
        admitted_semantic_scores: u64,
        degradation_reason_code: Option<String>,
    ) -> Result<Self, RecoveryContractError> {
        let contract = Self {
            requested_topology,
            realized_topology,
            coverage_ppm,
            admitted_semantic_scores,
            degradation_reason_code,
        };
        contract.validate()?;
        Ok(contract)
    }

    /// Retrieval topology the caller requested.
    #[must_use]
    pub const fn requested_topology(&self) -> RetrievalTopology {
        self.requested_topology
    }

    /// Retrieval topology the response actually realizes.
    #[must_use]
    pub const fn realized_topology(&self) -> RetrievalTopology {
        self.realized_topology
    }

    /// Semantic document coverage in parts per million.
    #[must_use]
    pub const fn coverage_ppm(&self) -> u32 {
        self.coverage_ppm
    }

    /// Number of semantic scores admitted into the response.
    #[must_use]
    pub const fn admitted_semantic_scores(&self) -> u64 {
        self.admitted_semantic_scores
    }

    /// Typed degradation reason for lexical-only hybrid fallback.
    #[must_use]
    pub fn degradation_reason_code(&self) -> Option<&str> {
        self.degradation_reason_code.as_deref()
    }

    /// Replace the planning-boundary zero with the count admitted by a
    /// completed response, re-validating non-semantic invariants.
    ///
    /// # Errors
    ///
    /// Returns [`RecoveryContractError::NonSemanticScoresAdmitted`] if a
    /// lexical/hash response attempts to claim semantic contribution.
    pub fn with_admitted_semantic_scores(
        mut self,
        admitted_semantic_scores: u64,
    ) -> Result<Self, RecoveryContractError> {
        self.admitted_semantic_scores = admitted_semantic_scores;
        self.validate()?;
        Ok(self)
    }

    fn validate(&self) -> Result<(), RecoveryContractError> {
        let topology_compatible =
            retrieval_topology_fits_request(self.requested_topology, self.realized_topology);
        if !topology_compatible {
            return Err(RecoveryContractError::IncompatibleResponseTopology {
                requested: self.requested_topology,
                realized: self.realized_topology,
            });
        }

        let coverage_valid = match self.realized_topology {
            RetrievalTopology::LexicalOnly | RetrievalTopology::HashControl => {
                self.coverage_ppm == 0
            }
            RetrievalTopology::FastOnly
            | RetrievalTopology::QualityOnly
            | RetrievalTopology::FullProgressive => self.coverage_ppm == COMPLETE_COVERAGE_PPM,
            RetrievalTopology::PartialQuality { coverage_ppm } => {
                coverage_ppm == self.coverage_ppm
                    && (1..COMPLETE_COVERAGE_PPM).contains(&coverage_ppm)
            }
        };
        if !coverage_valid {
            return Err(RecoveryContractError::InvalidCoverage {
                topology: self.realized_topology,
                coverage_ppm: self.coverage_ppm,
            });
        }

        if !self.realized_topology.is_semantic() && self.admitted_semantic_scores != 0 {
            return Err(RecoveryContractError::NonSemanticScoresAdmitted {
                topology: self.realized_topology,
                admitted: self.admitted_semantic_scores,
            });
        }

        match (
            self.requested_topology,
            self.realized_topology,
            self.degradation_reason_code.as_deref(),
        ) {
            (RetrievalTopology::LexicalOnly, RetrievalTopology::LexicalOnly, None) => Ok(()),
            (RetrievalTopology::LexicalOnly, RetrievalTopology::LexicalOnly, Some(_)) => {
                Err(RecoveryContractError::UnexpectedDegradationReason)
            }
            (_, RetrievalTopology::LexicalOnly, Some(code)) if !code.trim().is_empty() => Ok(()),
            (_, RetrievalTopology::LexicalOnly, _) => {
                Err(RecoveryContractError::MissingDegradationReason)
            }
            (_, _, None) => Ok(()),
            (_, _, Some(_)) => Err(RecoveryContractError::UnexpectedDegradationReason),
        }
    }
}

/// The full plan: current state, verdict for the requested mode, and the
/// truthful next action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RecoveryPlan {
    /// [`RECOVERY_PLAN_SCHEMA_VERSION`].
    schema_version: RecoveryPlanSchemaVersion,
    /// The readiness state the plan was computed from.
    state: SemanticReadiness,
    /// Stable code for `state` (denormalized for consumers that do not
    /// decode the enum).
    state_code: String,
    /// Producer trust classification derived from `state`.
    provenance: SemanticProvenance,
    /// The mode the caller requested.
    mode: RequestMode,
    /// Exact retrieval topology the caller requested.
    requested_topology: RetrievalTopology,
    /// The policy the plan was computed under.
    policy: RecoveryPolicy,
    /// Whether semantic results can be served right now.
    semantic_available: bool,
    /// Whether retrying can succeed, and under what condition.
    retryability: Retryability,
    /// The truthful next action; `None` when the state needs none (ready,
    /// or the emptiness was request-scoped).
    action: Option<RecoveryAction>,
    /// Response shape allowed by this decision. `None` means the explicit
    /// request fails closed and no response may be emitted.
    response_contract: Option<SemanticResponseContract>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecoveryPlanWire {
    schema_version: RecoveryPlanSchemaVersion,
    state: SemanticReadiness,
    state_code: String,
    provenance: SemanticProvenance,
    mode: RequestMode,
    requested_topology: RetrievalTopology,
    policy: RecoveryPolicy,
    semantic_available: bool,
    retryability: Retryability,
    #[serde(default)]
    action: WireField<Option<RecoveryActionWire>>,
    #[serde(default)]
    response_contract: WireField<Option<SemanticResponseContractWire>>,
}

impl RecoveryPlanWire {
    fn validate_required_option_presence(&self) -> Result<(), &'static str> {
        let action = self
            .action
            .required_option_ref()
            .map_err(|()| "missing required field action")?;
        let response = self
            .response_contract
            .required_option_ref()
            .map_err(|()| "missing required field response_contract")?;
        if let Some(action) = action {
            action
                .required_authorization
                .required_option_ref()
                .map_err(|()| "missing required field action.required_authorization")?;
        }
        if let Some(response) = response {
            response
                .degradation_reason_code
                .required_option_ref()
                .map_err(|()| "missing required field response_contract.degradation_reason_code")?;
        }
        Ok(())
    }
}

/// Opaque, non-executable recovery-plan payload decoded from an untrusted
/// transport.
///
/// No action or argv accessor exists on this type. The only promotion path is
/// [`Self::validate_against`], which compares every payload field with a new
/// plan derived exclusively from a [`TrustedRecoveryContext`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UntrustedRecoveryPlan {
    wire: RecoveryPlanWire,
}

impl<'de> Deserialize<'de> for UntrustedRecoveryPlan {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = RecoveryPlanWire::deserialize(deserializer)?;
        wire.validate_required_option_presence()
            .map_err(serde::de::Error::custom)?;
        Ok(Self { wire })
    }
}

/// Independently sourced inputs that may promote one untrusted payload to a
/// trusted, executable [`RecoveryPlan`].
///
/// Every value must come from the current readiness probe, caller request,
/// environment policy, or frozen manifest. Never populate this context from
/// the payload being validated.
#[derive(Debug, Clone, Copy)]
pub struct TrustedRecoveryContext<'a> {
    state: &'a SemanticReadiness,
    request: RecoveryRequest,
    policy: &'a RecoveryPolicy,
    acquisition_target: Option<&'a ModelAcquisitionTarget>,
    evaluation_time_unix_seconds: u64,
}

impl<'a> TrustedRecoveryContext<'a> {
    /// Bind authoritative runtime inputs for untrusted-plan validation.
    #[must_use]
    pub const fn new(
        state: &'a SemanticReadiness,
        request: RecoveryRequest,
        policy: &'a RecoveryPolicy,
        acquisition_target: Option<&'a ModelAcquisitionTarget>,
        evaluation_time_unix_seconds: u64,
    ) -> Self {
        Self {
            state,
            request,
            policy,
            acquisition_target,
            evaluation_time_unix_seconds,
        }
    }
}

impl UntrustedRecoveryPlan {
    /// Compare every payload field with a fresh plan derived exclusively from
    /// trusted inputs.
    ///
    /// # Errors
    ///
    /// Returns [`RecoveryContractError::InconsistentRecoveryPlan`] for any
    /// payload substitution, including a coherent multi-field forgery.
    /// Trusted-context planning and target-validation errors are preserved.
    pub fn validate_against(
        self,
        trusted: TrustedRecoveryContext<'_>,
    ) -> Result<RecoveryPlan, RecoveryContractError> {
        let canonical = plan(trusted)?;
        validate_wire_against_canonical(&self.wire, &canonical)?;
        Ok(canonical)
    }
}

impl RecoveryPlan {
    /// Wire schema version.
    #[must_use]
    pub const fn schema_version(&self) -> RecoveryPlanSchemaVersion {
        self.schema_version
    }

    /// Readiness state used by the planner.
    #[must_use]
    pub const fn state(&self) -> &SemanticReadiness {
        &self.state
    }

    /// Stable code derived from [`Self::state`].
    #[must_use]
    pub fn state_code(&self) -> &str {
        &self.state_code
    }

    /// Producer provenance derived from readiness.
    #[must_use]
    pub const fn provenance(&self) -> SemanticProvenance {
        self.provenance
    }

    /// Requested mode.
    #[must_use]
    pub const fn mode(&self) -> RequestMode {
        self.mode
    }

    /// Requested retrieval topology.
    #[must_use]
    pub const fn requested_topology(&self) -> RetrievalTopology {
        self.requested_topology
    }

    /// Trusted policy used to construct the plan.
    #[must_use]
    pub const fn policy(&self) -> &RecoveryPolicy {
        &self.policy
    }

    /// Whether semantic results are currently admissible.
    #[must_use]
    pub const fn semantic_available(&self) -> bool {
        self.semantic_available
    }

    /// Retry verdict derived by the planner.
    #[must_use]
    pub const fn retryability(&self) -> Retryability {
        self.retryability
    }

    /// Canonical next action, when one exists.
    #[must_use]
    pub const fn action(&self) -> Option<&RecoveryAction> {
        self.action.as_ref()
    }

    /// Canonical response-admission contract, when a response is allowed.
    #[must_use]
    pub const fn response_contract(&self) -> Option<&SemanticResponseContract> {
        self.response_contract.as_ref()
    }

    /// Revalidate acquisition authorization immediately before execution.
    ///
    /// Planning and untrusted-wire promotion validate against the trusted time
    /// supplied for those operations. They do not mint a timeless capability:
    /// an executor must call this method again at the acquisition boundary
    /// using independently trusted current time. An interactive plan created
    /// before consent cannot execute: the caller must re-plan with the freshly
    /// granted exact authorization in [`RecoveryPolicy::acquisition_authorization`].
    ///
    /// # Errors
    ///
    /// Returns an authorization structural, scope-binding, not-yet-valid, or
    /// expiry error when the action is no longer executable at
    /// `evaluation_time_unix_seconds`.
    pub fn validate_for_execution_at(
        &self,
        evaluation_time_unix_seconds: u64,
    ) -> Result<(), RecoveryContractError> {
        validate_execution_authorization_binding(
            self.action
                .as_ref()
                .and_then(RecoveryAction::required_authorization),
            self.policy.acquisition_authorization.as_ref(),
            evaluation_time_unix_seconds,
        )
    }
}

fn inconsistent(field: &'static str) -> RecoveryContractError {
    RecoveryContractError::InconsistentRecoveryPlan { field }
}

fn validate_wire_against_canonical(
    wire: &RecoveryPlanWire,
    canonical: &RecoveryPlan,
) -> Result<(), RecoveryContractError> {
    if wire.schema_version != canonical.schema_version {
        return Err(inconsistent("schema_version"));
    }
    if wire.state != canonical.state {
        return Err(inconsistent("state"));
    }
    if wire.state_code != canonical.state_code {
        return Err(inconsistent("state_code"));
    }
    if wire.provenance != canonical.provenance {
        return Err(inconsistent("provenance"));
    }
    if wire.mode != canonical.mode {
        return Err(inconsistent("mode"));
    }
    if wire.requested_topology != canonical.requested_topology {
        return Err(inconsistent("requested_topology"));
    }
    if wire.policy != canonical.policy {
        return Err(inconsistent("policy"));
    }
    if wire.semantic_available != canonical.semantic_available {
        return Err(inconsistent("semantic_available"));
    }
    if wire.retryability != canonical.retryability {
        return Err(inconsistent("retryability"));
    }
    let wire_action = wire
        .action
        .required_option_ref()
        .map_err(|()| inconsistent("action"))?;
    validate_action_wire(wire_action, canonical.action.as_ref())?;
    let wire_response = wire
        .response_contract
        .required_option_ref()
        .map_err(|()| inconsistent("response_contract"))?;
    validate_response_wire(wire_response, canonical.response_contract.as_ref())?;
    Ok(())
}

fn validate_action_wire(
    wire: Option<&RecoveryActionWire>,
    canonical: Option<&RecoveryAction>,
) -> Result<(), RecoveryContractError> {
    let (Some(wire), Some(canonical)) = (wire, canonical) else {
        return if wire.is_none() && canonical.is_none() {
            Ok(())
        } else {
            Err(inconsistent("action"))
        };
    };
    if wire.code != canonical.code {
        return Err(inconsistent("action.code"));
    }
    if wire.explanation != canonical.explanation {
        return Err(inconsistent("action.explanation"));
    }
    if wire.argv != canonical.argv {
        return Err(inconsistent("action.argv"));
    }
    if wire.network_required != canonical.network_required {
        return Err(inconsistent("action.network_required"));
    }
    if wire.consent_required != canonical.consent_required {
        return Err(inconsistent("action.consent_required"));
    }
    if wire.preserves_old_data != canonical.preserves_old_data {
        return Err(inconsistent("action.preserves_old_data"));
    }
    if wire.potentially_destructive != canonical.potentially_destructive {
        return Err(inconsistent("action.potentially_destructive"));
    }
    if wire.prerequisites != canonical.prerequisites {
        return Err(inconsistent("action.prerequisites"));
    }
    if wire.expected_postcondition != canonical.expected_postcondition {
        return Err(inconsistent("action.expected_postcondition"));
    }
    let wire_authorization = wire
        .required_authorization
        .required_option_ref()
        .map_err(|()| inconsistent("action.required_authorization"))?;
    if wire_authorization != canonical.required_authorization.as_ref() {
        return Err(inconsistent("action.required_authorization"));
    }
    Ok(())
}

fn validate_response_wire(
    wire: Option<&SemanticResponseContractWire>,
    canonical: Option<&SemanticResponseContract>,
) -> Result<(), RecoveryContractError> {
    let (Some(wire), Some(canonical)) = (wire, canonical) else {
        return if wire.is_none() && canonical.is_none() {
            Ok(())
        } else {
            Err(inconsistent("response_contract"))
        };
    };
    if wire.requested_topology != canonical.requested_topology {
        return Err(inconsistent("response_contract.requested_topology"));
    }
    if wire.realized_topology != canonical.realized_topology {
        return Err(inconsistent("response_contract.realized_topology"));
    }
    if wire.coverage_ppm != canonical.coverage_ppm {
        return Err(inconsistent("response_contract.coverage_ppm"));
    }
    if wire.admitted_semantic_scores != canonical.admitted_semantic_scores {
        return Err(inconsistent("response_contract.admitted_semantic_scores"));
    }
    let wire_degradation_reason = wire
        .degradation_reason_code
        .required_option_ref()
        .map_err(|()| inconsistent("response_contract.degradation_reason_code"))?;
    if wire_degradation_reason != canonical.degradation_reason_code.as_ref() {
        return Err(inconsistent("response_contract.degradation_reason_code"));
    }
    Ok(())
}

fn mismatched_authorization_field(
    required: &ModelAcquisitionAuthorization,
    supplied: &ModelAcquisitionAuthorization,
) -> Option<&'static str> {
    if required.model_id != supplied.model_id {
        Some("model_id")
    } else if required.model_tier != supplied.model_tier {
        Some("model_tier")
    } else if required.upstream_revision != supplied.upstream_revision {
        Some("upstream_revision")
    } else if required.embedding_space != supplied.embedding_space {
        Some("embedding_space")
    } else if required.manifest_fingerprint != supplied.manifest_fingerprint {
        Some("manifest_fingerprint")
    } else if required.license_spdx != supplied.license_spdx {
        Some("license_spdx")
    } else if required.source != supplied.source {
        Some("source")
    } else if required.byte_budget != supplied.byte_budget {
        Some("byte_budget")
    } else if required.destination_class != supplied.destination_class {
        Some("destination_class")
    } else if required.destination_fingerprint != supplied.destination_fingerprint {
        Some("destination_fingerprint")
    } else if required.document_count != supplied.document_count {
        Some("document_count")
    } else if required.estimated_reindex_duration_ms != supplied.estimated_reindex_duration_ms {
        Some("estimated_reindex_duration_ms")
    } else if required.issued_at_unix_seconds != supplied.issued_at_unix_seconds {
        Some("issued_at_unix_seconds")
    } else if required.expires_at_unix_seconds != supplied.expires_at_unix_seconds {
        Some("expires_at_unix_seconds")
    } else if required.nonce != supplied.nonce {
        Some("nonce")
    } else {
        None
    }
}

fn validate_authorization_binding(
    required: Option<&ModelAcquisitionAuthorization>,
    supplied: Option<&ModelAcquisitionAuthorization>,
    evaluation_time_unix_seconds: u64,
) -> Result<(), RecoveryContractError> {
    match (required, supplied) {
        (None, None) => Ok(()),
        (None, Some(_)) => Err(RecoveryContractError::SurplusAcquisitionAuthorization),
        (Some(required), None) => required.validate_at(evaluation_time_unix_seconds),
        (Some(required), Some(supplied)) => {
            required.validate_at(evaluation_time_unix_seconds)?;
            supplied.validate_at(evaluation_time_unix_seconds)?;
            if let Some(field) = mismatched_authorization_field(required, supplied) {
                return Err(RecoveryContractError::MismatchedAcquisitionAuthorization { field });
            }
            Ok(())
        }
    }
}

fn validate_execution_authorization_binding(
    required: Option<&ModelAcquisitionAuthorization>,
    supplied: Option<&ModelAcquisitionAuthorization>,
    evaluation_time_unix_seconds: u64,
) -> Result<(), RecoveryContractError> {
    match (required, supplied) {
        (None, None) => Ok(()),
        (None, Some(_)) => Err(RecoveryContractError::SurplusAcquisitionAuthorization),
        (Some(_), None) => Err(RecoveryContractError::MissingAcquisitionAuthorization),
        (Some(required), Some(supplied)) => {
            required.validate_at(evaluation_time_unix_seconds)?;
            supplied.validate_at(evaluation_time_unix_seconds)?;
            if let Some(field) = mismatched_authorization_field(required, supplied) {
                return Err(RecoveryContractError::MismatchedAcquisitionAuthorization { field });
            }
            Ok(())
        }
    }
}

/// Compute the truthful plan for a readiness state under a request and
/// policy. Pure and deterministic: identical inputs yield identical plans.
///
/// # Errors
///
/// Rejects ambiguous request topology, invalid partial coverage, hash
/// requests without hash-control readiness, and malformed acquisition
/// targets before returning executable recovery metadata. Trusted evaluation
/// time is mandatory and never read from the serialized plan or ambient
/// process state.
pub fn plan(trusted: TrustedRecoveryContext<'_>) -> Result<RecoveryPlan, RecoveryContractError> {
    let TrustedRecoveryContext {
        state,
        request,
        policy,
        acquisition_target,
        evaluation_time_unix_seconds,
    } = trusted;
    let state = state.clone();
    let policy = policy.clone();
    let request = request.validate()?;
    if let Some(authorization) = &policy.acquisition_authorization {
        authorization.validate()?;
    }
    validate_hash_mode_state(request.mode, &state)?;
    validate_readiness(&state)?;
    validate_acquisition_tier(&state, request.requested_topology, acquisition_target)?;

    let action = action_for(
        &state,
        request.requested_topology,
        policy.network,
        acquisition_target,
    )?;
    validate_authorization_binding(
        action
            .as_ref()
            .and_then(RecoveryAction::required_authorization),
        policy.acquisition_authorization.as_ref(),
        evaluation_time_unix_seconds,
    )?;
    let (action, retryability) = match action {
        None => (
            None,
            if state.semantic_available()
                || matches!(
                    (&state, request.mode),
                    (SemanticReadiness::HashControl, RequestMode::HashControl)
                )
            {
                Retryability::NotNeeded
            } else {
                Retryability::AfterRequestChange
            },
        ),
        Some(mut action) => {
            let network_blocked =
                action.network_required && matches!(policy.network, NetworkPolicy::Offline);
            if network_blocked {
                push_prerequisite(&mut action, "recovery.policy.allow_network");
            }
            let authorization_satisfied =
                action
                    .required_authorization
                    .as_ref()
                    .is_some_and(|required| {
                        policy.acquisition_authorization.as_ref() == Some(required)
                    });
            let binding_missing = matches!(
                action.code.as_str(),
                "recovery.action.acquire_model" | "recovery.action.reacquire_model"
            ) && action.required_authorization.is_none();
            if binding_missing {
                push_prerequisite(&mut action, "recovery.policy.bind_model");
            }
            let consent_blocked = action.consent_required
                && matches!(policy.interaction, InteractionPolicy::NonInteractive)
                && !authorization_satisfied;
            if consent_blocked {
                push_prerequisite(&mut action, "recovery.policy.grant_consent");
            }
            let capability_blocked = action
                .prerequisites
                .iter()
                .any(|code| code.starts_with("recovery.capability."));
            let retryability = if capability_blocked {
                Retryability::BlockedByCapability
            } else if network_blocked
                || binding_missing
                || consent_blocked
                || !action.prerequisites.is_empty()
            {
                Retryability::BlockedByPolicy
            } else {
                Retryability::AfterAction
            };
            (Some(action), retryability)
        }
    };

    let semantic_available = state.semantic_available();
    let response_contract = response_contract_for(&state, request)?;
    let state_code = state.state_code().to_owned();
    let provenance = state.provenance();

    Ok(RecoveryPlan {
        schema_version: RecoveryPlanSchemaVersion::V4,
        state,
        state_code,
        provenance,
        mode: request.mode,
        requested_topology: request.requested_topology,
        policy,
        semantic_available,
        retryability,
        action,
        response_contract,
    })
}

fn validate_hash_mode_state(
    mode: RequestMode,
    state: &SemanticReadiness,
) -> Result<(), RecoveryContractError> {
    let hash_request = matches!(mode, RequestMode::HashControl);
    let hash_state = matches!(state, SemanticReadiness::HashControl);
    if hash_request == hash_state {
        Ok(())
    } else {
        Err(RecoveryContractError::HashControlModeReadinessMismatch)
    }
}

fn validate_readiness(state: &SemanticReadiness) -> Result<(), RecoveryContractError> {
    if let SemanticReadiness::PartialQualityCoverage { coverage_ppm, .. } = state
        && !(1..COMPLETE_COVERAGE_PPM).contains(coverage_ppm)
    {
        return Err(RecoveryContractError::InvalidCoverage {
            topology: RetrievalTopology::PartialQuality {
                coverage_ppm: *coverage_ppm,
            },
            coverage_ppm: *coverage_ppm,
        });
    }
    Ok(())
}

fn validate_acquisition_tier(
    state: &SemanticReadiness,
    requested_topology: RetrievalTopology,
    acquisition_target: Option<&ModelAcquisitionTarget>,
) -> Result<(), RecoveryContractError> {
    let readiness_tier = match state {
        SemanticReadiness::ModelMissing { tier } | SemanticReadiness::ModelUnloadable { tier } => {
            *tier
        }
        _ => return Ok(()),
    };
    let topology_matches = match requested_topology {
        RetrievalTopology::FastOnly => readiness_tier == ModelTier::Fast,
        RetrievalTopology::QualityOnly => readiness_tier == ModelTier::Quality,
        RetrievalTopology::FullProgressive => true,
        RetrievalTopology::LexicalOnly
        | RetrievalTopology::PartialQuality { .. }
        | RetrievalTopology::HashControl => false,
    };
    if !topology_matches {
        return Err(RecoveryContractError::UnavailableTierTopologyMismatch {
            tier: readiness_tier,
            requested_topology,
        });
    }
    if let Some(target) = acquisition_target
        && target.model_tier != readiness_tier
    {
        return Err(RecoveryContractError::AcquisitionTargetTierMismatch {
            readiness_tier,
            target_tier: target.model_tier,
        });
    }
    Ok(())
}

fn response_contract_for(
    state: &SemanticReadiness,
    request: RecoveryRequest,
) -> Result<Option<SemanticResponseContract>, RecoveryContractError> {
    if matches!(request.mode, RequestMode::HashControl) {
        return SemanticResponseContract::new(
            RetrievalTopology::HashControl,
            RetrievalTopology::HashControl,
            0,
            0,
            None,
        )
        .map(Some);
    }

    if state.semantic_available() {
        let (realized_topology, coverage_ppm) = match (state, request.requested_topology) {
            (
                SemanticReadiness::PartialQualityCoverage { coverage_ppm, .. },
                RetrievalTopology::QualityOnly | RetrievalTopology::FullProgressive,
            ) => (
                RetrievalTopology::PartialQuality {
                    coverage_ppm: *coverage_ppm,
                },
                *coverage_ppm,
            ),
            _ => (request.requested_topology, COMPLETE_COVERAGE_PPM),
        };
        return SemanticResponseContract::new(
            request.requested_topology,
            realized_topology,
            coverage_ppm,
            0,
            None,
        )
        .map(Some);
    }

    if matches!(request.mode, RequestMode::Hybrid) {
        return SemanticResponseContract::new(
            request.requested_topology,
            RetrievalTopology::LexicalOnly,
            0,
            0,
            Some(state.state_code().to_owned()),
        )
        .map(Some);
    }

    Ok(None)
}

fn push_prerequisite(action: &mut RecoveryAction, code: &str) {
    if !action.prerequisites.iter().any(|existing| existing == code) {
        action.prerequisites.push(code.to_owned());
    }
}

// The parameters mirror RecoveryAction's schema-mandated boolean fields
// one-to-one; an intermediate flags type would only restate the struct.
#[allow(clippy::fn_params_excessive_bools)]
fn simple_action(
    code: &str,
    explanation: &str,
    argv: &[&str],
    network_required: bool,
    consent_required: bool,
    preserves_old_data: bool,
    potentially_destructive: bool,
    expected_postcondition: &str,
) -> RecoveryAction {
    RecoveryAction {
        code: code.to_owned(),
        explanation: explanation.to_owned(),
        argv: argv.iter().map(|&a| a.to_owned()).collect(),
        network_required,
        consent_required,
        preserves_old_data,
        potentially_destructive,
        prerequisites: Vec::new(),
        expected_postcondition: expected_postcondition.to_owned(),
        required_authorization: None,
    }
}

fn block_unbound_semantic_index_action(
    mut action: RecoveryAction,
    requested_topology: RetrievalTopology,
    quality_capability: &str,
) -> RecoveryAction {
    action.argv.clear();
    let capability = if matches!(requested_topology, RetrievalTopology::FastOnly) {
        "recovery.capability.execute_bound_semantic_index"
    } else {
        quality_capability
    };
    push_prerequisite(&mut action, capability);
    action
}

/// The exhaustive state → action table. No wildcard arm: adding a
/// readiness state without planning for it is a compile error.
fn action_for(
    state: &SemanticReadiness,
    requested_topology: RetrievalTopology,
    network: NetworkPolicy,
    acquisition_target: Option<&ModelAcquisitionTarget>,
) -> Result<Option<RecoveryAction>, RecoveryContractError> {
    match state {
        SemanticReadiness::Ready { .. } | SemanticReadiness::HashControl => Ok(None),
        SemanticReadiness::ModelMissing { .. } => {
            model_acquisition_action(false, network, acquisition_target).map(Some)
        }
        SemanticReadiness::ModelUnloadable { .. } => {
            model_acquisition_action(true, network, acquisition_target).map(Some)
        }
        SemanticReadiness::IndexAbsent => Ok(Some(block_unbound_semantic_index_action(
            simple_action(
                "recovery.action.build_index",
                "A verified model is present but no vector index exists. Current generic fsfs \
                 indexing can still fall through to a non-semantic producer, so recovery needs \
                 an executor bound to the attested semantic space and requested topology.",
                &[],
                false,
                false,
                true,
                false,
                "recovery.post.index_built",
            ),
            requested_topology,
            "recovery.capability.build_quality_tier",
        ))),
        SemanticReadiness::IdentityMismatch => Ok(Some(block_unbound_semantic_index_action(
            simple_action(
                "recovery.action.reindex_full",
                "The index was built in a different embedding space than the configured model. \
                 Rebuild only if the identity change is intentional, through an executor bound \
                 to the attested semantic space and requested topology; the existing index is \
                 replaced.",
                &[],
                false,
                true,
                false,
                true,
                "recovery.post.index_rebuilt",
            ),
            requested_topology,
            "recovery.capability.reindex_quality_tier",
        ))),
        SemanticReadiness::DaemonMismatch => {
            let mut action = simple_action(
                "recovery.action.restart_daemon",
                "The embedding daemon serves a different space than the local configuration, \
                 but fsfs currently has no parser-executable restart operation. A bound daemon \
                 lifecycle capability must land before recovery can run.",
                &[],
                false,
                false,
                true,
                false,
                "recovery.post.daemon_aligned",
            );
            push_prerequisite(&mut action, "recovery.capability.restart_daemon");
            Ok(Some(action))
        }
        SemanticReadiness::IndexEmpty(reason) => Ok(plan_for_empty(*reason, requested_topology)),
        SemanticReadiness::ManifestUnsafe => Ok(Some(block_unbound_semantic_index_action(
            simple_action(
                "recovery.action.reindex_full",
                "The manifest failed safety validation and its artifacts must not be trusted; \
                 rebuild from source content only through an executor bound to the attested \
                 semantic space and requested topology.",
                &[],
                false,
                true,
                false,
                true,
                "recovery.post.index_rebuilt",
            ),
            requested_topology,
            "recovery.capability.reindex_quality_tier",
        ))),
        SemanticReadiness::AnnStale => {
            let mut action = simple_action(
                "recovery.action.rebuild_ann",
                "The ANN sidecar belongs to an older index generation, but generic indexing \
                 does not bind the exact ANN generation to rebuild. Exact search remains \
                 correct while the dedicated generation-aware capability is unavailable.",
                &[],
                false,
                false,
                true,
                false,
                "recovery.post.ann_rebuilt",
            );
            push_prerequisite(&mut action, "recovery.capability.rebuild_ann_generation");
            Ok(Some(action))
        }
        SemanticReadiness::GenerationIncomplete => Ok(Some(block_unbound_semantic_index_action(
            simple_action(
                "recovery.action.resume_index",
                "An index generation was interrupted before publication. Completing it requires \
                 an executor bound to the attested semantic space, topology, and generation; \
                 published data remains untouched.",
                &[],
                false,
                false,
                true,
                false,
                "recovery.post.generation_completed",
            ),
            requested_topology,
            "recovery.capability.resume_quality_generation",
        ))),
        SemanticReadiness::PartialQualityCoverage { .. }
            if matches!(requested_topology, RetrievalTopology::FastOnly) =>
        {
            Ok(None)
        }
        SemanticReadiness::PartialQualityCoverage { .. } => {
            let mut action = simple_action(
                "recovery.action.backfill_quality",
                "Some records lack quality-tier embeddings, but generic indexing does not \
                 express a quality-only backfill. Search remains available while the dedicated \
                 tier-aware capability is unavailable.",
                &[],
                false,
                false,
                true,
                false,
                "recovery.post.coverage_completed",
            );
            push_prerequisite(&mut action, "recovery.capability.backfill_quality_tier");
            Ok(Some(action))
        }
        SemanticReadiness::RemoteUnverified => {
            let mut action = simple_action(
                "recovery.action.provide_attestation",
                "Explicit remote intent cannot be admitted because its producer space is not \
                 attested. Supply a pinned producer attester through the caller-owned \
                 configuration boundary; there is no safe generic command that can invent \
                 this trust root.",
                &[],
                false,
                false,
                true,
                false,
                "recovery.post.remote_attested",
            );
            push_prerequisite(&mut action, "recovery.policy.provide_attestation");
            Ok(Some(action))
        }
    }
}

fn model_acquisition_action(
    reacquire: bool,
    network: NetworkPolicy,
    acquisition_target: Option<&ModelAcquisitionTarget>,
) -> Result<RecoveryAction, RecoveryContractError> {
    let code = if reacquire {
        "recovery.action.reacquire_model"
    } else {
        "recovery.action.acquire_model"
    };
    let required_authorization = acquisition_target
        .map(|target| target.authorization_for(network))
        .transpose()?;
    let explanation = match (network, reacquire) {
        (NetworkPolicy::Allowed, false) => {
            "The configured semantic model must be acquired under the exact frozen \
             authorization, but current fsfs download syntax cannot bind every authorized \
             identity, byte-budget, destination, corpus, and reindex field to execution. This \
             action is deliberately non-executable until a bound executor exists."
        }
        (NetworkPolicy::Allowed, true) => {
            "The cached model failed verification or load self-test, but current fsfs download \
             syntax cannot bind the complete frozen re-acquisition authorization to execution. \
             This action is deliberately non-executable until a bound executor exists; index \
             data is untouched."
        }
        (NetworkPolicy::Offline, false) => {
            "A complete local-bundle import is required, but the current fsfs parser has no \
             offline importer. This action is deliberately non-executable until that capability \
             exists; no network access is permitted."
        }
        (NetworkPolicy::Offline, true) => {
            "Replacing the unloadable cache from a complete local bundle requires an importer \
             that the current fsfs parser does not provide. This action is deliberately \
             non-executable until that capability exists; index data is untouched."
        }
    };
    let mut action = simple_action(
        code,
        explanation,
        &[],
        matches!(network, NetworkPolicy::Allowed),
        true,
        !reacquire,
        reacquire,
        "recovery.post.model_acquired_unverified",
    );
    action.required_authorization = required_authorization;
    match network {
        NetworkPolicy::Allowed => push_prerequisite(
            &mut action,
            "recovery.capability.execute_bound_model_acquisition",
        ),
        NetworkPolicy::Offline => {
            push_prerequisite(&mut action, "recovery.capability.import_model_bundle");
        }
    }
    Ok(action)
}

/// Empty-index planning follows the zero-signal classification: benign
/// state emptiness wants ingestion, availability failures want rebuilds,
/// the ANN anomaly wants a sidecar rebuild, and request-scoped reasons
/// need no system action at all.
fn plan_for_empty(
    reason: ZeroSignalReason,
    requested_topology: RetrievalTopology,
) -> Option<RecoveryAction> {
    match reason {
        ZeroSignalReason::NewlyCreatedEmpty
        | ZeroSignalReason::AllTombstoned
        | ZeroSignalReason::WalOnlyNoLiveRecords => Some(block_unbound_semantic_index_action(
            simple_action(
                "recovery.action.ingest_content",
                "The index holds no live records. Populate it only through an executor bound to \
                 the attested semantic space and requested topology.",
                &[],
                false,
                false,
                true,
                false,
                "recovery.post.index_populated",
            ),
            requested_topology,
            "recovery.capability.ingest_quality_tier",
        )),
        ZeroSignalReason::NoUsableVectors => Some(block_unbound_semantic_index_action(
            simple_action(
                "recovery.action.reindex_full",
                "Live records exist but none of their stored vectors is usable (zero-norm or \
                 corrupt); rebuild only through an executor bound to the attested semantic space \
                 and requested topology.",
                &[],
                false,
                true,
                false,
                true,
                "recovery.post.index_rebuilt",
            ),
            requested_topology,
            "recovery.capability.reindex_quality_tier",
        )),
        ZeroSignalReason::AnnReturnedEmptyDespiteUsableVectors => {
            let mut action = simple_action(
                "recovery.action.rebuild_ann",
                "The ANN graph returned no candidates although usable live vectors exist, but \
                 generic indexing does not bind the exact ANN generation to rebuild. The vector \
                 index remains untouched.",
                &[],
                false,
                false,
                true,
                false,
                "recovery.post.ann_rebuilt",
            );
            push_prerequisite(&mut action, "recovery.capability.rebuild_ann_generation");
            Some(action)
        }
        ZeroSignalReason::CallerRequestedZeroK
        | ZeroSignalReason::FilterEliminatedAll
        | ZeroSignalReason::NonFiniteQuery
        | ZeroSignalReason::ZeroNormQuery => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision_plane::ReasonCode;
    use crate::generation::{
        EMBEDDING_SPACE_IDENTITY_SCHEMA_V1, EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1,
    };

    const TEST_NOW_UNIX_SECONDS: u64 = 2_000_000_000;
    const TEST_AUTHORIZATION_ISSUED_AT_UNIX_SECONDS: u64 = TEST_NOW_UNIX_SECONDS - 60;
    const TEST_AUTHORIZATION_EXPIRES_AT_UNIX_SECONDS: u64 = TEST_NOW_UNIX_SECONDS + 600;
    const TEST_AUTHORIZATION_NONCE: &str = "0123456789abcdef0123456789abcdef";

    fn semantic_space() -> EmbeddingSpaceIdentityV1 {
        EmbeddingSpaceIdentityV1 {
            schema_version: EMBEDDING_SPACE_IDENTITY_SCHEMA_V1,
            logical_model_id: "fixture-semantic-model".to_owned(),
            immutable_revision: "revision-0123456789abcdef".to_owned(),
            kind: EmbeddingSpaceKindV1::Semantic,
            artifact_manifest_fingerprint: "a".repeat(64),
            artifacts: vec![EmbeddingArtifactIdentityV1 {
                role: "weights".to_owned(),
                sha256: "c".repeat(64),
                size: 42_000_000,
            }],
            tokenizer_fingerprint: "d".repeat(64),
            vocabulary_fingerprint: "e".repeat(64),
            model_config_fingerprint: "f".repeat(64),
            model_preprocessing: "nfc-v1".to_owned(),
            sequence_policy: "truncate-256-v1".to_owned(),
            query_instruction: String::new(),
            document_instruction: String::new(),
            pooling: "mean-v1".to_owned(),
            output_normalization: "l2-v1".to_owned(),
            dimension: 384,
            input_contract_fingerprint: "1".repeat(64),
            hash_control: None,
            projection: None,
        }
    }

    fn target_for(model_tier: ModelTier) -> ModelAcquisitionTarget {
        ModelAcquisitionTarget {
            model_id: "fixture-semantic-model".to_owned(),
            model_tier,
            embedding_space: semantic_space(),
            manifest_fingerprint: "a".repeat(64),
            upstream_revision: "revision-0123456789abcdef".to_owned(),
            license_spdx: "Apache-2.0".to_owned(),
            network_source_hosts: vec!["models.example.test".to_owned()],
            byte_budget: 42_000_000,
            destination_class: ModelDestinationClass::ManagedCache,
            destination_fingerprint: "b".repeat(64),
            document_count: 12_345,
            estimated_reindex_duration_ms: 98_765,
            issued_at_unix_seconds: TEST_AUTHORIZATION_ISSUED_AT_UNIX_SECONDS,
            expires_at_unix_seconds: TEST_AUTHORIZATION_EXPIRES_AT_UNIX_SECONDS,
            nonce: TEST_AUTHORIZATION_NONCE.to_owned(),
        }
    }

    fn target() -> ModelAcquisitionTarget {
        target_for(ModelTier::Quality)
    }

    const fn missing(tier: ModelTier) -> SemanticReadiness {
        SemanticReadiness::ModelMissing { tier }
    }

    const fn unloadable(tier: ModelTier) -> SemanticReadiness {
        SemanticReadiness::ModelUnloadable { tier }
    }

    const fn explicit(requested_topology: RetrievalTopology) -> RecoveryRequest {
        RecoveryRequest {
            mode: RequestMode::ExplicitSemantic,
            requested_topology,
        }
    }

    const fn hybrid(requested_topology: RetrievalTopology) -> RecoveryRequest {
        RecoveryRequest {
            mode: RequestMode::Hybrid,
            requested_topology,
        }
    }

    const fn hash_control() -> RecoveryRequest {
        RecoveryRequest {
            mode: RequestMode::HashControl,
            requested_topology: RetrievalTopology::HashControl,
        }
    }

    fn semantic_requests() -> [RecoveryRequest; 6] {
        [
            explicit(RetrievalTopology::FastOnly),
            explicit(RetrievalTopology::QualityOnly),
            explicit(RetrievalTopology::FullProgressive),
            hybrid(RetrievalTopology::FastOnly),
            hybrid(RetrievalTopology::QualityOnly),
            hybrid(RetrievalTopology::FullProgressive),
        ]
    }

    fn representative_states() -> Vec<SemanticReadiness> {
        vec![
            SemanticReadiness::Ready {
                provenance: VerifiedSemanticProvenance::Local,
            },
            SemanticReadiness::Ready {
                provenance: VerifiedSemanticProvenance::Remote,
            },
            SemanticReadiness::Ready {
                provenance: VerifiedSemanticProvenance::Daemon,
            },
            missing(ModelTier::Quality),
            unloadable(ModelTier::Quality),
            SemanticReadiness::IndexAbsent,
            SemanticReadiness::IdentityMismatch,
            SemanticReadiness::DaemonMismatch,
            SemanticReadiness::IndexEmpty(ZeroSignalReason::NewlyCreatedEmpty),
            SemanticReadiness::IndexEmpty(ZeroSignalReason::AllTombstoned),
            SemanticReadiness::IndexEmpty(ZeroSignalReason::WalOnlyNoLiveRecords),
            SemanticReadiness::IndexEmpty(ZeroSignalReason::CallerRequestedZeroK),
            SemanticReadiness::IndexEmpty(ZeroSignalReason::NoUsableVectors),
            SemanticReadiness::IndexEmpty(ZeroSignalReason::AnnReturnedEmptyDespiteUsableVectors),
            SemanticReadiness::ManifestUnsafe,
            SemanticReadiness::AnnStale,
            SemanticReadiness::GenerationIncomplete,
            SemanticReadiness::PartialQualityCoverage {
                provenance: VerifiedSemanticProvenance::Local,
                coverage_ppm: 750_000,
            },
            SemanticReadiness::PartialQualityCoverage {
                provenance: VerifiedSemanticProvenance::Remote,
                coverage_ppm: 500_000,
            },
            SemanticReadiness::RemoteUnverified,
            SemanticReadiness::HashControl,
        ]
    }

    fn unready_states() -> Vec<SemanticReadiness> {
        representative_states()
            .into_iter()
            .filter(|state| {
                !state.semantic_available() && !matches!(state, SemanticReadiness::HashControl)
            })
            .collect()
    }

    fn all_policies() -> Vec<RecoveryPolicy> {
        let mut out = Vec::new();
        for interaction in [
            InteractionPolicy::Interactive,
            InteractionPolicy::NonInteractive,
        ] {
            for network in [NetworkPolicy::Allowed, NetworkPolicy::Offline] {
                out.push(RecoveryPolicy {
                    interaction,
                    network,
                    acquisition_authorization: None,
                });
            }
        }
        out
    }

    fn permissive() -> RecoveryPolicy {
        RecoveryPolicy {
            interaction: InteractionPolicy::Interactive,
            network: NetworkPolicy::Allowed,
            acquisition_authorization: None,
        }
    }

    // These test builders deliberately own state and policy so call sites can
    // pass temporary fixtures without introducing local bindings solely for
    // borrow lifetimes.
    #[allow(clippy::needless_pass_by_value)]
    fn plan(
        state: SemanticReadiness,
        request: RecoveryRequest,
        policy: RecoveryPolicy,
        acquisition_target: Option<&ModelAcquisitionTarget>,
    ) -> Result<RecoveryPlan, RecoveryContractError> {
        super::plan(TrustedRecoveryContext::new(
            &state,
            request,
            &policy,
            acquisition_target,
            TEST_NOW_UNIX_SECONDS,
        ))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn planned(
        state: SemanticReadiness,
        request: RecoveryRequest,
        policy: RecoveryPolicy,
    ) -> RecoveryPlan {
        plan(state, request, policy, Some(&target())).expect("valid recovery plan")
    }

    fn tier_for_request(request: RecoveryRequest) -> ModelTier {
        match request.requested_topology {
            RetrievalTopology::FastOnly => ModelTier::Fast,
            RetrievalTopology::QualityOnly | RetrievalTopology::FullProgressive => {
                ModelTier::Quality
            }
            RetrievalTopology::LexicalOnly
            | RetrievalTopology::PartialQuality { .. }
            | RetrievalTopology::HashControl => {
                panic!("semantic recovery helper received a non-semantic topology")
            }
        }
    }

    fn state_for_request(state: &SemanticReadiness, request: RecoveryRequest) -> SemanticReadiness {
        match state {
            SemanticReadiness::ModelMissing { .. } => missing(tier_for_request(request)),
            SemanticReadiness::ModelUnloadable { .. } => unloadable(tier_for_request(request)),
            _ => state.clone(),
        }
    }

    fn decode_and_validate(
        value: serde_json::Value,
        state: &SemanticReadiness,
        request: RecoveryRequest,
        policy: &RecoveryPolicy,
        acquisition_target: Option<&ModelAcquisitionTarget>,
    ) -> Result<RecoveryPlan, String> {
        let untrusted: UntrustedRecoveryPlan =
            serde_json::from_value(value).map_err(|error| error.to_string())?;
        untrusted
            .validate_against(TrustedRecoveryContext::new(
                state,
                request,
                policy,
                acquisition_target,
                TEST_NOW_UNIX_SECONDS,
            ))
            .map_err(|error| error.to_string())
    }

    #[test]
    fn every_stable_code_is_valid_and_unique() {
        let mut codes = Vec::new();
        let states = representative_states();
        for state in &states {
            codes.push(state.state_code().to_owned());
            if !matches!(state, SemanticReadiness::HashControl) {
                for request in semantic_requests() {
                    for policy in all_policies() {
                        let tier = tier_for_request(request);
                        let state = state_for_request(state, request);
                        let plan = plan(state, request, policy, Some(&target_for(tier)))
                            .expect("representative plan");
                        if let Some(action) = plan.action {
                            codes.push(action.code.clone());
                            codes.push(action.expected_postcondition.clone());
                            codes.extend(action.prerequisites);
                        }
                    }
                }
            }
        }
        let hash_plan = planned(SemanticReadiness::HashControl, hash_control(), permissive());
        codes.push(hash_plan.state_code);
        let unbound = plan(
            missing(ModelTier::Fast),
            explicit(RetrievalTopology::FastOnly),
            permissive(),
            None,
        )
        .expect("unbound acquisition still returns a blocked plan");
        codes.extend(unbound.action.expect("model action").prerequisites);
        codes.push("recovery.policy.allow_network".to_owned());
        for code in &codes {
            assert!(
                ReasonCode::new(code.as_str()).is_valid(),
                "invalid stable code format: {code}"
            );
        }
        // Distinct states never share a code with distinct actions.
        let state_codes: std::collections::HashSet<_> =
            states.iter().map(|s| s.state_code()).collect();
        assert_eq!(state_codes.len(), 13, "one code per state variant");

        let emitted: std::collections::HashSet<_> = codes.iter().map(String::as_str).collect();
        let v1_codes = [
            "recovery.state.ready",
            "recovery.state.model_missing",
            "recovery.state.model_unloadable",
            "recovery.state.index_absent",
            "recovery.state.identity_mismatch",
            "recovery.state.daemon_mismatch",
            "recovery.state.index_empty",
            "recovery.state.manifest_unsafe",
            "recovery.state.ann_stale",
            "recovery.state.generation_incomplete",
            "recovery.state.partial_quality_coverage",
            "recovery.action.acquire_model",
            "recovery.action.reacquire_model",
            "recovery.action.build_index",
            "recovery.action.reindex_full",
            "recovery.action.restart_daemon",
            "recovery.action.ingest_content",
            "recovery.action.rebuild_ann",
            "recovery.action.resume_index",
            "recovery.action.backfill_quality",
            "recovery.post.model_acquired_unverified",
            "recovery.post.index_built",
            "recovery.post.index_rebuilt",
            "recovery.post.daemon_aligned",
            "recovery.post.index_populated",
            "recovery.post.ann_rebuilt",
            "recovery.post.generation_completed",
            "recovery.post.coverage_completed",
            "recovery.policy.allow_network",
            "recovery.policy.grant_consent",
        ];
        for old_code in v1_codes {
            assert!(
                emitted.contains(old_code),
                "v1 code disappeared: {old_code}"
            );
        }
        for appended in [
            "recovery.state.remote_unverified",
            "recovery.state.hash_control",
            "recovery.action.provide_attestation",
            "recovery.post.remote_attested",
            "recovery.policy.provide_attestation",
            "recovery.policy.bind_model",
            "recovery.capability.backfill_quality_tier",
            "recovery.capability.build_quality_tier",
            "recovery.capability.execute_bound_model_acquisition",
            "recovery.capability.execute_bound_semantic_index",
            "recovery.capability.ingest_quality_tier",
            "recovery.capability.import_model_bundle",
            "recovery.capability.rebuild_ann_generation",
            "recovery.capability.reindex_quality_tier",
            "recovery.capability.restart_daemon",
            "recovery.capability.resume_quality_generation",
        ] {
            assert!(
                emitted.contains(appended),
                "appended code absent: {appended}"
            );
        }
    }

    #[test]
    fn explicit_semantic_fails_closed_for_every_unready_state() {
        for state in unready_states() {
            let plan = planned(
                state.clone(),
                explicit(RetrievalTopology::FullProgressive),
                permissive(),
            );
            assert!(!plan.semantic_available);
            assert!(
                plan.response_contract.is_none(),
                "explicit semantic never degrades to lexical: {state:?}"
            );
        }
    }

    #[test]
    fn hybrid_degrades_with_metadata_exactly_when_unavailable() {
        for state in unready_states() {
            let plan = planned(
                state.clone(),
                hybrid(RetrievalTopology::FullProgressive),
                permissive(),
            );
            let response = plan
                .response_contract
                .expect("unavailable hybrid must carry response contract");
            assert_eq!(
                response.requested_topology(),
                RetrievalTopology::FullProgressive
            );
            assert_eq!(response.realized_topology(), RetrievalTopology::LexicalOnly);
            assert_eq!(response.coverage_ppm(), 0);
            assert_eq!(response.admitted_semantic_scores(), 0);
            assert_eq!(response.degradation_reason_code(), Some(state.state_code()));
        }
    }

    #[test]
    fn acquisition_never_claims_readiness() {
        for state in [missing(ModelTier::Quality), unloadable(ModelTier::Quality)] {
            let reacquire = matches!(state, SemanticReadiness::ModelUnloadable { .. });
            let plan = planned(
                state,
                explicit(RetrievalTopology::QualityOnly),
                permissive(),
            );
            assert_eq!(plan.retryability, Retryability::BlockedByCapability);
            let action = plan.action.expect("acquisition state has an action");
            assert_eq!(
                action.expected_postcondition,
                "recovery.post.model_acquired_unverified"
            );
            assert_ne!(action.expected_postcondition, "recovery.state.ready");
            assert!(action.network_required);
            assert!(action.consent_required);
            assert_eq!(action.preserves_old_data, !reacquire);
            assert_eq!(action.potentially_destructive, reacquire);
            assert_eq!(
                action.code,
                if reacquire {
                    "recovery.action.reacquire_model"
                } else {
                    "recovery.action.acquire_model"
                }
            );
            assert!(
                action.argv.is_empty(),
                "partial download syntax must not masquerade as exact bound authorization"
            );
            assert_eq!(
                action.prerequisites,
                ["recovery.capability.execute_bound_model_acquisition"]
            );
            let authorization = action
                .required_authorization
                .expect("acquisition binds exact authorization");
            assert_eq!(authorization.model_id, "fixture-semantic-model");
            assert_eq!(authorization.model_tier, ModelTier::Quality);
            assert_eq!(authorization.embedding_space, semantic_space());
            assert_eq!(authorization.manifest_fingerprint, "a".repeat(64));
            assert_eq!(authorization.upstream_revision, "revision-0123456789abcdef");
            assert_eq!(authorization.license_spdx, "Apache-2.0");
            assert_eq!(authorization.byte_budget, 42_000_000);
            assert_eq!(
                authorization.destination_class,
                ModelDestinationClass::ManagedCache
            );
            assert_eq!(authorization.destination_fingerprint, "b".repeat(64));
            assert_eq!(authorization.document_count, 12_345);
            assert_eq!(authorization.estimated_reindex_duration_ms, 98_765);
            assert_eq!(
                authorization.issued_at_unix_seconds,
                TEST_AUTHORIZATION_ISSUED_AT_UNIX_SECONDS
            );
            assert_eq!(
                authorization.expires_at_unix_seconds,
                TEST_AUTHORIZATION_EXPIRES_AT_UNIX_SECONDS
            );
            assert_eq!(authorization.nonce, TEST_AUTHORIZATION_NONCE);
            assert!(matches!(
                authorization.source,
                ModelAcquisitionSource::Network { .. }
            ));
        }
    }

    #[test]
    fn offline_policy_reports_missing_import_capability_without_fictional_argv() {
        let policy = RecoveryPolicy {
            interaction: InteractionPolicy::Interactive,
            network: NetworkPolicy::Offline,
            acquisition_authorization: None,
        };
        for state in [missing(ModelTier::Quality), unloadable(ModelTier::Quality)] {
            let plan = planned(
                state,
                hybrid(RetrievalTopology::FullProgressive),
                policy.clone(),
            );
            assert_eq!(plan.retryability, Retryability::BlockedByCapability);
            let action = plan.action.expect("action still recommended");
            assert!(
                action.argv.is_empty(),
                "offline recovery must not publish argv that fsfs cannot parse"
            );
            assert!(!action.network_required);
            assert!(action.consent_required);
            assert_eq!(
                action.prerequisites,
                ["recovery.capability.import_model_bundle"]
            );
            assert!(matches!(
                action.required_authorization.expect("offline scope").source,
                ModelAcquisitionSource::LocalBundle
            ));
        }
    }

    #[test]
    fn planned_semantic_mutations_never_publish_unbound_argv() {
        for state in unready_states() {
            let full_recovery = planned(
                state.clone(),
                explicit(RetrievalTopology::FullProgressive),
                permissive(),
            );
            if let Some(action) = full_recovery.action {
                assert!(
                    action.argv.is_empty(),
                    "quality/full recovery must not claim generic indexing realizes the \
                     requested topology for {state:?}: {:?}",
                    action.argv
                );
            }

            let request = explicit(RetrievalTopology::FastOnly);
            let fast_state = state_for_request(&state, request);
            let fast_recovery = plan(
                fast_state,
                request,
                permissive(),
                Some(&target_for(ModelTier::Fast)),
            )
            .expect("valid fast-tier recovery plan");
            let Some(action) = fast_recovery.action else {
                continue;
            };
            assert!(
                action.argv.is_empty(),
                "fast-only recovery must not execute until semantic producer identity and \
                 generation are bound for {state:?}: {:?}",
                action.argv
            );
        }

        for state in [
            SemanticReadiness::IndexAbsent,
            SemanticReadiness::IdentityMismatch,
            SemanticReadiness::ManifestUnsafe,
            SemanticReadiness::GenerationIncomplete,
            SemanticReadiness::IndexEmpty(ZeroSignalReason::NewlyCreatedEmpty),
            SemanticReadiness::IndexEmpty(ZeroSignalReason::NoUsableVectors),
        ] {
            let recovery = plan(
                state.clone(),
                explicit(RetrievalTopology::FastOnly),
                permissive(),
                Some(&target_for(ModelTier::Fast)),
            )
            .expect("fast-tier semantic mutation plan");
            let action = recovery.action.expect("semantic mutation action");
            assert!(action.argv.is_empty(), "{state:?}");
            assert!(
                action
                    .prerequisites
                    .contains(&"recovery.capability.execute_bound_semantic_index".to_owned()),
                "{state:?}"
            );
            assert_eq!(
                recovery.retryability,
                Retryability::BlockedByCapability,
                "{state:?}"
            );
        }

        let daemon = planned(
            SemanticReadiness::DaemonMismatch,
            explicit(RetrievalTopology::FullProgressive),
            permissive(),
        );
        let daemon_action = daemon.action.expect("daemon recovery");
        assert!(daemon_action.argv.is_empty());
        assert_eq!(
            daemon_action.prerequisites,
            ["recovery.capability.restart_daemon"]
        );
        assert_eq!(daemon.retryability, Retryability::BlockedByCapability);
    }

    #[test]
    fn noninteractive_policy_blocks_consent_actions_with_prerequisite() {
        let policy = RecoveryPolicy {
            interaction: InteractionPolicy::NonInteractive,
            network: NetworkPolicy::Allowed,
            acquisition_authorization: None,
        };
        for state in [
            SemanticReadiness::IdentityMismatch,
            SemanticReadiness::ManifestUnsafe,
            SemanticReadiness::IndexEmpty(ZeroSignalReason::NoUsableVectors),
        ] {
            let plan = planned(
                state.clone(),
                explicit(RetrievalTopology::FullProgressive),
                policy.clone(),
            );
            assert_eq!(
                plan.retryability,
                Retryability::BlockedByCapability,
                "{state:?}"
            );
            let action = plan.action.expect("destructive states have actions");
            assert!(action.consent_required);
            assert!(action.potentially_destructive);
            assert!(!action.preserves_old_data);
            assert!(
                action
                    .prerequisites
                    .contains(&"recovery.policy.grant_consent".to_owned())
            );
            assert!(
                action
                    .prerequisites
                    .contains(&"recovery.capability.reindex_quality_tier".to_owned())
            );
        }
    }

    #[test]
    fn request_scoped_emptiness_needs_no_system_action() {
        for reason in [
            ZeroSignalReason::CallerRequestedZeroK,
            ZeroSignalReason::FilterEliminatedAll,
            ZeroSignalReason::NonFiniteQuery,
            ZeroSignalReason::ZeroNormQuery,
        ] {
            let plan = planned(
                SemanticReadiness::IndexEmpty(reason),
                explicit(RetrievalTopology::QualityOnly),
                permissive(),
            );
            assert!(plan.action.is_none(), "{reason:?}");
            assert_eq!(plan.retryability, Retryability::AfterRequestChange);
            assert!(!plan.semantic_available);
        }
    }

    #[test]
    fn partial_coverage_is_available_with_a_backfill_action() {
        let plan = planned(
            SemanticReadiness::PartialQualityCoverage {
                provenance: VerifiedSemanticProvenance::Remote,
                coverage_ppm: 625_000,
            },
            hybrid(RetrievalTopology::FullProgressive),
            permissive(),
        );
        assert!(plan.semantic_available);
        assert_eq!(plan.provenance, SemanticProvenance::VerifiedRemote);
        let response = plan.response_contract.as_ref().expect("response contract");
        assert_eq!(
            response.realized_topology(),
            RetrievalTopology::PartialQuality {
                coverage_ppm: 625_000
            }
        );
        assert_eq!(response.coverage_ppm(), 625_000);
        assert_eq!(response.admitted_semantic_scores(), 0);
        assert!(response.degradation_reason_code().is_none());
        let action = plan.action.expect("backfill recommended");
        assert_eq!(action.code, "recovery.action.backfill_quality");
        assert!(action.preserves_old_data);
        // The lane serves partial results, but no current command can express
        // the required quality-only backfill.
        assert_eq!(plan.retryability, Retryability::BlockedByCapability);
        assert_eq!(
            action.prerequisites,
            ["recovery.capability.backfill_quality_tier"]
        );
    }

    #[test]
    fn partial_quality_coverage_only_backfills_topologies_that_request_quality() {
        let state = SemanticReadiness::PartialQualityCoverage {
            provenance: VerifiedSemanticProvenance::Remote,
            coverage_ppm: 625_000,
        };
        for mode in [RequestMode::ExplicitSemantic, RequestMode::Hybrid] {
            let fast = planned(
                state.clone(),
                RecoveryRequest {
                    mode,
                    requested_topology: RetrievalTopology::FastOnly,
                },
                permissive(),
            );
            assert!(fast.semantic_available);
            assert_eq!(fast.retryability, Retryability::NotNeeded);
            assert!(
                fast.action.is_none(),
                "complete fast-tier coverage needs no quality backfill"
            );
            let response = fast.response_contract.expect("fast response contract");
            assert_eq!(response.realized_topology(), RetrievalTopology::FastOnly);
            assert_eq!(response.coverage_ppm(), COMPLETE_COVERAGE_PPM);

            for requested_topology in [
                RetrievalTopology::QualityOnly,
                RetrievalTopology::FullProgressive,
            ] {
                let quality = planned(
                    state.clone(),
                    RecoveryRequest {
                        mode,
                        requested_topology,
                    },
                    permissive(),
                );
                assert!(quality.semantic_available);
                assert_eq!(quality.retryability, Retryability::BlockedByCapability);
                let action = quality.action.expect("quality request needs backfill");
                assert_eq!(action.code, "recovery.action.backfill_quality");
                assert!(action.argv.is_empty());
                assert_eq!(
                    action.prerequisites,
                    ["recovery.capability.backfill_quality_tier"]
                );
                let response = quality
                    .response_contract
                    .expect("quality response contract");
                assert_eq!(
                    response.realized_topology(),
                    RetrievalTopology::PartialQuality {
                        coverage_ppm: 625_000
                    }
                );
                assert_eq!(response.coverage_ppm(), 625_000);
            }
        }
    }

    #[test]
    fn request_topology_matrix_is_explicit_and_hash_isolated() {
        for request in semantic_requests().into_iter().chain([hash_control()]) {
            assert_eq!(request.validate(), Ok(request));
            let json = serde_json::to_string(&request).expect("serialize request");
            let decoded: RecoveryRequest =
                serde_json::from_str(&json).expect("deserialize valid request");
            assert_eq!(decoded, request);
        }

        let invalid = [
            explicit(RetrievalTopology::LexicalOnly),
            explicit(RetrievalTopology::HashControl),
            explicit(RetrievalTopology::PartialQuality {
                coverage_ppm: 500_000,
            }),
            hybrid(RetrievalTopology::LexicalOnly),
            hybrid(RetrievalTopology::HashControl),
            hybrid(RetrievalTopology::PartialQuality {
                coverage_ppm: 500_000,
            }),
            RecoveryRequest {
                mode: RequestMode::HashControl,
                requested_topology: RetrievalTopology::LexicalOnly,
            },
            RecoveryRequest {
                mode: RequestMode::HashControl,
                requested_topology: RetrievalTopology::FastOnly,
            },
            RecoveryRequest {
                mode: RequestMode::HashControl,
                requested_topology: RetrievalTopology::QualityOnly,
            },
            RecoveryRequest {
                mode: RequestMode::HashControl,
                requested_topology: RetrievalTopology::FullProgressive,
            },
            RecoveryRequest {
                mode: RequestMode::HashControl,
                requested_topology: RetrievalTopology::PartialQuality {
                    coverage_ppm: 500_000,
                },
            },
        ];
        for request in invalid {
            assert!(matches!(
                request.validate(),
                Err(RecoveryContractError::InvalidRequestTopology { .. })
            ));
            let json = serde_json::to_string(&request).expect("serialize invalid request");
            assert!(
                serde_json::from_str::<RecoveryRequest>(&json).is_err(),
                "serde must not bypass request validation: {request:?}"
            );
        }

        for state in [
            SemanticReadiness::Ready {
                provenance: VerifiedSemanticProvenance::Local,
            },
            missing(ModelTier::Quality),
            SemanticReadiness::RemoteUnverified,
        ] {
            assert_eq!(
                plan(state, hash_control(), permissive(), Some(&target())),
                Err(RecoveryContractError::HashControlModeReadinessMismatch)
            );
        }
        for request in semantic_requests() {
            assert_eq!(
                plan(
                    SemanticReadiness::HashControl,
                    request,
                    permissive(),
                    Some(&target()),
                ),
                Err(RecoveryContractError::HashControlModeReadinessMismatch)
            );
        }
    }

    #[test]
    fn provenance_matrix_never_promotes_unverified_or_hash_producers() {
        for (producer, expected) in [
            (
                VerifiedSemanticProvenance::Local,
                SemanticProvenance::VerifiedLocal,
            ),
            (
                VerifiedSemanticProvenance::Remote,
                SemanticProvenance::VerifiedRemote,
            ),
            (
                VerifiedSemanticProvenance::Daemon,
                SemanticProvenance::VerifiedDaemon,
            ),
        ] {
            let plan = planned(
                SemanticReadiness::Ready {
                    provenance: producer,
                },
                explicit(RetrievalTopology::QualityOnly),
                permissive(),
            );
            assert!(plan.semantic_available);
            assert_eq!(plan.provenance, expected);
            assert_eq!(
                plan.response_contract
                    .expect("verified producer may respond")
                    .realized_topology(),
                RetrievalTopology::QualityOnly
            );
        }

        let remote = planned(
            SemanticReadiness::RemoteUnverified,
            explicit(RetrievalTopology::QualityOnly),
            permissive(),
        );
        assert_eq!(remote.provenance, SemanticProvenance::UnverifiedRemote);
        assert!(!remote.semantic_available);
        assert!(remote.response_contract.is_none());
        let action = remote.action.expect("attestation recovery");
        assert_eq!(action.code, "recovery.action.provide_attestation");
        assert!(action.argv.is_empty());
        assert_eq!(
            action.prerequisites,
            ["recovery.policy.provide_attestation"]
        );

        let hash = planned(SemanticReadiness::HashControl, hash_control(), permissive());
        assert_eq!(hash.provenance, SemanticProvenance::HashControl);
        assert!(!hash.semantic_available);
        assert_eq!(
            hash.response_contract
                .expect("explicit hash control response")
                .realized_topology(),
            RetrievalTopology::HashControl
        );
    }

    #[test]
    fn response_contract_accepts_only_truthful_topology_and_coverage_pairs() {
        for (requested, realized, coverage) in [
            (
                RetrievalTopology::FastOnly,
                RetrievalTopology::FastOnly,
                COMPLETE_COVERAGE_PPM,
            ),
            (
                RetrievalTopology::QualityOnly,
                RetrievalTopology::QualityOnly,
                COMPLETE_COVERAGE_PPM,
            ),
            (
                RetrievalTopology::FullProgressive,
                RetrievalTopology::FullProgressive,
                COMPLETE_COVERAGE_PPM,
            ),
            (
                RetrievalTopology::FullProgressive,
                RetrievalTopology::FastOnly,
                COMPLETE_COVERAGE_PPM,
            ),
            (
                RetrievalTopology::FullProgressive,
                RetrievalTopology::QualityOnly,
                COMPLETE_COVERAGE_PPM,
            ),
            (
                RetrievalTopology::QualityOnly,
                RetrievalTopology::PartialQuality {
                    coverage_ppm: 250_000,
                },
                250_000,
            ),
            (
                RetrievalTopology::FullProgressive,
                RetrievalTopology::PartialQuality {
                    coverage_ppm: 750_000,
                },
                750_000,
            ),
        ] {
            let contract = SemanticResponseContract::new(requested, realized, coverage, 7, None)
                .expect("truthful semantic response");
            assert_eq!(contract.admitted_semantic_scores(), 7);
            let json = serde_json::to_string(&contract).expect("serialize response");
            assert!(
                serde_json::from_str::<SemanticResponseContractWire>(&json).is_ok(),
                "wire shape remains decodable only into the private raw type"
            );
        }

        for requested in [
            RetrievalTopology::FastOnly,
            RetrievalTopology::QualityOnly,
            RetrievalTopology::FullProgressive,
        ] {
            SemanticResponseContract::new(
                requested,
                RetrievalTopology::LexicalOnly,
                0,
                0,
                Some("recovery.state.model_missing".to_owned()),
            )
            .expect("typed lexical degradation");
        }
        SemanticResponseContract::new(
            RetrievalTopology::HashControl,
            RetrievalTopology::HashControl,
            0,
            0,
            None,
        )
        .expect("explicit hash control");
    }

    #[test]
    fn response_contract_rejects_silent_or_impossible_contribution_claims() {
        assert_eq!(
            SemanticResponseContract::new(
                RetrievalTopology::LexicalOnly,
                RetrievalTopology::LexicalOnly,
                0,
                0,
                Some("recovery.state.model_missing".to_owned()),
            ),
            Err(RecoveryContractError::UnexpectedDegradationReason)
        );
        SemanticResponseContract::new(
            RetrievalTopology::LexicalOnly,
            RetrievalTopology::LexicalOnly,
            0,
            0,
            None,
        )
        .expect("an explicitly lexical request is not a degradation");
        assert!(matches!(
            SemanticResponseContract::new(
                RetrievalTopology::PartialQuality {
                    coverage_ppm: 500_000,
                },
                RetrievalTopology::LexicalOnly,
                0,
                0,
                Some("recovery.state.model_missing".to_owned()),
            ),
            Err(RecoveryContractError::IncompatibleResponseTopology { .. })
        ));
        for (requested, realized) in [
            (RetrievalTopology::FastOnly, RetrievalTopology::QualityOnly),
            (
                RetrievalTopology::QualityOnly,
                RetrievalTopology::FullProgressive,
            ),
            (
                RetrievalTopology::HashControl,
                RetrievalTopology::LexicalOnly,
            ),
        ] {
            assert!(matches!(
                SemanticResponseContract::new(requested, realized, COMPLETE_COVERAGE_PPM, 0, None,),
                Err(RecoveryContractError::IncompatibleResponseTopology { .. })
            ));
        }

        for (realized, coverage) in [
            (RetrievalTopology::FastOnly, 0),
            (RetrievalTopology::QualityOnly, COMPLETE_COVERAGE_PPM - 1),
            (
                RetrievalTopology::FullProgressive,
                COMPLETE_COVERAGE_PPM + 1,
            ),
            (RetrievalTopology::LexicalOnly, 1),
            (RetrievalTopology::HashControl, 1),
            (RetrievalTopology::PartialQuality { coverage_ppm: 0 }, 0),
            (
                RetrievalTopology::PartialQuality {
                    coverage_ppm: COMPLETE_COVERAGE_PPM,
                },
                COMPLETE_COVERAGE_PPM,
            ),
            (
                RetrievalTopology::PartialQuality {
                    coverage_ppm: 250_000,
                },
                500_000,
            ),
        ] {
            let requested = match realized {
                RetrievalTopology::HashControl => RetrievalTopology::HashControl,
                RetrievalTopology::LexicalOnly | RetrievalTopology::FastOnly => {
                    RetrievalTopology::FastOnly
                }
                RetrievalTopology::QualityOnly | RetrievalTopology::PartialQuality { .. } => {
                    RetrievalTopology::QualityOnly
                }
                RetrievalTopology::FullProgressive => RetrievalTopology::FullProgressive,
            };
            let reason = matches!(realized, RetrievalTopology::LexicalOnly)
                .then(|| "recovery.state.model_missing".to_owned());
            assert!(matches!(
                SemanticResponseContract::new(requested, realized, coverage, 0, reason),
                Err(RecoveryContractError::InvalidCoverage { .. })
            ));
        }

        for topology in [
            RetrievalTopology::LexicalOnly,
            RetrievalTopology::HashControl,
        ] {
            let requested = if matches!(topology, RetrievalTopology::HashControl) {
                RetrievalTopology::HashControl
            } else {
                RetrievalTopology::FastOnly
            };
            let reason = matches!(topology, RetrievalTopology::LexicalOnly)
                .then(|| "recovery.state.model_missing".to_owned());
            assert!(matches!(
                SemanticResponseContract::new(requested, topology, 0, 1, reason),
                Err(RecoveryContractError::NonSemanticScoresAdmitted { .. })
            ));
        }

        assert_eq!(
            SemanticResponseContract::new(
                RetrievalTopology::FastOnly,
                RetrievalTopology::LexicalOnly,
                0,
                0,
                None,
            ),
            Err(RecoveryContractError::MissingDegradationReason)
        );
        assert_eq!(
            SemanticResponseContract::new(
                RetrievalTopology::FastOnly,
                RetrievalTopology::FastOnly,
                COMPLETE_COVERAGE_PPM,
                1,
                Some("recovery.state.model_missing".to_owned()),
            ),
            Err(RecoveryContractError::UnexpectedDegradationReason)
        );
    }

    #[test]
    fn noninteractive_acquisition_requires_exact_scoped_authorization() {
        for network in [NetworkPolicy::Allowed, NetworkPolicy::Offline] {
            let required = planned(
                missing(ModelTier::Quality),
                explicit(RetrievalTopology::QualityOnly),
                RecoveryPolicy {
                    interaction: InteractionPolicy::Interactive,
                    network,
                    acquisition_authorization: None,
                },
            )
            .action
            .expect("acquisition action")
            .required_authorization
            .expect("scoped authorization");

            let exact = planned(
                missing(ModelTier::Quality),
                explicit(RetrievalTopology::QualityOnly),
                RecoveryPolicy {
                    interaction: InteractionPolicy::NonInteractive,
                    network,
                    acquisition_authorization: Some(required.clone()),
                },
            );
            let exact_action = exact.action.expect("acquisition action");
            assert_eq!(exact.retryability, Retryability::BlockedByCapability);
            assert_eq!(
                exact_action.prerequisites,
                [match network {
                    NetworkPolicy::Allowed => {
                        "recovery.capability.execute_bound_model_acquisition"
                    }
                    NetworkPolicy::Offline => "recovery.capability.import_model_bundle",
                }]
            );

            let mut mismatches = Vec::new();
            let mut authorization = required.clone();
            authorization.model_id.push_str("-different");
            authorization.embedding_space.logical_model_id = authorization.model_id.clone();
            mismatches.push(("model_id", authorization));
            let mut authorization = required.clone();
            authorization.model_tier = match required.model_tier {
                ModelTier::Fast => ModelTier::Quality,
                ModelTier::Quality => ModelTier::Fast,
            };
            mismatches.push(("model_tier", authorization));
            let mut authorization = required.clone();
            authorization.embedding_space.dimension += 1;
            mismatches.push(("embedding_space", authorization));
            let mut authorization = required.clone();
            authorization.manifest_fingerprint = "c".repeat(64);
            mismatches.push(("manifest_fingerprint", authorization));
            let mut authorization = required.clone();
            authorization.upstream_revision.push_str("-different");
            authorization.embedding_space.immutable_revision =
                authorization.upstream_revision.clone();
            mismatches.push(("upstream_revision", authorization));
            let mut authorization = required.clone();
            authorization.license_spdx = "MIT".to_owned();
            mismatches.push(("license_spdx", authorization));
            let mut authorization = required.clone();
            authorization.source = match network {
                NetworkPolicy::Allowed => ModelAcquisitionSource::LocalBundle,
                NetworkPolicy::Offline => ModelAcquisitionSource::Network {
                    source_hosts: vec!["models.example.test".to_owned()],
                },
            };
            mismatches.push(("source", authorization));
            let mut authorization = required.clone();
            authorization.byte_budget += 1;
            mismatches.push(("byte_budget", authorization));
            let mut authorization = required.clone();
            authorization.destination_class = ModelDestinationClass::ExplicitDirectory;
            mismatches.push(("destination_class", authorization));
            let mut authorization = required.clone();
            authorization.destination_fingerprint = "d".repeat(64);
            mismatches.push(("destination_fingerprint", authorization));
            let mut authorization = required.clone();
            authorization.document_count += 1;
            mismatches.push(("document_count", authorization));
            let mut authorization = required.clone();
            authorization.estimated_reindex_duration_ms += 1;
            mismatches.push(("estimated_reindex_duration_ms", authorization));
            let mut authorization = required.clone();
            authorization.issued_at_unix_seconds -= 1;
            mismatches.push(("issued_at_unix_seconds", authorization));
            let mut authorization = required.clone();
            authorization.expires_at_unix_seconds += 1;
            mismatches.push(("expires_at_unix_seconds", authorization));
            let mut authorization = required.clone();
            authorization.nonce = "fedcba9876543210fedcba9876543210".to_owned();
            mismatches.push(("nonce", authorization));

            for (field, authorization) in mismatches {
                let result = plan(
                    missing(ModelTier::Quality),
                    explicit(RetrievalTopology::QualityOnly),
                    RecoveryPolicy {
                        interaction: InteractionPolicy::NonInteractive,
                        network,
                        acquisition_authorization: Some(authorization),
                    },
                    Some(&target()),
                );
                assert_eq!(
                    result,
                    Err(RecoveryContractError::MismatchedAcquisitionAuthorization { field }),
                    "scope mismatch must fail closed for {field}"
                );
            }
        }
    }

    #[test]
    fn acquisition_authorization_enforces_window_and_nonce_boundaries() {
        let authorization = target()
            .authorization_for(NetworkPolicy::Allowed)
            .expect("valid authorization fixture");

        for (issued_at_unix_seconds, expires_at_unix_seconds) in [(100, 100), (101, 100)] {
            let mut invalid = authorization.clone();
            invalid.issued_at_unix_seconds = issued_at_unix_seconds;
            invalid.expires_at_unix_seconds = expires_at_unix_seconds;
            assert_eq!(
                invalid.validate(),
                Err(
                    RecoveryContractError::InvalidAcquisitionAuthorizationWindow {
                        issued_at_unix_seconds,
                        expires_at_unix_seconds,
                    }
                )
            );
        }

        let mut maximum_lifetime = authorization.clone();
        maximum_lifetime.issued_at_unix_seconds = 1_000;
        maximum_lifetime.expires_at_unix_seconds =
            1_000 + MAX_MODEL_ACQUISITION_AUTHORIZATION_LIFETIME_SECONDS;
        maximum_lifetime
            .validate()
            .expect("the maximum authorization lifetime is inclusive");

        let mut excessive_lifetime = maximum_lifetime.clone();
        excessive_lifetime.expires_at_unix_seconds =
            excessive_lifetime.expires_at_unix_seconds.saturating_add(1);
        assert_eq!(
            excessive_lifetime.validate(),
            Err(
                RecoveryContractError::AcquisitionAuthorizationLifetimeExceeded {
                    lifetime_seconds: MAX_MODEL_ACQUISITION_AUTHORIZATION_LIFETIME_SECONDS + 1,
                    max_lifetime_seconds: MAX_MODEL_ACQUISITION_AUTHORIZATION_LIFETIME_SECONDS,
                }
            )
        );

        for nonce in [
            "0123456789abcdef0123456789abcde",
            "0123456789abcdef0123456789abcdef0",
            "0123456789ABCDEF0123456789ABCDEF",
            "0123456789abcdef0123456789abcdeg",
            "00000000000000000000000000000000",
        ] {
            let mut invalid = authorization.clone();
            invalid.nonce = nonce.to_owned();
            assert_eq!(
                invalid.validate(),
                Err(RecoveryContractError::InvalidAcquisitionAuthorizationNonce),
                "invalid nonce unexpectedly admitted: {nonce}"
            );
        }

        for nonce in [
            "00000000000000000000000000000001",
            "ffffffffffffffffffffffffffffffff",
        ] {
            let mut valid = authorization.clone();
            valid.nonce = nonce.to_owned();
            valid
                .validate()
                .unwrap_or_else(|error| panic!("valid nonce {nonce} rejected: {error}"));
        }
    }

    #[test]
    fn acquisition_authorization_uses_trusted_time_and_exclusive_expiry() {
        let authorization = target()
            .authorization_for(NetworkPolicy::Allowed)
            .expect("valid authorization fixture");
        let before_issuance = authorization.issued_at_unix_seconds.saturating_sub(1);

        assert_eq!(
            authorization.validate_at(before_issuance),
            Err(RecoveryContractError::AcquisitionAuthorizationNotYetValid {
                issued_at_unix_seconds: authorization.issued_at_unix_seconds,
                evaluation_time_unix_seconds: before_issuance,
            })
        );
        authorization
            .validate_at(authorization.issued_at_unix_seconds)
            .expect("authorization is valid at its inclusive issuance boundary");
        authorization
            .validate_at(authorization.expires_at_unix_seconds - 1)
            .expect("authorization is valid immediately before expiry");
        assert_eq!(
            authorization.validate_at(authorization.expires_at_unix_seconds),
            Err(RecoveryContractError::AcquisitionAuthorizationExpired {
                expires_at_unix_seconds: authorization.expires_at_unix_seconds,
                evaluation_time_unix_seconds: authorization.expires_at_unix_seconds,
            })
        );
    }

    #[test]
    fn authorization_binding_rejects_surplus_and_stale_scopes() {
        let required = target()
            .authorization_for(NetworkPolicy::Allowed)
            .expect("valid authorization fixture");
        assert_eq!(
            validate_authorization_binding(None, Some(&required), TEST_NOW_UNIX_SECONDS),
            Err(RecoveryContractError::SurplusAcquisitionAuthorization)
        );

        assert_eq!(
            validate_authorization_binding(Some(&required), None, required.expires_at_unix_seconds,),
            Err(RecoveryContractError::AcquisitionAuthorizationExpired {
                expires_at_unix_seconds: required.expires_at_unix_seconds,
                evaluation_time_unix_seconds: required.expires_at_unix_seconds,
            })
        );

        let mut stale_supplied = required.clone();
        stale_supplied.expires_at_unix_seconds = TEST_NOW_UNIX_SECONDS;
        assert_eq!(
            validate_authorization_binding(
                Some(&required),
                Some(&stale_supplied),
                TEST_NOW_UNIX_SECONDS,
            ),
            Err(RecoveryContractError::AcquisitionAuthorizationExpired {
                expires_at_unix_seconds: TEST_NOW_UNIX_SECONDS,
                evaluation_time_unix_seconds: TEST_NOW_UNIX_SECONDS,
            })
        );
    }

    #[test]
    fn planner_rejects_surplus_authorization_when_no_action_requires_it() {
        let surplus = target()
            .authorization_for(NetworkPolicy::Allowed)
            .expect("valid authorization fixture");
        let result = plan(
            SemanticReadiness::Ready {
                provenance: VerifiedSemanticProvenance::Local,
            },
            explicit(RetrievalTopology::FastOnly),
            RecoveryPolicy {
                interaction: InteractionPolicy::NonInteractive,
                network: NetworkPolicy::Allowed,
                acquisition_authorization: Some(surplus),
            },
            None,
        );
        assert_eq!(
            result,
            Err(RecoveryContractError::SurplusAcquisitionAuthorization)
        );
    }

    #[test]
    fn promotion_and_execution_each_recheck_authorization_expiry() {
        let state = missing(ModelTier::Quality);
        let request = explicit(RetrievalTopology::QualityOnly);
        let acquisition_target = target();
        let presentation_policy = RecoveryPolicy {
            interaction: InteractionPolicy::Interactive,
            network: NetworkPolicy::Allowed,
            acquisition_authorization: None,
        };
        let presentation_plan = super::plan(TrustedRecoveryContext::new(
            &state,
            request,
            &presentation_policy,
            Some(&acquisition_target),
            acquisition_target.expires_at_unix_seconds - 1,
        ))
        .expect("interactive presentation plan before authorization");
        assert_eq!(
            presentation_plan
                .validate_for_execution_at(acquisition_target.expires_at_unix_seconds - 1),
            Err(RecoveryContractError::MissingAcquisitionAuthorization),
            "a required but ungranted authorization must never pass the execution gate"
        );

        let exact_authorization = acquisition_target
            .authorization_for(NetworkPolicy::Allowed)
            .expect("exact authorization");
        let policy = RecoveryPolicy {
            interaction: InteractionPolicy::NonInteractive,
            network: NetworkPolicy::Allowed,
            acquisition_authorization: Some(exact_authorization),
        };
        let canonical = super::plan(TrustedRecoveryContext::new(
            &state,
            request,
            &policy,
            Some(&acquisition_target),
            TEST_NOW_UNIX_SECONDS,
        ))
        .expect("serialize a currently valid recovery plan");
        let serialized = serde_json::to_value(canonical).expect("serialize recovery plan");
        let promoted = serde_json::from_value::<UntrustedRecoveryPlan>(serialized.clone())
            .expect("decode untrusted recovery plan")
            .validate_against(TrustedRecoveryContext::new(
                &state,
                request,
                &policy,
                Some(&acquisition_target),
                acquisition_target.expires_at_unix_seconds - 1,
            ))
            .expect("promotion succeeds immediately before expiry");
        promoted
            .validate_for_execution_at(acquisition_target.expires_at_unix_seconds - 1)
            .expect("exact supplied authorization executes immediately before expiry");
        assert_eq!(
            promoted.validate_for_execution_at(acquisition_target.expires_at_unix_seconds),
            Err(RecoveryContractError::AcquisitionAuthorizationExpired {
                expires_at_unix_seconds: acquisition_target.expires_at_unix_seconds,
                evaluation_time_unix_seconds: acquisition_target.expires_at_unix_seconds,
            }),
            "a previously promoted plan is not a timeless execution capability"
        );

        let untrusted: UntrustedRecoveryPlan =
            serde_json::from_value(serialized).expect("decode second untrusted recovery plan");
        assert_eq!(
            untrusted.validate_against(TrustedRecoveryContext::new(
                &state,
                request,
                &policy,
                Some(&acquisition_target),
                acquisition_target.expires_at_unix_seconds,
            )),
            Err(RecoveryContractError::AcquisitionAuthorizationExpired {
                expires_at_unix_seconds: acquisition_target.expires_at_unix_seconds,
                evaluation_time_unix_seconds: acquisition_target.expires_at_unix_seconds,
            })
        );
    }

    #[test]
    fn planner_rejects_programmatically_constructed_invalid_policy_authorization() {
        let mut invalid = target()
            .authorization_for(NetworkPolicy::Allowed)
            .expect("valid authorization fixture");
        invalid.byte_budget = 0;
        let result = plan(
            SemanticReadiness::Ready {
                provenance: VerifiedSemanticProvenance::Local,
            },
            explicit(RetrievalTopology::FastOnly),
            RecoveryPolicy {
                interaction: InteractionPolicy::NonInteractive,
                network: NetworkPolicy::Allowed,
                acquisition_authorization: Some(invalid),
            },
            None,
        );
        assert_eq!(
            result,
            Err(RecoveryContractError::ZeroAcquisitionByteBudget),
            "planner must not serialize invalid authorization even when the ready action ignores it"
        );
    }

    #[test]
    fn acquisition_target_must_be_bound_and_well_formed() {
        let unbound = plan(
            missing(ModelTier::Fast),
            explicit(RetrievalTopology::FastOnly),
            permissive(),
            None,
        )
        .expect("missing binding yields a non-executable plan");
        assert_eq!(unbound.retryability, Retryability::BlockedByCapability);
        let action = unbound.action.expect("acquisition action");
        assert!(action.required_authorization.is_none());
        assert!(
            action.argv.is_empty(),
            "an unbound target cannot name an exact model and must not be executable"
        );
        assert_eq!(
            action.prerequisites,
            [
                "recovery.capability.execute_bound_model_acquisition",
                "recovery.policy.bind_model",
            ]
        );

        let mut malformed = target();
        malformed.model_id = " ".to_owned();
        assert!(matches!(
            plan(
                missing(ModelTier::Quality),
                explicit(RetrievalTopology::QualityOnly),
                permissive(),
                Some(&malformed),
            ),
            Err(RecoveryContractError::InvalidAcquisitionScopeField { field: "model_id" })
        ));

        let mut malformed = target();
        malformed.manifest_fingerprint = " ".to_owned();
        assert!(matches!(
            plan(
                missing(ModelTier::Quality),
                explicit(RetrievalTopology::QualityOnly),
                permissive(),
                Some(&malformed),
            ),
            Err(RecoveryContractError::InvalidAcquisitionScopeField {
                field: "manifest_fingerprint"
            })
        ));

        let mut malformed = target();
        malformed.upstream_revision = "revision\ninjected".to_owned();
        assert!(matches!(
            plan(
                missing(ModelTier::Quality),
                explicit(RetrievalTopology::QualityOnly),
                permissive(),
                Some(&malformed),
            ),
            Err(RecoveryContractError::InvalidAcquisitionScopeField {
                field: "upstream_revision"
            })
        ));

        let mut malformed = target();
        malformed.license_spdx.clear();
        assert!(matches!(
            plan(
                missing(ModelTier::Quality),
                explicit(RetrievalTopology::QualityOnly),
                permissive(),
                Some(&malformed),
            ),
            Err(RecoveryContractError::InvalidAcquisitionScopeField {
                field: "license_spdx"
            })
        ));

        let mut malformed = target();
        malformed.destination_fingerprint.clear();
        assert!(matches!(
            plan(
                missing(ModelTier::Quality),
                explicit(RetrievalTopology::QualityOnly),
                permissive(),
                Some(&malformed),
            ),
            Err(RecoveryContractError::InvalidAcquisitionScopeField {
                field: "destination_fingerprint"
            })
        ));

        let mut malformed = target();
        malformed.byte_budget = 0;
        assert_eq!(
            plan(
                missing(ModelTier::Quality),
                explicit(RetrievalTopology::QualityOnly),
                permissive(),
                Some(&malformed),
            ),
            Err(RecoveryContractError::ZeroAcquisitionByteBudget)
        );

        let mut missing_host = target();
        missing_host.network_source_hosts.clear();
        assert_eq!(
            plan(
                missing(ModelTier::Quality),
                explicit(RetrievalTopology::QualityOnly),
                permissive(),
                Some(&missing_host),
            ),
            Err(RecoveryContractError::MissingNetworkSourceHosts)
        );
        let offline = plan(
            missing(ModelTier::Quality),
            explicit(RetrievalTopology::QualityOnly),
            RecoveryPolicy {
                interaction: InteractionPolicy::Interactive,
                network: NetworkPolicy::Offline,
                acquisition_authorization: None,
            },
            Some(&missing_host),
        )
        .expect("local bundle does not depend on network hosts");
        assert!(matches!(
            offline
                .action
                .expect("offline action")
                .required_authorization
                .expect("offline authorization")
                .source,
            ModelAcquisitionSource::LocalBundle
        ));
    }

    #[test]
    fn acquisition_tier_and_embedding_space_are_anchored_to_trusted_inputs() {
        assert!(matches!(
            plan(
                missing(ModelTier::Quality),
                explicit(RetrievalTopology::FastOnly),
                permissive(),
                Some(&target_for(ModelTier::Quality)),
            ),
            Err(RecoveryContractError::UnavailableTierTopologyMismatch {
                tier: ModelTier::Quality,
                ..
            })
        ));
        assert!(matches!(
            plan(
                missing(ModelTier::Fast),
                explicit(RetrievalTopology::QualityOnly),
                permissive(),
                Some(&target_for(ModelTier::Fast)),
            ),
            Err(RecoveryContractError::UnavailableTierTopologyMismatch {
                tier: ModelTier::Fast,
                ..
            })
        ));
        assert_eq!(
            plan(
                missing(ModelTier::Fast),
                explicit(RetrievalTopology::FullProgressive),
                permissive(),
                Some(&target_for(ModelTier::Quality)),
            ),
            Err(RecoveryContractError::AcquisitionTargetTierMismatch {
                readiness_tier: ModelTier::Fast,
                target_tier: ModelTier::Quality,
            })
        );
        plan(
            missing(ModelTier::Fast),
            explicit(RetrievalTopology::FullProgressive),
            permissive(),
            Some(&target_for(ModelTier::Fast)),
        )
        .expect("full progressive may recover the exact unavailable fast tier");
        plan(
            missing(ModelTier::Quality),
            explicit(RetrievalTopology::FullProgressive),
            permissive(),
            Some(&target_for(ModelTier::Quality)),
        )
        .expect("full progressive may recover the exact unavailable quality tier");

        let mut malformed = target();
        malformed.embedding_space.dimension = 0;
        assert!(matches!(
            plan(
                missing(ModelTier::Quality),
                explicit(RetrievalTopology::QualityOnly),
                permissive(),
                Some(&malformed),
            ),
            Err(RecoveryContractError::InvalidAcquisitionSpaceIdentity { .. })
        ));

        let mut hash_target = target();
        hash_target.embedding_space =
            EmbeddingIdentityBundleV1::explicit_test_model("hash-control", 384).space;
        hash_target.model_id = hash_target.embedding_space.logical_model_id.clone();
        hash_target.upstream_revision = hash_target.embedding_space.immutable_revision.clone();
        assert_eq!(
            plan(
                missing(ModelTier::Quality),
                explicit(RetrievalTopology::QualityOnly),
                permissive(),
                Some(&hash_target),
            ),
            Err(RecoveryContractError::NonSemanticAcquisitionSpace)
        );

        let mut inconsistent = target();
        inconsistent.model_id = "other-semantic-model".to_owned();
        assert_eq!(
            plan(
                missing(ModelTier::Quality),
                explicit(RetrievalTopology::QualityOnly),
                permissive(),
                Some(&inconsistent),
            ),
            Err(RecoveryContractError::InconsistentAcquisitionIdentity { field: "model_id" })
        );

        let mut inconsistent = target();
        inconsistent.upstream_revision = "different-immutable-revision".to_owned();
        assert_eq!(
            plan(
                missing(ModelTier::Quality),
                explicit(RetrievalTopology::QualityOnly),
                permissive(),
                Some(&inconsistent),
            ),
            Err(RecoveryContractError::InconsistentAcquisitionIdentity {
                field: "upstream_revision"
            })
        );
    }

    #[test]
    fn network_source_hosts_are_path_free_credential_free_authorities() {
        for host in [
            "models.example.test",
            "MODELS.EXAMPLE.TEST:443",
            "localhost",
            "127.0.0.1",
            "127.0.0.1:8443",
            "[2001:db8::1]",
            "[2001:db8::1]:443",
        ] {
            let mut valid = target();
            valid.network_source_hosts = vec![host.to_owned()];
            plan(
                missing(ModelTier::Quality),
                explicit(RetrievalTopology::QualityOnly),
                permissive(),
                Some(&valid),
            )
            .expect("credential-free host authority is valid");
        }

        for host in [
            "",
            " ",
            "https://models.example.test",
            "user:secret@models.example.test",
            "models.example.test/path",
            "models.example.test?variant=quality",
            "models.example.test#fragment",
            "models.example.test\\path",
            "models .example.test",
            "models.example.test\n",
            "models_example.test",
            "-models.example.test",
            "models-.example.test",
            "models..example.test",
            "models.example.test.",
            "999.999.999.999",
            "models.example.test:",
            "models.example.test:0",
            "models.example.test:+443",
            "models.example.test:65536",
            "models.example.test:https",
            "2001:db8::1",
            "[2001:db8::1",
            "[2001:db8::1]suffix",
            "[2001:db8::1]:0",
            "[2001:db8::1]:65536",
            "[not-ipv6]",
        ] {
            let mut malformed = target();
            malformed.network_source_hosts = vec![host.to_owned()];
            assert_eq!(
                plan(
                    missing(ModelTier::Quality),
                    explicit(RetrievalTopology::QualityOnly),
                    permissive(),
                    Some(&malformed),
                ),
                Err(RecoveryContractError::InvalidNetworkSourceHost),
                "unexpectedly accepted source host {host:?}"
            );
        }
    }

    #[test]
    fn invalid_partial_readiness_never_enters_a_plan() {
        for coverage_ppm in [0, COMPLETE_COVERAGE_PPM, COMPLETE_COVERAGE_PPM + 1] {
            assert!(matches!(
                plan(
                    SemanticReadiness::PartialQualityCoverage {
                        provenance: VerifiedSemanticProvenance::Local,
                        coverage_ppm,
                    },
                    explicit(RetrievalTopology::QualityOnly),
                    permissive(),
                    Some(&target()),
                ),
                Err(RecoveryContractError::InvalidCoverage { .. })
            ));
        }
    }

    #[test]
    fn serde_rejects_unknown_versions_fields_and_mismatched_scopes() {
        let state = missing(ModelTier::Quality);
        let request = hybrid(RetrievalTopology::FullProgressive);
        let policy = permissive();
        let acquisition_target = target();
        let canonical = plan(
            state.clone(),
            request,
            policy.clone(),
            Some(&acquisition_target),
        )
        .expect("canonical plan");
        let plan_json = serde_json::to_value(canonical).expect("serialize plan");
        let rejects = |value| {
            decode_and_validate(value, &state, request, &policy, Some(&acquisition_target)).is_err()
        };

        for legacy_version in [
            "frankensearch.recovery_plan.v1",
            "frankensearch.recovery_plan.v2",
            "frankensearch.recovery_plan.v3",
        ] {
            let mut changed = plan_json.clone();
            changed["schema_version"] = serde_json::Value::String(legacy_version.to_owned());
            assert!(
                rejects(changed),
                "legacy plan schema unexpectedly decoded: {legacy_version}"
            );
        }

        let mut changed = plan_json.clone();
        changed["unknown_contract_field"] = serde_json::Value::Bool(true);
        assert!(rejects(changed));

        let mut changed = plan_json.clone();
        changed["state_code"] = serde_json::Value::String("recovery.state.ready".to_owned());
        assert!(rejects(changed));

        let mut changed = plan_json.clone();
        changed["provenance"] = serde_json::Value::String("verified_local".to_owned());
        assert!(rejects(changed));

        let mut changed = plan_json.clone();
        changed["semantic_available"] = serde_json::Value::Bool(true);
        assert!(rejects(changed));

        let mut changed = plan_json.clone();
        changed["retryability"] = serde_json::Value::String("not_needed".to_owned());
        assert!(rejects(changed));

        let action_mutations = [
            ("code", serde_json::json!("recovery.action.reacquire_model")),
            ("explanation", serde_json::json!("trust the payload")),
            (
                "argv",
                serde_json::json!(["fsfs", "download-models", "--model", "different-model"]),
            ),
            ("network_required", serde_json::json!(false)),
            ("consent_required", serde_json::json!(false)),
            ("preserves_old_data", serde_json::json!(false)),
            ("potentially_destructive", serde_json::json!(true)),
            (
                "prerequisites",
                serde_json::json!(["recovery.policy.allow_network"]),
            ),
            (
                "expected_postcondition",
                serde_json::json!("recovery.state.ready"),
            ),
        ];
        for (field, value) in action_mutations {
            let mut changed = plan_json.clone();
            changed["action"][field] = value;
            assert!(rejects(changed), "trusted wire action field {field}");
        }

        let mut changed = plan_json.clone();
        changed["action"]["required_authorization"]["byte_budget"] =
            serde_json::Value::from(42_000_001_u64);
        assert!(rejects(changed), "trusted wire authorization substitution");

        let response_mutations = [
            (
                "requested_topology",
                serde_json::json!({"topology": "quality_only"}),
            ),
            (
                "realized_topology",
                serde_json::json!({"topology": "fast_only"}),
            ),
            ("coverage_ppm", serde_json::json!(1)),
            ("admitted_semantic_scores", serde_json::json!(1)),
            (
                "degradation_reason_code",
                serde_json::json!("recovery.state.model_unloadable"),
            ),
        ];
        for (field, value) in response_mutations {
            let mut changed = plan_json.clone();
            changed["response_contract"][field] = value;
            assert!(rejects(changed), "trusted wire response field {field}");
        }

        let authorization = plan_json["action"]["required_authorization"].clone();
        for legacy_version in [
            "frankensearch.model_acquisition_authorization.v1",
            "frankensearch.model_acquisition_authorization.v2",
        ] {
            let mut changed = authorization.clone();
            changed["schema_version"] = serde_json::Value::String(legacy_version.to_owned());
            assert!(
                serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err(),
                "legacy authorization schema unexpectedly decoded: {legacy_version}"
            );
        }

        let mut changed = authorization.clone();
        changed["model_id"] = serde_json::Value::String(" ".to_owned());
        assert!(serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err());

        let mut changed = authorization.clone();
        changed["model_tier"] = serde_json::Value::String("Quality".to_owned());
        assert!(
            serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err(),
            "v4 recovery authorization must not inherit ModelTier's Rust casing"
        );

        for required_field in [
            "model_id",
            "model_tier",
            "embedding_space",
            "document_count",
            "estimated_reindex_duration_ms",
            "issued_at_unix_seconds",
            "expires_at_unix_seconds",
            "nonce",
        ] {
            let mut changed = authorization.clone();
            changed
                .as_object_mut()
                .expect("authorization JSON object")
                .remove(required_field);
            assert!(
                serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err(),
                "missing required consent field unexpectedly decoded: {required_field}"
            );
        }

        for (field, invalid_value) in [
            ("issued_at_unix_seconds", serde_json::json!("now")),
            ("expires_at_unix_seconds", serde_json::Value::Null),
            ("nonce", serde_json::json!(123_u64)),
        ] {
            let mut changed = authorization.clone();
            changed[field] = invalid_value;
            assert!(
                serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err(),
                "wrongly typed authorization field unexpectedly decoded: {field}"
            );
        }

        let mut changed = authorization.clone();
        changed["source"]["kind"] = serde_json::Value::String("ambient".to_owned());
        assert!(serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err());

        let mut changed = authorization.clone();
        changed["destination_class"] = serde_json::Value::String("unbounded_path".to_owned());
        assert!(serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err());

        let mut changed = authorization.clone();
        changed["byte_budget"] = serde_json::Value::from(0);
        assert!(serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err());

        let mut changed = authorization.clone();
        changed["embedding_space"]["dimension"] = serde_json::Value::from(0);
        assert!(serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err());

        let mut changed = authorization.clone();
        changed["embedding_space"]["logical_model_id"] =
            serde_json::Value::String("different-model".to_owned());
        assert!(serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err());

        for host in [
            "https://models.example.test",
            "user:secret@models.example.test",
            "models.example.test/path",
            "models.example.test?variant=quality",
            "models.example.test#fragment",
            "models example.test",
            "models.example.test:0",
            "models.example.test:+443",
            "2001:db8::1",
            "[not-ipv6]",
        ] {
            let mut changed = authorization.clone();
            changed["source"]["source_hosts"] = serde_json::json!([host]);
            assert!(
                serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err(),
                "serde unexpectedly accepted source host {host:?}"
            );
        }

        let mut changed = authorization;
        changed["manifest_fingerprint"] = serde_json::Value::String(" ".to_owned());
        assert!(serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err());

        let mut changed = plan_json.clone();
        changed["action"]["argv"] =
            serde_json::json!(["fsfs", "download-models", "--model", "different-model"]);
        assert!(rejects(changed), "argv must remain bound to trusted target");

        let mut changed = plan_json;
        changed["action"]["required_authorization"]["source"] = serde_json::json!({
            "kind": "local_bundle",
            "source_hosts": ["must-not-be-ignored.example.test"]
        });
        assert!(rejects(changed));
    }

    #[test]
    fn untrusted_wire_rejects_unknown_fields_at_every_nested_object_boundary() {
        let state = missing(ModelTier::Quality);
        let request = hybrid(RetrievalTopology::FullProgressive);
        let policy = permissive();
        let acquisition_target = target();
        let canonical = plan(state, request, policy, Some(&acquisition_target))
            .expect("canonical nested-wire fixture");
        let plan_json = serde_json::to_value(canonical).expect("serialize nested-wire fixture");
        let rejects_plan = |value| serde_json::from_value::<UntrustedRecoveryPlan>(value).is_err();

        let mut changed = plan_json.clone();
        changed["unexpected"] = serde_json::json!(true);
        assert!(rejects_plan(changed), "plan envelope");

        let mut changed = plan_json.clone();
        changed["state"]["unexpected"] = serde_json::json!(true);
        assert!(rejects_plan(changed), "readiness envelope");

        let mut changed = plan_json.clone();
        changed["state"]["detail"]["unexpected"] = serde_json::json!(true);
        assert!(rejects_plan(changed), "readiness detail");

        let mut changed = plan_json.clone();
        changed["requested_topology"]["unexpected"] = serde_json::json!(true);
        assert!(rejects_plan(changed), "requested topology");

        let mut changed = plan_json.clone();
        changed["policy"]["unexpected"] = serde_json::json!(true);
        assert!(rejects_plan(changed), "policy");

        let mut changed = plan_json.clone();
        changed["action"]["unexpected"] = serde_json::json!(true);
        assert!(rejects_plan(changed), "action");

        let mut changed = plan_json.clone();
        changed["action"]["required_authorization"]["unexpected"] = serde_json::json!(true);
        assert!(rejects_plan(changed), "acquisition authorization");

        let mut changed = plan_json.clone();
        changed["action"]["required_authorization"]["embedding_space"]["unexpected"] =
            serde_json::json!(true);
        assert!(rejects_plan(changed), "embedding-space identity");

        let mut changed = plan_json.clone();
        changed["action"]["required_authorization"]["embedding_space"]["artifacts"][0]["unexpected"] =
            serde_json::json!(true);
        assert!(rejects_plan(changed), "embedding artifact identity");

        let mut changed = plan_json.clone();
        changed["action"]["required_authorization"]["source"]["unexpected"] =
            serde_json::json!(true);
        assert!(rejects_plan(changed), "acquisition source");

        let mut changed = plan_json.clone();
        changed["response_contract"]["unexpected"] = serde_json::json!(true);
        assert!(rejects_plan(changed), "response contract");

        let mut changed = plan_json.clone();
        changed["response_contract"]["requested_topology"]["unexpected"] = serde_json::json!(true);
        assert!(rejects_plan(changed), "response requested topology");

        let mut changed = plan_json;
        changed["response_contract"]["realized_topology"]["unexpected"] = serde_json::json!(true);
        assert!(rejects_plan(changed), "response realized topology");

        for state_json in [
            serde_json::json!({
                "state": "ready",
                "detail": {"provenance": "local", "unexpected": true}
            }),
            serde_json::json!({
                "state": "model_missing",
                "detail": {"tier": "fast", "unexpected": true}
            }),
            serde_json::json!({
                "state": "model_unloadable",
                "detail": {"tier": "quality", "unexpected": true}
            }),
            serde_json::json!({
                "state": "partial_quality_coverage",
                "detail": {
                    "provenance": "daemon",
                    "coverage_ppm": 625_000,
                    "unexpected": true
                }
            }),
        ] {
            assert!(
                serde_json::from_value::<SemanticReadiness>(state_json).is_err(),
                "object-valued readiness detail discarded an unknown field"
            );
        }

        assert!(
            serde_json::from_value::<SemanticReadiness>(serde_json::json!({
                "state": "index_absent",
                "unexpected": true
            }))
            .is_err(),
            "unit readiness envelope discarded an unknown field"
        );

        assert!(
            serde_json::from_value::<SemanticReadiness>(serde_json::json!({
                "state": "model_missing",
                "detail": {"tier": "Quality"}
            }))
            .is_err(),
            "v3 readiness tier must use the recovery-local lowercase spelling"
        );

        assert!(
            serde_json::from_value::<RecoveryRequest>(serde_json::json!({
                "mode": "hybrid",
                "requested_topology": {
                    "topology": "partial_quality",
                    "coverage_ppm": 625_000,
                    "unexpected": true
                }
            }))
            .is_err(),
            "structured topology discarded an unknown field"
        );

        for invalid_topology in [
            serde_json::json!({"topology": "fast_only", "coverage_ppm": 625_000}),
            serde_json::json!({"topology": "partial_quality"}),
            serde_json::json!({"topology": "partial_quality", "coverage_ppm": null}),
        ] {
            assert!(
                serde_json::from_value::<RecoveryRequest>(serde_json::json!({
                    "mode": "hybrid",
                    "requested_topology": invalid_topology
                }))
                .is_err(),
                "topology-specific field presence was not enforced"
            );
        }
    }

    #[test]
    fn tagged_wire_envelopes_enforce_variant_specific_fields() {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct TopologyHarness {
            value: RetrievalTopology,
        }

        for expected in [
            RetrievalTopology::LexicalOnly,
            RetrievalTopology::HashControl,
            RetrievalTopology::FastOnly,
            RetrievalTopology::QualityOnly,
            RetrievalTopology::FullProgressive,
        ] {
            let topology_json = serde_json::to_value(expected).expect("serialize topology");
            let decoded: TopologyHarness =
                serde_json::from_value(serde_json::json!({"value": topology_json.clone()}))
                    .expect("decode exact unit topology");
            assert_eq!(decoded.value, expected);

            let mut unknown = topology_json.clone();
            unknown["unexpected"] = serde_json::json!(true);
            assert!(
                serde_json::from_value::<TopologyHarness>(serde_json::json!({"value": unknown}))
                    .is_err(),
                "unit topology accepted an unknown field: {expected:?}"
            );

            let mut forbidden = topology_json;
            forbidden["coverage_ppm"] = serde_json::json!(625_000);
            assert!(
                serde_json::from_value::<TopologyHarness>(serde_json::json!({"value": forbidden}))
                    .is_err(),
                "unit topology accepted partial-only coverage: {expected:?}"
            );
            for forbidden_value in [serde_json::Value::Null, serde_json::json!({})] {
                let mut forbidden = serde_json::to_value(expected).expect("serialize topology");
                forbidden["coverage_ppm"] = forbidden_value;
                assert!(
                    serde_json::from_value::<TopologyHarness>(
                        serde_json::json!({"value": forbidden})
                    )
                    .is_err(),
                    "unit topology accepted null or structured coverage: {expected:?}"
                );
            }
        }

        let partial = RetrievalTopology::PartialQuality {
            coverage_ppm: 625_000,
        };
        let partial_json = serde_json::to_value(partial).expect("serialize partial topology");
        let decoded: TopologyHarness =
            serde_json::from_value(serde_json::json!({"value": partial_json.clone()}))
                .expect("decode exact partial topology");
        assert_eq!(decoded.value, partial);
        for invalid in [
            serde_json::json!({"topology": "partial_quality"}),
            serde_json::json!({"topology": "partial_quality", "coverage_ppm": null}),
            serde_json::json!({
                "topology": "partial_quality",
                "coverage_ppm": 625_000,
                "unexpected": true
            }),
            serde_json::json!({}),
            serde_json::json!({"topology": null}),
        ] {
            assert!(
                serde_json::from_value::<TopologyHarness>(serde_json::json!({"value": invalid}))
                    .is_err(),
                "topology envelope accepted a missing, null, or forbidden field"
            );
        }

        let unit_states = [
            SemanticReadiness::IndexAbsent,
            SemanticReadiness::IdentityMismatch,
            SemanticReadiness::DaemonMismatch,
            SemanticReadiness::ManifestUnsafe,
            SemanticReadiness::AnnStale,
            SemanticReadiness::GenerationIncomplete,
            SemanticReadiness::RemoteUnverified,
            SemanticReadiness::HashControl,
        ];
        for expected in unit_states {
            let state_json = serde_json::to_value(&expected).expect("serialize unit readiness");
            assert_eq!(
                serde_json::from_value::<SemanticReadiness>(state_json.clone())
                    .expect("decode exact unit readiness"),
                expected
            );

            let mut unknown = state_json.clone();
            unknown["unexpected"] = serde_json::json!(true);
            assert!(
                serde_json::from_value::<SemanticReadiness>(unknown).is_err(),
                "unit readiness accepted an unknown field: {expected:?}"
            );

            for forbidden_detail in [
                serde_json::json!({}),
                serde_json::json!("caller_requested_zero_k"),
                serde_json::json!({"tier": "fast"}),
            ] {
                let mut forbidden = state_json.clone();
                forbidden["detail"] = forbidden_detail;
                assert!(
                    serde_json::from_value::<SemanticReadiness>(forbidden).is_err(),
                    "unit readiness accepted scalar or structured detail: {expected:?}"
                );
            }

            let mut null_detail = state_json;
            null_detail["detail"] = serde_json::Value::Null;
            assert!(
                serde_json::from_value::<SemanticReadiness>(null_detail).is_err(),
                "unit readiness accepted null detail: {expected:?}"
            );
        }

        for expected in [
            SemanticReadiness::Ready {
                provenance: VerifiedSemanticProvenance::Local,
            },
            missing(ModelTier::Fast),
            unloadable(ModelTier::Quality),
            SemanticReadiness::IndexEmpty(ZeroSignalReason::CallerRequestedZeroK),
            SemanticReadiness::PartialQualityCoverage {
                provenance: VerifiedSemanticProvenance::Daemon,
                coverage_ppm: 625_000,
            },
        ] {
            let state_json = serde_json::to_value(&expected).expect("serialize detailed readiness");
            assert_eq!(
                serde_json::from_value::<SemanticReadiness>(state_json.clone())
                    .expect("decode exact detailed readiness"),
                expected
            );

            let mut missing_detail = state_json.clone();
            missing_detail
                .as_object_mut()
                .expect("readiness object")
                .remove("detail");
            assert!(
                serde_json::from_value::<SemanticReadiness>(missing_detail).is_err(),
                "detailed readiness accepted missing detail: {expected:?}"
            );

            let mut null_detail = state_json.clone();
            null_detail["detail"] = serde_json::Value::Null;
            assert!(
                serde_json::from_value::<SemanticReadiness>(null_detail).is_err(),
                "detailed readiness accepted null detail: {expected:?}"
            );

            let mut forbidden_detail = state_json.clone();
            if let Some(detail) = forbidden_detail["detail"].as_object_mut() {
                detail.insert("unexpected".to_owned(), serde_json::json!(true));
            } else {
                forbidden_detail["detail"] = serde_json::json!({"unexpected": true});
            }
            assert!(
                serde_json::from_value::<SemanticReadiness>(forbidden_detail).is_err(),
                "detailed readiness discarded a forbidden detail field: {expected:?}"
            );

            let mut unknown = state_json;
            unknown["unexpected"] = serde_json::json!(true);
            assert!(
                serde_json::from_value::<SemanticReadiness>(unknown).is_err(),
                "detailed readiness accepted an unknown envelope field: {expected:?}"
            );
        }
        assert!(serde_json::from_value::<SemanticReadiness>(serde_json::json!({})).is_err());
        assert!(
            serde_json::from_value::<SemanticReadiness>(serde_json::json!({"state": null}))
                .is_err()
        );
        for cross_variant in [
            serde_json::json!({
                "state": "ready",
                "detail": {"tier": "fast"}
            }),
            serde_json::json!({
                "state": "model_missing",
                "detail": {"provenance": "local"}
            }),
            serde_json::json!({
                "state": "model_unloadable",
                "detail": {"provenance": "daemon", "coverage_ppm": 625_000}
            }),
            serde_json::json!({
                "state": "index_empty",
                "detail": {"provenance": "local", "coverage_ppm": 625_000}
            }),
            serde_json::json!({
                "state": "partial_quality_coverage",
                "detail": "caller_requested_zero_k"
            }),
        ] {
            assert!(
                serde_json::from_value::<SemanticReadiness>(cross_variant).is_err(),
                "readiness tag accepted another variant's otherwise-valid detail"
            );
        }

        let local_bundle = serde_json::to_value(ModelAcquisitionSource::LocalBundle)
            .expect("serialize local bundle");
        assert_eq!(
            serde_json::from_value::<ModelAcquisitionSource>(local_bundle.clone())
                .expect("decode exact local bundle"),
            ModelAcquisitionSource::LocalBundle
        );
        for invalid in [
            serde_json::json!({"kind": "local_bundle", "unexpected": true}),
            serde_json::json!({"kind": "local_bundle", "source_hosts": []}),
            serde_json::json!({
                "kind": "local_bundle",
                "source_hosts": ["models.example.test"]
            }),
            serde_json::json!({"kind": "local_bundle", "source_hosts": null}),
            serde_json::json!({}),
            serde_json::json!({"kind": null}),
        ] {
            assert!(
                serde_json::from_value::<ModelAcquisitionSource>(invalid).is_err(),
                "local-bundle source accepted a missing, null, or forbidden field"
            );
        }

        let network = ModelAcquisitionSource::Network {
            source_hosts: vec!["models.example.test".to_owned()],
        };
        let network_json = serde_json::to_value(&network).expect("serialize network source");
        assert_eq!(
            serde_json::from_value::<ModelAcquisitionSource>(network_json.clone())
                .expect("decode exact network source"),
            network
        );
        for invalid in [
            serde_json::json!({"kind": "network"}),
            serde_json::json!({"kind": "network", "source_hosts": null}),
            serde_json::json!({
                "kind": "network",
                "source_hosts": ["models.example.test"],
                "unexpected": true
            }),
        ] {
            assert!(
                serde_json::from_value::<ModelAcquisitionSource>(invalid).is_err(),
                "network source accepted a missing, null, or forbidden field"
            );
        }
    }

    #[test]
    fn recovery_v4_requires_every_canonical_null_field_to_be_present() {
        let ready = planned(
            SemanticReadiness::Ready {
                provenance: VerifiedSemanticProvenance::Local,
            },
            explicit(RetrievalTopology::FastOnly),
            permissive(),
        );
        let ready_json = serde_json::to_value(ready).expect("serialize ready plan");
        assert!(ready_json["policy"]["acquisition_authorization"].is_null());
        assert!(ready_json["action"].is_null());
        assert!(ready_json["response_contract"]["degradation_reason_code"].is_null());
        serde_json::from_value::<UntrustedRecoveryPlan>(ready_json.clone())
            .expect("explicit canonical nulls decode");

        let mut missing = ready_json.clone();
        missing["policy"]
            .as_object_mut()
            .expect("policy object")
            .remove("acquisition_authorization");
        assert!(
            serde_json::from_value::<UntrustedRecoveryPlan>(missing).is_err(),
            "missing policy.acquisition_authorization was treated as explicit null"
        );

        let mut missing = ready_json.clone();
        missing
            .as_object_mut()
            .expect("plan object")
            .remove("action");
        assert!(
            serde_json::from_value::<UntrustedRecoveryPlan>(missing).is_err(),
            "missing plan.action was treated as explicit null"
        );

        let mut missing = ready_json;
        missing["response_contract"]
            .as_object_mut()
            .expect("response object")
            .remove("degradation_reason_code");
        assert!(
            serde_json::from_value::<UntrustedRecoveryPlan>(missing).is_err(),
            "missing response.degradation_reason_code was treated as explicit null"
        );

        let unavailable = planned(
            SemanticReadiness::IndexAbsent,
            explicit(RetrievalTopology::FastOnly),
            permissive(),
        );
        let unavailable_json =
            serde_json::to_value(unavailable).expect("serialize unavailable plan");
        assert!(unavailable_json["action"]["required_authorization"].is_null());
        assert!(unavailable_json["response_contract"].is_null());
        serde_json::from_value::<UntrustedRecoveryPlan>(unavailable_json.clone())
            .expect("explicit action/response nulls decode");

        let mut missing = unavailable_json.clone();
        missing["action"]
            .as_object_mut()
            .expect("action object")
            .remove("required_authorization");
        assert!(
            serde_json::from_value::<UntrustedRecoveryPlan>(missing).is_err(),
            "missing action.required_authorization was treated as explicit null"
        );

        let mut missing = unavailable_json;
        missing
            .as_object_mut()
            .expect("plan object")
            .remove("response_contract");
        assert!(
            serde_json::from_value::<UntrustedRecoveryPlan>(missing).is_err(),
            "missing plan.response_contract was treated as explicit null"
        );
    }

    #[test]
    fn trusted_context_rejects_coherent_plan_and_target_substitution() {
        let state = missing(ModelTier::Quality);
        let request = hybrid(RetrievalTopology::FullProgressive);
        let policy = permissive();
        let acquisition_target = target();

        let ready_forgery = plan(
            SemanticReadiness::Ready {
                provenance: VerifiedSemanticProvenance::Local,
            },
            request,
            policy.clone(),
            Some(&acquisition_target),
        )
        .expect("internally coherent ready plan");
        assert!(
            decode_and_validate(
                serde_json::to_value(ready_forgery).expect("serialize forgery"),
                &state,
                request,
                &policy,
                Some(&acquisition_target),
            )
            .is_err(),
            "a coherent state, response, and retryability forgery must not replace probe state"
        );

        let request_forgery = plan(
            state.clone(),
            hybrid(RetrievalTopology::QualityOnly),
            policy.clone(),
            Some(&acquisition_target),
        )
        .expect("internally coherent alternate-request plan");
        assert!(
            decode_and_validate(
                serde_json::to_value(request_forgery).expect("serialize forgery"),
                &state,
                request,
                &policy,
                Some(&acquisition_target),
            )
            .is_err(),
            "a coherent topology and response forgery must not replace the caller request"
        );

        let offline_policy = RecoveryPolicy {
            interaction: InteractionPolicy::Interactive,
            network: NetworkPolicy::Offline,
            acquisition_authorization: None,
        };
        let policy_forgery = plan(
            state.clone(),
            request,
            offline_policy,
            Some(&acquisition_target),
        )
        .expect("internally coherent alternate-policy plan");
        assert!(
            decode_and_validate(
                serde_json::to_value(policy_forgery).expect("serialize forgery"),
                &state,
                request,
                &policy,
                Some(&acquisition_target),
            )
            .is_err(),
            "wire policy must not grant or remove execution capabilities"
        );

        let trusted_coverage = SemanticReadiness::PartialQualityCoverage {
            provenance: VerifiedSemanticProvenance::Local,
            coverage_ppm: 625_000,
        };
        let forged_coverage = SemanticReadiness::PartialQualityCoverage {
            provenance: VerifiedSemanticProvenance::Local,
            coverage_ppm: 700_000,
        };
        let coverage_forgery = plan(forged_coverage, request, policy.clone(), None)
            .expect("internally coherent alternate-coverage plan");
        assert!(
            decode_and_validate(
                serde_json::to_value(coverage_forgery).expect("serialize forgery"),
                &trusted_coverage,
                request,
                &policy,
                None,
            )
            .is_err(),
            "matching state and response coverage substitutions must not replace the census"
        );

        let mut alternate_target = target();
        alternate_target.model_id = "alternate-semantic-model".to_owned();
        alternate_target.embedding_space.logical_model_id = alternate_target.model_id.clone();
        alternate_target.upstream_revision = "alternate-immutable-revision".to_owned();
        alternate_target.embedding_space.immutable_revision =
            alternate_target.upstream_revision.clone();
        let target_forgery = plan(
            state.clone(),
            request,
            policy.clone(),
            Some(&alternate_target),
        )
        .expect("internally coherent alternate-target plan");
        assert!(
            decode_and_validate(
                serde_json::to_value(target_forgery).expect("serialize forgery"),
                &state,
                request,
                &policy,
                Some(&acquisition_target),
            )
            .is_err(),
            "a self-consistent action and authorization cannot substitute another frozen target"
        );
    }

    #[test]
    fn shell_rendering_quotes_unsafe_arguments() {
        let action = RecoveryAction {
            code: "recovery.action.build_index".to_owned(),
            explanation: String::new(),
            argv: vec![
                "fsfs".to_owned(),
                "index".to_owned(),
                "--index-dir".to_owned(),
                "/data/My Projects/it's here".to_owned(),
                ARG_SOURCE_DIR.to_owned(),
            ],
            network_required: false,
            consent_required: false,
            preserves_old_data: true,
            potentially_destructive: false,
            prerequisites: Vec::new(),
            expected_postcondition: "recovery.post.index_built".to_owned(),
            required_authorization: None,
        };
        assert_eq!(
            action.shell_command(),
            "fsfs index --index-dir '/data/My Projects/it'\\''s here' <source-dir>"
        );
    }

    #[test]
    fn serialization_roundtrips_and_locks_schema_version() {
        for state in representative_states() {
            if matches!(state, SemanticReadiness::HashControl) {
                continue;
            }
            for request in semantic_requests() {
                for policy in all_policies() {
                    let tier = tier_for_request(request);
                    let state = state_for_request(&state, request);
                    let acquisition_target = target_for(tier);
                    let original = plan(
                        state.clone(),
                        request,
                        policy.clone(),
                        Some(&acquisition_target),
                    )
                    .expect("valid representative plan");
                    let json = serde_json::to_value(&original).expect("serialize plan");
                    assert_eq!(
                        json["schema_version"],
                        serde_json::Value::String(RECOVERY_PLAN_SCHEMA_VERSION.to_owned())
                    );
                    let decoded = decode_and_validate(
                        json,
                        &state,
                        request,
                        &policy,
                        Some(&acquisition_target),
                    )
                    .expect("validate untrusted plan against authoritative context");
                    assert_eq!(decoded, original);
                }
            }
        }
        let hash = planned(SemanticReadiness::HashControl, hash_control(), permissive());
        let json = serde_json::to_value(&hash).expect("serialize hash plan");
        let decoded = decode_and_validate(
            json,
            &SemanticReadiness::HashControl,
            hash_control(),
            &permissive(),
            None,
        )
        .expect("validate hash plan");
        assert_eq!(decoded, hash);
    }

    #[test]
    fn golden_plan_json_for_model_missing_offline_hybrid() {
        let plan = planned(
            missing(ModelTier::Quality),
            hybrid(RetrievalTopology::FullProgressive),
            RecoveryPolicy {
                interaction: InteractionPolicy::NonInteractive,
                network: NetworkPolicy::Offline,
                acquisition_authorization: None,
            },
        );
        let json = serde_json::to_value(&plan).expect("serialize plan");
        assert_eq!(json["schema_version"], "frankensearch.recovery_plan.v4");
        assert_eq!(json["state"]["state"], "model_missing");
        assert_eq!(json["state"]["detail"]["tier"], "quality");
        assert_eq!(json["state_code"], "recovery.state.model_missing");
        assert_eq!(json["provenance"], "unavailable");
        assert_eq!(json["mode"], "hybrid");
        assert_eq!(json["requested_topology"]["topology"], "full_progressive");
        assert_eq!(json["policy"]["interaction"], "non_interactive");
        assert_eq!(json["policy"]["network"], "offline");
        assert!(json["policy"]["acquisition_authorization"].is_null());
        assert_eq!(json["semantic_available"], false);
        assert_eq!(json["retryability"], "blocked_by_capability");
        assert_eq!(json["action"]["code"], "recovery.action.acquire_model");
        assert_eq!(json["action"]["argv"], serde_json::json!([]));
        assert_eq!(json["action"]["network_required"], false);
        assert_eq!(json["action"]["consent_required"], true);
        assert_eq!(
            json["action"]["prerequisites"][0],
            "recovery.capability.import_model_bundle"
        );
        assert_eq!(
            json["action"]["prerequisites"][1],
            "recovery.policy.grant_consent"
        );
        assert_eq!(
            json["action"]["required_authorization"]["schema_version"],
            "frankensearch.model_acquisition_authorization.v3"
        );
        assert_eq!(
            json["action"]["required_authorization"]["model_id"],
            "fixture-semantic-model"
        );
        assert_eq!(
            json["action"]["required_authorization"]["model_tier"],
            "quality"
        );
        assert_eq!(
            json["action"]["required_authorization"]["embedding_space"]["logical_model_id"],
            "fixture-semantic-model"
        );
        assert_eq!(
            json["action"]["required_authorization"]["embedding_space"]["dimension"],
            384
        );
        assert_eq!(
            json["action"]["required_authorization"]["source"]["kind"],
            "local_bundle"
        );
        assert_eq!(
            json["action"]["required_authorization"]["byte_budget"],
            42_000_000
        );
        assert_eq!(
            json["action"]["required_authorization"]["document_count"],
            12_345
        );
        assert_eq!(
            json["action"]["required_authorization"]["estimated_reindex_duration_ms"],
            98_765
        );
        assert_eq!(
            json["action"]["required_authorization"]["issued_at_unix_seconds"],
            TEST_AUTHORIZATION_ISSUED_AT_UNIX_SECONDS
        );
        assert_eq!(
            json["action"]["required_authorization"]["expires_at_unix_seconds"],
            TEST_AUTHORIZATION_EXPIRES_AT_UNIX_SECONDS
        );
        assert_eq!(
            json["action"]["required_authorization"]["nonce"],
            TEST_AUTHORIZATION_NONCE
        );
        assert_eq!(
            json["response_contract"]["requested_topology"]["topology"],
            "full_progressive"
        );
        assert_eq!(
            json["response_contract"]["realized_topology"]["topology"],
            "lexical_only"
        );
        assert_eq!(json["response_contract"]["coverage_ppm"], 0);
        assert_eq!(json["response_contract"]["admitted_semantic_scores"], 0);
        assert_eq!(
            json["response_contract"]["degradation_reason_code"],
            "recovery.state.model_missing"
        );
    }

    #[test]
    fn golden_transition_table_is_stable() {
        // One compact row per unique state code and semantic mode under the
        // permissive policy, plus the explicit hash-control row.
        let states = vec![
            SemanticReadiness::Ready {
                provenance: VerifiedSemanticProvenance::Local,
            },
            missing(ModelTier::Quality),
            unloadable(ModelTier::Quality),
            SemanticReadiness::IndexAbsent,
            SemanticReadiness::IdentityMismatch,
            SemanticReadiness::DaemonMismatch,
            SemanticReadiness::IndexEmpty(ZeroSignalReason::NewlyCreatedEmpty),
            SemanticReadiness::ManifestUnsafe,
            SemanticReadiness::AnnStale,
            SemanticReadiness::GenerationIncomplete,
            SemanticReadiness::PartialQualityCoverage {
                provenance: VerifiedSemanticProvenance::Local,
                coverage_ppm: 750_000,
            },
            SemanticReadiness::RemoteUnverified,
        ];
        let mut rows = Vec::new();
        for state in &states {
            for request in [
                explicit(RetrievalTopology::FullProgressive),
                hybrid(RetrievalTopology::FullProgressive),
            ] {
                let plan = planned(state.clone(), request, permissive());
                let mode_tag = match request.mode {
                    RequestMode::ExplicitSemantic => "semantic",
                    RequestMode::Hybrid => "hybrid",
                    RequestMode::HashControl => unreachable!("semantic request array"),
                };
                let action_tag = plan.action.as_ref().map_or_else(
                    || "none,false,false".to_owned(),
                    |a| format!("{},{},{}", a.code, a.network_required, a.consent_required),
                );
                let retry_tag = match plan.retryability {
                    Retryability::NotNeeded => "not_needed",
                    Retryability::AfterAction => "after_action",
                    Retryability::AfterRequestChange => "after_request_change",
                    Retryability::BlockedByPolicy => "blocked",
                    Retryability::BlockedByCapability => "capability_blocked",
                };
                let (realized, coverage) = plan
                    .response_contract
                    .as_ref()
                    .map_or(("none", 0), |response| {
                        (response.realized_topology().code(), response.coverage_ppm())
                    });
                rows.push(format!(
                    "{}|{} => {},{},{},{}",
                    plan.state_code, mode_tag, action_tag, retry_tag, realized, coverage
                ));
            }
        }
        let hash = planned(SemanticReadiness::HashControl, hash_control(), permissive());
        let hash_response = hash.response_contract.expect("hash response");
        rows.push(format!(
            "{}|hash_control => none,false,false,not_needed,{},{}",
            hash.state_code,
            hash_response.realized_topology().code(),
            hash_response.coverage_ppm()
        ));
        let expected: Vec<&str> = vec![
            "recovery.state.ready|semantic => none,false,false,not_needed,full_progressive,1000000",
            "recovery.state.ready|hybrid => none,false,false,not_needed,full_progressive,1000000",
            "recovery.state.model_missing|semantic => recovery.action.acquire_model,true,true,capability_blocked,none,0",
            "recovery.state.model_missing|hybrid => recovery.action.acquire_model,true,true,capability_blocked,lexical_only,0",
            "recovery.state.model_unloadable|semantic => recovery.action.reacquire_model,true,true,capability_blocked,none,0",
            "recovery.state.model_unloadable|hybrid => recovery.action.reacquire_model,true,true,capability_blocked,lexical_only,0",
            "recovery.state.index_absent|semantic => recovery.action.build_index,false,false,capability_blocked,none,0",
            "recovery.state.index_absent|hybrid => recovery.action.build_index,false,false,capability_blocked,lexical_only,0",
            "recovery.state.identity_mismatch|semantic => recovery.action.reindex_full,false,true,capability_blocked,none,0",
            "recovery.state.identity_mismatch|hybrid => recovery.action.reindex_full,false,true,capability_blocked,lexical_only,0",
            "recovery.state.daemon_mismatch|semantic => recovery.action.restart_daemon,false,false,capability_blocked,none,0",
            "recovery.state.daemon_mismatch|hybrid => recovery.action.restart_daemon,false,false,capability_blocked,lexical_only,0",
            "recovery.state.index_empty|semantic => recovery.action.ingest_content,false,false,capability_blocked,none,0",
            "recovery.state.index_empty|hybrid => recovery.action.ingest_content,false,false,capability_blocked,lexical_only,0",
            "recovery.state.manifest_unsafe|semantic => recovery.action.reindex_full,false,true,capability_blocked,none,0",
            "recovery.state.manifest_unsafe|hybrid => recovery.action.reindex_full,false,true,capability_blocked,lexical_only,0",
            "recovery.state.ann_stale|semantic => recovery.action.rebuild_ann,false,false,capability_blocked,none,0",
            "recovery.state.ann_stale|hybrid => recovery.action.rebuild_ann,false,false,capability_blocked,lexical_only,0",
            "recovery.state.generation_incomplete|semantic => recovery.action.resume_index,false,false,capability_blocked,none,0",
            "recovery.state.generation_incomplete|hybrid => recovery.action.resume_index,false,false,capability_blocked,lexical_only,0",
            "recovery.state.partial_quality_coverage|semantic => recovery.action.backfill_quality,false,false,capability_blocked,partial_quality,750000",
            "recovery.state.partial_quality_coverage|hybrid => recovery.action.backfill_quality,false,false,capability_blocked,partial_quality,750000",
            "recovery.state.remote_unverified|semantic => recovery.action.provide_attestation,false,false,blocked,none,0",
            "recovery.state.remote_unverified|hybrid => recovery.action.provide_attestation,false,false,blocked,lexical_only,0",
            "recovery.state.hash_control|hash_control => none,false,false,not_needed,hash_control,0",
        ];
        assert_eq!(
            rows, expected,
            "transition table changed: contract review required"
        );
    }
}
