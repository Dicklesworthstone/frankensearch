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
//! - Explicit semantic requests fail closed: `semantic_available` is `false`
//!   for every non-ready state. Hybrid requests may proceed lexical-only,
//!   but only with a [`SemanticResponseContract`] that names the requested
//!   and realized topologies, reports zero coverage and admits zero semantic
//!   scores.
//! - The planner is pure and exhaustive: every enumerated state is matched
//!   without wildcards, so a new readiness state fails compilation until it
//!   is planned for.
//! - Model acquisition is always scoped to one frozen manifest, revision,
//!   license assertion, source, byte budget, and path-free destination
//!   identity. An authorization for any other scope is not interchangeable.
//!
//! # Why schema v2
//!
//! The original v1 foundation represented offline recovery as a blocked
//! network download, represented request mode without retrieval topology,
//! and had no producer provenance, response-admission contract, or scoped
//! acquisition authorization. Correcting those facts changes required wire
//! fields and reverses the meaning of the offline transition, so decoding the
//! new contract as v1 would be unsafe. V2 deliberately fails closed on v1
//! payloads instead of installing a compatibility shim. Every v1 stable
//! state/action/postcondition/policy code remains unchanged; v2 only appends
//! new codes.
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

use crate::{config::ZeroSignalReason, types::RetrievalTopology};

/// Schema version for serialized [`RecoveryPlan`] payloads.
pub const RECOVERY_PLAN_SCHEMA_VERSION: &str = "frankensearch.recovery_plan.v2";

/// Schema version for a scoped [`ModelAcquisitionAuthorization`].
pub const MODEL_ACQUISITION_AUTHORIZATION_SCHEMA_VERSION: &str =
    "frankensearch.model_acquisition_authorization.v1";

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

/// Placeholder token for an operator-supplied, complete local model bundle.
///
/// Unlike path-bearing arguments, this token intentionally has no shell
/// punctuation so machine clients can replace it without confusing it with
/// a path that the library observed.
pub const ARG_MODEL_BUNDLE: &str = "ARG_MODEL_BUNDLE";

/// Wire discriminator for [`RecoveryPlan`].
///
/// Using a closed enum rather than an arbitrary string makes serde reject
/// v1 or unknown schemas before a caller can accidentally execute their
/// actions with v2 semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryPlanSchemaVersion {
    #[serde(rename = "frankensearch.recovery_plan.v2")]
    V2,
}

/// Wire discriminator for [`ModelAcquisitionAuthorization`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelAcquisitionAuthorizationSchemaVersion {
    #[serde(rename = "frankensearch.model_acquisition_authorization.v1")]
    V1,
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "state", content = "detail")]
pub enum SemanticReadiness {
    /// The semantic lane is fully usable: verified model, loadable, and a
    /// compatible index with usable live vectors.
    Ready {
        provenance: VerifiedSemanticProvenance,
    },
    /// No usable model artifact exists in the configured cache.
    ModelMissing,
    /// A model artifact exists but failed verification or load self-test.
    ModelUnloadable,
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

impl SemanticReadiness {
    /// Stable three-segment state code.
    #[must_use]
    pub const fn state_code(&self) -> &'static str {
        match self {
            Self::Ready { .. } => "recovery.state.ready",
            Self::ModelMissing => "recovery.state.model_missing",
            Self::ModelUnloadable => "recovery.state.model_unloadable",
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
            Self::ModelMissing
            | Self::ModelUnloadable
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryPolicy {
    pub interaction: InteractionPolicy,
    pub network: NetworkPolicy,
    /// Exact non-TTY/programmatic model-acquisition authorization, when
    /// already granted. It satisfies consent only when byte-for-byte equal
    /// to the action's required authorization.
    pub acquisition_authorization: Option<ModelAcquisitionAuthorization>,
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case", tag = "kind")]
pub enum ModelAcquisitionSource {
    /// Immutable HTTPS sources named by credential-free host.
    Network { source_hosts: Vec<String> },
    /// Complete operator-supplied artifact tree.
    LocalBundle,
}

/// Exact, path-free authorization required before model bytes are acquired.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelAcquisitionAuthorization {
    pub schema_version: ModelAcquisitionAuthorizationSchemaVersion,
    pub manifest_fingerprint: String,
    pub upstream_revision: String,
    pub license_spdx: String,
    pub source: ModelAcquisitionSource,
    pub byte_budget: u64,
    pub destination_class: ModelDestinationClass,
    /// Bounded hash of the canonical destination, never the raw path.
    pub destination_fingerprint: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelAcquisitionAuthorizationWire {
    schema_version: ModelAcquisitionAuthorizationSchemaVersion,
    manifest_fingerprint: String,
    upstream_revision: String,
    license_spdx: String,
    source: ModelAcquisitionSource,
    byte_budget: u64,
    destination_class: ModelDestinationClass,
    destination_fingerprint: String,
}

impl<'de> Deserialize<'de> for ModelAcquisitionAuthorization {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = ModelAcquisitionAuthorizationWire::deserialize(deserializer)?;
        let authorization = Self {
            schema_version: wire.schema_version,
            manifest_fingerprint: wire.manifest_fingerprint,
            upstream_revision: wire.upstream_revision,
            license_spdx: wire.license_spdx,
            source: wire.source,
            byte_budget: wire.byte_budget,
            destination_class: wire.destination_class,
            destination_fingerprint: wire.destination_fingerprint,
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
        validate_scope_text("manifest_fingerprint", &self.manifest_fingerprint)?;
        validate_scope_text("upstream_revision", &self.upstream_revision)?;
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
        Ok(())
    }
}

/// Source-independent target from which the planner derives the exact
/// network or local-bundle authorization required by policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelAcquisitionTarget {
    pub manifest_fingerprint: String,
    pub upstream_revision: String,
    pub license_spdx: String,
    pub network_source_hosts: Vec<String>,
    pub byte_budget: u64,
    pub destination_class: ModelDestinationClass,
    pub destination_fingerprint: String,
}

impl ModelAcquisitionTarget {
    fn authorization_for(
        &self,
        network: NetworkPolicy,
    ) -> Result<ModelAcquisitionAuthorization, RecoveryContractError> {
        let authorization = ModelAcquisitionAuthorization {
            schema_version: ModelAcquisitionAuthorizationSchemaVersion::V1,
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
        };
        authorization.validate()?;
        Ok(authorization)
    }
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
    #[error("model acquisition byte budget must be non-zero")]
    ZeroAcquisitionByteBudget,
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
    /// The recommended action cannot run under the current policy; the
    /// listed prerequisites must be granted first.
    BlockedByPolicy,
}

/// One truthful next action.
///
/// The four booleans are independent schema-mandated facts about the
/// action (bd-vmv7's field list), not an encoded state machine, so a
/// bitflag or enum representation would obscure the serialized contract.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryAction {
    /// Stable action code, append-only within a schema version.
    pub code: String,
    /// Human-readable explanation of what the action does and why.
    pub explanation: String,
    /// Command as an argv array. Placeholder tokens ([`ARG_INDEX_DIR`],
    /// [`ARG_SOURCE_DIR`]) are substituted by integrators; the pure
    /// planner never handles real paths.
    pub argv: Vec<String>,
    /// The action needs network access (model downloads).
    pub network_required: bool,
    /// The action needs explicit human consent (it replaces existing
    /// artifacts).
    pub consent_required: bool,
    /// Existing user data (documents, index contents) survives the action.
    pub preserves_old_data: bool,
    /// The action replaces or rewrites existing artifacts.
    pub potentially_destructive: bool,
    /// Stable codes of conditions that must hold before the action can
    /// run (policy grants).
    pub prerequisites: Vec<String>,
    /// Stable code of the state expected after the action succeeds. Never
    /// `recovery.state.ready` for acquisition actions: readiness
    /// additionally requires the load self-test and a compatible
    /// non-empty index.
    pub expected_postcondition: String,
    /// Exact acquisition authorization this action requires. `None` for
    /// non-acquisition actions and when the caller failed to bind a frozen
    /// model target (which blocks the plan through a prerequisite).
    pub required_authorization: Option<ModelAcquisitionAuthorization>,
}

impl RecoveryAction {
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
/// are permanently constrained to zero.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SemanticResponseContract {
    pub requested_topology: RetrievalTopology,
    pub realized_topology: RetrievalTopology,
    pub coverage_ppm: u32,
    pub admitted_semantic_scores: u64,
    /// Present exactly for a lexical-only hybrid degradation.
    pub degradation_reason_code: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SemanticResponseContractWire {
    requested_topology: RetrievalTopology,
    realized_topology: RetrievalTopology,
    coverage_ppm: u32,
    admitted_semantic_scores: u64,
    degradation_reason_code: Option<String>,
}

impl<'de> Deserialize<'de> for SemanticResponseContract {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = SemanticResponseContractWire::deserialize(deserializer)?;
        Self::new(
            wire.requested_topology,
            wire.realized_topology,
            wire.coverage_ppm,
            wire.admitted_semantic_scores,
            wire.degradation_reason_code,
        )
        .map_err(serde::de::Error::custom)
    }
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
        let topology_compatible = match self.requested_topology {
            RetrievalTopology::FastOnly => matches!(
                self.realized_topology,
                RetrievalTopology::FastOnly | RetrievalTopology::LexicalOnly
            ),
            RetrievalTopology::QualityOnly => matches!(
                self.realized_topology,
                RetrievalTopology::QualityOnly
                    | RetrievalTopology::PartialQuality { .. }
                    | RetrievalTopology::LexicalOnly
            ),
            RetrievalTopology::FullProgressive => matches!(
                self.realized_topology,
                RetrievalTopology::FullProgressive
                    | RetrievalTopology::PartialQuality { .. }
                    | RetrievalTopology::LexicalOnly
            ),
            RetrievalTopology::HashControl => {
                matches!(self.realized_topology, RetrievalTopology::HashControl)
            }
            RetrievalTopology::LexicalOnly | RetrievalTopology::PartialQuality { .. } => false,
        };
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
            self.realized_topology,
            self.degradation_reason_code.as_deref(),
        ) {
            (RetrievalTopology::LexicalOnly, Some(code)) if !code.trim().is_empty() => Ok(()),
            (RetrievalTopology::LexicalOnly, _) => {
                Err(RecoveryContractError::MissingDegradationReason)
            }
            (_, None) => Ok(()),
            (_, Some(_)) => Err(RecoveryContractError::UnexpectedDegradationReason),
        }
    }
}

/// The full plan: current state, verdict for the requested mode, and the
/// truthful next action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RecoveryPlan {
    /// [`RECOVERY_PLAN_SCHEMA_VERSION`].
    pub schema_version: RecoveryPlanSchemaVersion,
    /// The readiness state the plan was computed from.
    pub state: SemanticReadiness,
    /// Stable code for `state` (denormalized for consumers that do not
    /// decode the enum).
    pub state_code: String,
    /// Producer trust classification derived from `state`.
    pub provenance: SemanticProvenance,
    /// The mode the caller requested.
    pub mode: RequestMode,
    /// Exact retrieval topology the caller requested.
    pub requested_topology: RetrievalTopology,
    /// The policy the plan was computed under.
    pub policy: RecoveryPolicy,
    /// Whether semantic results can be served right now.
    pub semantic_available: bool,
    /// Whether retrying can succeed, and under what condition.
    pub retryability: Retryability,
    /// The truthful next action; `None` when the state needs none (ready,
    /// or the emptiness was request-scoped).
    pub action: Option<RecoveryAction>,
    /// Response shape allowed by this decision. `None` means the explicit
    /// request fails closed and no response may be emitted.
    pub response_contract: Option<SemanticResponseContract>,
}

#[derive(Deserialize)]
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
    action: Option<RecoveryAction>,
    response_contract: Option<SemanticResponseContract>,
}

impl<'de> Deserialize<'de> for RecoveryPlan {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = RecoveryPlanWire::deserialize(deserializer)?;
        let plan = Self {
            schema_version: wire.schema_version,
            state: wire.state,
            state_code: wire.state_code,
            provenance: wire.provenance,
            mode: wire.mode,
            requested_topology: wire.requested_topology,
            policy: wire.policy,
            semantic_available: wire.semantic_available,
            retryability: wire.retryability,
            action: wire.action,
            response_contract: wire.response_contract,
        };
        plan.validate().map_err(serde::de::Error::custom)?;
        Ok(plan)
    }
}

impl RecoveryPlan {
    /// Validate denormalized fields and cross-field response/action invariants.
    ///
    /// This is also applied during deserialization, so changing a state code,
    /// provenance, topology, response admission, or retry verdict in JSON
    /// cannot manufacture a different decision than the typed state allows.
    ///
    /// # Errors
    ///
    /// Returns [`RecoveryContractError::InconsistentRecoveryPlan`] when a
    /// denormalized field conflicts with the typed state, request, action, or
    /// response contract. Nested request, readiness, response, and
    /// authorization validation errors are preserved.
    pub fn validate(&self) -> Result<(), RecoveryContractError> {
        RecoveryRequest {
            mode: self.mode,
            requested_topology: self.requested_topology,
        }
        .validate()?;
        validate_hash_mode_state(self.mode, &self.state)?;
        validate_readiness(&self.state)?;

        if self.state_code != self.state.state_code() {
            return Err(RecoveryContractError::InconsistentRecoveryPlan {
                field: "state_code",
            });
        }
        if self.provenance != self.state.provenance() {
            return Err(RecoveryContractError::InconsistentRecoveryPlan {
                field: "provenance",
            });
        }
        if self.semantic_available != self.state.semantic_available() {
            return Err(RecoveryContractError::InconsistentRecoveryPlan {
                field: "semantic_available",
            });
        }
        self.validate_response_contract()?;
        self.validate_action_and_retryability()
    }

    fn validate_response_contract(&self) -> Result<(), RecoveryContractError> {
        let response = self.response_contract.as_ref();
        if response.is_some_and(|contract| contract.requested_topology != self.requested_topology) {
            return Err(RecoveryContractError::InconsistentRecoveryPlan {
                field: "response_contract.requested_topology",
            });
        }

        let response_consistent = match (self.mode, self.semantic_available, response) {
            (RequestMode::ExplicitSemantic | RequestMode::Hybrid, true, Some(contract)) => {
                contract.realized_topology.is_semantic()
                    && contract.degradation_reason_code.is_none()
            }
            (RequestMode::ExplicitSemantic, false, None) => true,
            (RequestMode::Hybrid, false, Some(contract)) => {
                contract.realized_topology == RetrievalTopology::LexicalOnly
                    && contract.coverage_ppm == 0
                    && contract.admitted_semantic_scores == 0
                    && contract.degradation_reason_code.as_deref() == Some(self.state_code.as_str())
            }
            (RequestMode::HashControl, false, Some(contract)) => {
                contract.requested_topology == RetrievalTopology::HashControl
                    && contract.realized_topology == RetrievalTopology::HashControl
                    && contract.coverage_ppm == 0
                    && contract.admitted_semantic_scores == 0
                    && contract.degradation_reason_code.is_none()
            }
            _ => false,
        };
        if response_consistent {
            Ok(())
        } else {
            Err(RecoveryContractError::InconsistentRecoveryPlan {
                field: "response_contract",
            })
        }
    }

    fn validate_action_and_retryability(&self) -> Result<(), RecoveryContractError> {
        let expected_action_code = expected_action_code(&self.state);
        if self.action.as_ref().map(|action| action.code.as_str()) != expected_action_code {
            return Err(RecoveryContractError::InconsistentRecoveryPlan {
                field: "action.code",
            });
        }

        let expected_retryability = self.action.as_ref().map_or_else(
            || {
                if self.semantic_available || matches!(self.mode, RequestMode::HashControl) {
                    Retryability::NotNeeded
                } else {
                    Retryability::AfterAction
                }
            },
            |action| {
                if action.prerequisites.is_empty() {
                    Retryability::AfterAction
                } else {
                    Retryability::BlockedByPolicy
                }
            },
        );
        if self.retryability != expected_retryability {
            return Err(RecoveryContractError::InconsistentRecoveryPlan {
                field: "retryability",
            });
        }

        if let Some(action) = &self.action {
            if let Some(authorization) = &action.required_authorization {
                authorization.validate()?;
            }
            let acquisition = matches!(
                action.code.as_str(),
                "recovery.action.acquire_model" | "recovery.action.reacquire_model"
            );
            if acquisition != action.required_authorization.is_some()
                && !(acquisition
                    && action
                        .prerequisites
                        .iter()
                        .any(|code| code == "recovery.policy.bind_model"))
            {
                return Err(RecoveryContractError::InconsistentRecoveryPlan {
                    field: "action.required_authorization",
                });
            }
            if let Some(authorization) = &action.required_authorization {
                let source_matches_policy = matches!(
                    (
                        &authorization.source,
                        self.policy.network,
                        action.network_required,
                    ),
                    (
                        ModelAcquisitionSource::Network { .. },
                        NetworkPolicy::Allowed,
                        true,
                    ) | (
                        ModelAcquisitionSource::LocalBundle,
                        NetworkPolicy::Offline,
                        false,
                    )
                );
                if !source_matches_policy {
                    return Err(RecoveryContractError::InconsistentRecoveryPlan {
                        field: "action.required_authorization.source",
                    });
                }
            }

            let has_binding_prerequisite = action
                .prerequisites
                .iter()
                .any(|code| code == "recovery.policy.bind_model");
            if has_binding_prerequisite != (acquisition && action.required_authorization.is_none())
            {
                return Err(RecoveryContractError::InconsistentRecoveryPlan {
                    field: "action.prerequisites.bind_model",
                });
            }
            let authorization_satisfied =
                action
                    .required_authorization
                    .as_ref()
                    .is_some_and(|required| {
                        self.policy.acquisition_authorization.as_ref() == Some(required)
                    });
            let has_consent_prerequisite = action
                .prerequisites
                .iter()
                .any(|code| code == "recovery.policy.grant_consent");
            let consent_blocked = action.consent_required
                && matches!(self.policy.interaction, InteractionPolicy::NonInteractive)
                && !authorization_satisfied;
            if has_consent_prerequisite != consent_blocked {
                return Err(RecoveryContractError::InconsistentRecoveryPlan {
                    field: "action.prerequisites.grant_consent",
                });
            }
        }
        Ok(())
    }
}

fn expected_action_code(state: &SemanticReadiness) -> Option<&'static str> {
    match state {
        SemanticReadiness::Ready { .. } | SemanticReadiness::HashControl => None,
        SemanticReadiness::ModelMissing => Some("recovery.action.acquire_model"),
        SemanticReadiness::ModelUnloadable => Some("recovery.action.reacquire_model"),
        SemanticReadiness::IndexAbsent => Some("recovery.action.build_index"),
        SemanticReadiness::IdentityMismatch
        | SemanticReadiness::ManifestUnsafe
        | SemanticReadiness::IndexEmpty(ZeroSignalReason::NoUsableVectors) => {
            Some("recovery.action.reindex_full")
        }
        SemanticReadiness::DaemonMismatch => Some("recovery.action.restart_daemon"),
        SemanticReadiness::IndexEmpty(
            ZeroSignalReason::NewlyCreatedEmpty
            | ZeroSignalReason::AllTombstoned
            | ZeroSignalReason::WalOnlyNoLiveRecords,
        ) => Some("recovery.action.ingest_content"),
        SemanticReadiness::IndexEmpty(ZeroSignalReason::AnnReturnedEmptyDespiteUsableVectors)
        | SemanticReadiness::AnnStale => Some("recovery.action.rebuild_ann"),
        SemanticReadiness::IndexEmpty(
            ZeroSignalReason::CallerRequestedZeroK
            | ZeroSignalReason::FilterEliminatedAll
            | ZeroSignalReason::NonFiniteQuery
            | ZeroSignalReason::ZeroNormQuery,
        ) => None,
        SemanticReadiness::GenerationIncomplete => Some("recovery.action.resume_index"),
        SemanticReadiness::PartialQualityCoverage { .. } => {
            Some("recovery.action.backfill_quality")
        }
        SemanticReadiness::RemoteUnverified => Some("recovery.action.provide_attestation"),
    }
}

/// Compute the truthful plan for a readiness state under a request and
/// policy. Pure and deterministic: identical inputs yield identical plans.
///
/// # Errors
///
/// Rejects ambiguous request topology, invalid partial coverage, hash
/// requests without hash-control readiness, and malformed acquisition
/// targets before returning executable recovery metadata.
pub fn plan(
    state: SemanticReadiness,
    request: RecoveryRequest,
    policy: RecoveryPolicy,
    acquisition_target: Option<&ModelAcquisitionTarget>,
) -> Result<RecoveryPlan, RecoveryContractError> {
    let request = request.validate()?;
    validate_hash_mode_state(request.mode, &state)?;
    validate_readiness(&state)?;

    let action = action_for(&state, policy.network, acquisition_target)?;
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
                // Request-scoped emptiness: adjusting the request is on the
                // caller, not the system.
                Retryability::AfterAction
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
            let retryability = if network_blocked
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
        schema_version: RecoveryPlanSchemaVersion::V2,
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

/// The exhaustive state → action table. No wildcard arm: adding a
/// readiness state without planning for it is a compile error.
fn action_for(
    state: &SemanticReadiness,
    network: NetworkPolicy,
    acquisition_target: Option<&ModelAcquisitionTarget>,
) -> Result<Option<RecoveryAction>, RecoveryContractError> {
    match state {
        SemanticReadiness::Ready { .. } | SemanticReadiness::HashControl => Ok(None),
        SemanticReadiness::ModelMissing => {
            model_acquisition_action(false, network, acquisition_target).map(Some)
        }
        SemanticReadiness::ModelUnloadable => {
            model_acquisition_action(true, network, acquisition_target).map(Some)
        }
        SemanticReadiness::IndexAbsent => Ok(Some(simple_action(
            "recovery.action.build_index",
            "A verified model is present but no vector index exists; build one from the \
             corpus.",
            &[
                "fsfs",
                "index",
                "--index-dir",
                ARG_INDEX_DIR,
                ARG_SOURCE_DIR,
            ],
            false,
            false,
            true,
            false,
            "recovery.post.index_built",
        ))),
        SemanticReadiness::IdentityMismatch => Ok(Some(simple_action(
            "recovery.action.reindex_full",
            "The index was built in a different embedding space than the configured model. \
             Rebuild in place only if the identity change is intentional; the existing index \
             is replaced.",
            &[
                "fsfs",
                "index",
                "--full",
                "--index-dir",
                ARG_INDEX_DIR,
                ARG_SOURCE_DIR,
            ],
            false,
            true,
            false,
            true,
            "recovery.post.index_rebuilt",
        ))),
        SemanticReadiness::DaemonMismatch => Ok(Some(simple_action(
            "recovery.action.restart_daemon",
            "The embedding daemon serves a different space than the local configuration; \
             restart it so both sides agree before trusting daemon vectors.",
            &["fsfs", "daemon", "restart"],
            false,
            false,
            true,
            false,
            "recovery.post.daemon_aligned",
        ))),
        SemanticReadiness::IndexEmpty(reason) => Ok(plan_for_empty(*reason)),
        SemanticReadiness::ManifestUnsafe => Ok(Some(simple_action(
            "recovery.action.reindex_full",
            "The manifest failed safety validation and its artifacts must not be trusted; \
             rebuild index artifacts in place from source content.",
            &[
                "fsfs",
                "index",
                "--full",
                "--index-dir",
                ARG_INDEX_DIR,
                ARG_SOURCE_DIR,
            ],
            false,
            true,
            false,
            true,
            "recovery.post.index_rebuilt",
        ))),
        SemanticReadiness::AnnStale => Ok(Some(simple_action(
            "recovery.action.rebuild_ann",
            "The ANN sidecar belongs to an older index generation; rebuild it. Exact search \
             remains correct meanwhile and the vector index is untouched.",
            &[
                "fsfs",
                "index",
                "--index-dir",
                ARG_INDEX_DIR,
                ARG_SOURCE_DIR,
            ],
            false,
            false,
            true,
            false,
            "recovery.post.ann_rebuilt",
        ))),
        SemanticReadiness::GenerationIncomplete => Ok(Some(simple_action(
            "recovery.action.resume_index",
            "An index generation was interrupted before publication; re-run indexing to \
             complete it. Published data is untouched.",
            &[
                "fsfs",
                "index",
                "--index-dir",
                ARG_INDEX_DIR,
                ARG_SOURCE_DIR,
            ],
            false,
            false,
            true,
            false,
            "recovery.post.generation_completed",
        ))),
        SemanticReadiness::PartialQualityCoverage { .. } => Ok(Some(simple_action(
            "recovery.action.backfill_quality",
            "Some records lack quality-tier embeddings, so refinement coverage is partial; \
             re-run indexing to backfill them. Search remains available meanwhile.",
            &[
                "fsfs",
                "index",
                "--index-dir",
                ARG_INDEX_DIR,
                ARG_SOURCE_DIR,
            ],
            false,
            false,
            true,
            false,
            "recovery.post.coverage_completed",
        ))),
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
    let argv = match (network, reacquire) {
        (NetworkPolicy::Allowed, false) => {
            vec!["fsfs", "download-models", "--verify"]
        }
        (NetworkPolicy::Allowed, true) => {
            vec!["fsfs", "download-models", "--verify", "--force"]
        }
        (NetworkPolicy::Offline, false) => vec![
            "fsfs",
            "download-models",
            "--from-file",
            ARG_MODEL_BUNDLE,
            "--verify",
        ],
        (NetworkPolicy::Offline, true) => vec![
            "fsfs",
            "download-models",
            "--from-file",
            ARG_MODEL_BUNDLE,
            "--verify",
            "--force",
        ],
    };
    let explanation = match (network, reacquire) {
        (NetworkPolicy::Allowed, false) => {
            "Download and verify the configured semantic model. Acquisition alone does not make \
             the lane searchable: readiness additionally requires the load self-test and a \
             compatible non-empty index."
        }
        (NetworkPolicy::Allowed, true) => {
            "The cached model failed verification or load self-test; explicitly re-download and \
             verify the same frozen revision. Index data is untouched."
        }
        (NetworkPolicy::Offline, false) => {
            "Import and verify the configured semantic model from a complete local bundle. No \
             network access is permitted or required. Acquisition alone does not make the lane \
             searchable."
        }
        (NetworkPolicy::Offline, true) => {
            "Replace the unloadable cache generation from a complete local bundle and verify the \
             same frozen revision. No network access is permitted or required; index data is \
             untouched."
        }
    };
    let mut action = simple_action(
        code,
        explanation,
        &argv,
        matches!(network, NetworkPolicy::Allowed),
        true,
        true,
        false,
        "recovery.post.model_acquired_unverified",
    );
    action.required_authorization = acquisition_target
        .map(|target| target.authorization_for(network))
        .transpose()?;
    Ok(action)
}

/// Empty-index planning follows the zero-signal classification: benign
/// state emptiness wants ingestion, availability failures want rebuilds,
/// the ANN anomaly wants a sidecar rebuild, and request-scoped reasons
/// need no system action at all.
fn plan_for_empty(reason: ZeroSignalReason) -> Option<RecoveryAction> {
    match reason {
        ZeroSignalReason::NewlyCreatedEmpty
        | ZeroSignalReason::AllTombstoned
        | ZeroSignalReason::WalOnlyNoLiveRecords => Some(simple_action(
            "recovery.action.ingest_content",
            "The index holds no live records; ingest content to populate it.",
            &[
                "fsfs",
                "index",
                "--index-dir",
                ARG_INDEX_DIR,
                ARG_SOURCE_DIR,
            ],
            false,
            false,
            true,
            false,
            "recovery.post.index_populated",
        )),
        ZeroSignalReason::NoUsableVectors => Some(simple_action(
            "recovery.action.reindex_full",
            "Live records exist but none of their stored vectors is usable (zero-norm or \
             corrupt); rebuild index artifacts in place.",
            &[
                "fsfs",
                "index",
                "--full",
                "--index-dir",
                ARG_INDEX_DIR,
                ARG_SOURCE_DIR,
            ],
            false,
            true,
            false,
            true,
            "recovery.post.index_rebuilt",
        )),
        ZeroSignalReason::AnnReturnedEmptyDespiteUsableVectors => Some(simple_action(
            "recovery.action.rebuild_ann",
            "The ANN graph returned no candidates although usable live vectors exist; \
             rebuild the sidecar. The vector index is untouched.",
            &[
                "fsfs",
                "index",
                "--index-dir",
                ARG_INDEX_DIR,
                ARG_SOURCE_DIR,
            ],
            false,
            false,
            true,
            false,
            "recovery.post.ann_rebuilt",
        )),
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

    fn target() -> ModelAcquisitionTarget {
        ModelAcquisitionTarget {
            manifest_fingerprint: "a".repeat(64),
            upstream_revision: "revision-0123456789abcdef".to_owned(),
            license_spdx: "Apache-2.0".to_owned(),
            network_source_hosts: vec!["models.example.test".to_owned()],
            byte_budget: 42_000_000,
            destination_class: ModelDestinationClass::ManagedCache,
            destination_fingerprint: "b".repeat(64),
        }
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
            SemanticReadiness::ModelMissing,
            SemanticReadiness::ModelUnloadable,
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

    fn planned(
        state: SemanticReadiness,
        request: RecoveryRequest,
        policy: RecoveryPolicy,
    ) -> RecoveryPlan {
        plan(state, request, policy, Some(&target())).expect("valid recovery plan")
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
                        let plan = plan(state.clone(), request, policy, Some(&target()))
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
            SemanticReadiness::ModelMissing,
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
        ] {
            assert!(emitted.contains(appended), "v2 code absent: {appended}");
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
                response.requested_topology,
                RetrievalTopology::FullProgressive
            );
            assert_eq!(response.realized_topology, RetrievalTopology::LexicalOnly);
            assert_eq!(response.coverage_ppm, 0);
            assert_eq!(response.admitted_semantic_scores, 0);
            assert_eq!(
                response.degradation_reason_code.as_deref(),
                Some(state.state_code())
            );
        }
    }

    #[test]
    fn acquisition_never_claims_readiness() {
        for state in [
            SemanticReadiness::ModelMissing,
            SemanticReadiness::ModelUnloadable,
        ] {
            let plan = planned(state, explicit(RetrievalTopology::FastOnly), permissive());
            let action = plan.action.expect("acquisition state has an action");
            assert_eq!(
                action.expected_postcondition,
                "recovery.post.model_acquired_unverified"
            );
            assert_ne!(action.expected_postcondition, "recovery.state.ready");
            assert!(action.network_required);
            assert!(action.consent_required);
            let authorization = action
                .required_authorization
                .expect("acquisition binds exact authorization");
            assert_eq!(authorization.manifest_fingerprint, "a".repeat(64));
            assert_eq!(authorization.upstream_revision, "revision-0123456789abcdef");
            assert_eq!(authorization.license_spdx, "Apache-2.0");
            assert_eq!(authorization.byte_budget, 42_000_000);
            assert_eq!(
                authorization.destination_class,
                ModelDestinationClass::ManagedCache
            );
            assert_eq!(authorization.destination_fingerprint, "b".repeat(64));
            assert!(matches!(
                authorization.source,
                ModelAcquisitionSource::Network { .. }
            ));
        }
    }

    #[test]
    fn offline_policy_emits_local_bundle_action_with_zero_network() {
        let policy = RecoveryPolicy {
            interaction: InteractionPolicy::Interactive,
            network: NetworkPolicy::Offline,
            acquisition_authorization: None,
        };
        let plan = planned(
            SemanticReadiness::ModelMissing,
            hybrid(RetrievalTopology::FullProgressive),
            policy,
        );
        assert_eq!(plan.retryability, Retryability::AfterAction);
        let action = plan.action.expect("action still recommended");
        assert_eq!(
            action.argv,
            [
                "fsfs",
                "download-models",
                "--from-file",
                ARG_MODEL_BUNDLE,
                "--verify",
            ]
        );
        assert!(!action.network_required);
        assert!(action.consent_required);
        assert!(action.prerequisites.is_empty());
        assert!(matches!(
            action.required_authorization.expect("offline scope").source,
            ModelAcquisitionSource::LocalBundle
        ));
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
                Retryability::BlockedByPolicy,
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
            assert_eq!(plan.retryability, Retryability::AfterAction);
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
            response.realized_topology,
            RetrievalTopology::PartialQuality {
                coverage_ppm: 625_000
            }
        );
        assert_eq!(response.coverage_ppm, 625_000);
        assert_eq!(response.admitted_semantic_scores, 0);
        assert!(response.degradation_reason_code.is_none());
        let action = plan.action.expect("backfill recommended");
        assert_eq!(action.code, "recovery.action.backfill_quality");
        assert!(action.preserves_old_data);
        // An action exists, so retrying after it is the truthful verdict
        // even though the lane already serves.
        assert_eq!(plan.retryability, Retryability::AfterAction);
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
            SemanticReadiness::ModelMissing,
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
                    .realized_topology,
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
                .realized_topology,
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
            assert_eq!(contract.admitted_semantic_scores, 7);
            let json = serde_json::to_string(&contract).expect("serialize response");
            let decoded: SemanticResponseContract =
                serde_json::from_str(&json).expect("deserialize response");
            assert_eq!(decoded, contract);
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
        for requested in [
            RetrievalTopology::LexicalOnly,
            RetrievalTopology::PartialQuality {
                coverage_ppm: 500_000,
            },
        ] {
            assert!(matches!(
                SemanticResponseContract::new(
                    requested,
                    RetrievalTopology::LexicalOnly,
                    0,
                    0,
                    Some("recovery.state.model_missing".to_owned()),
                ),
                Err(RecoveryContractError::IncompatibleResponseTopology { .. })
            ));
        }
        for (requested, realized) in [
            (RetrievalTopology::FastOnly, RetrievalTopology::QualityOnly),
            (
                RetrievalTopology::QualityOnly,
                RetrievalTopology::FullProgressive,
            ),
            (
                RetrievalTopology::FullProgressive,
                RetrievalTopology::FastOnly,
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
                SemanticReadiness::ModelMissing,
                explicit(RetrievalTopology::FastOnly),
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
                SemanticReadiness::ModelMissing,
                explicit(RetrievalTopology::FastOnly),
                RecoveryPolicy {
                    interaction: InteractionPolicy::NonInteractive,
                    network,
                    acquisition_authorization: Some(required.clone()),
                },
            );
            assert_eq!(exact.retryability, Retryability::AfterAction);
            assert!(
                exact
                    .action
                    .expect("acquisition action")
                    .prerequisites
                    .is_empty()
            );

            let mut mismatches = Vec::new();
            let mut authorization = required.clone();
            authorization.manifest_fingerprint = "c".repeat(64);
            mismatches.push(authorization);
            let mut authorization = required.clone();
            authorization.upstream_revision.push_str("-different");
            mismatches.push(authorization);
            let mut authorization = required.clone();
            authorization.license_spdx = "MIT".to_owned();
            mismatches.push(authorization);
            let mut authorization = required.clone();
            authorization.source = match network {
                NetworkPolicy::Allowed => ModelAcquisitionSource::LocalBundle,
                NetworkPolicy::Offline => ModelAcquisitionSource::Network {
                    source_hosts: vec!["models.example.test".to_owned()],
                },
            };
            mismatches.push(authorization);
            let mut authorization = required.clone();
            authorization.byte_budget += 1;
            mismatches.push(authorization);
            let mut authorization = required.clone();
            authorization.destination_class = ModelDestinationClass::ExplicitDirectory;
            mismatches.push(authorization);
            let mut authorization = required.clone();
            authorization.destination_fingerprint = "d".repeat(64);
            mismatches.push(authorization);

            for authorization in mismatches {
                let plan = planned(
                    SemanticReadiness::ModelMissing,
                    explicit(RetrievalTopology::FastOnly),
                    RecoveryPolicy {
                        interaction: InteractionPolicy::NonInteractive,
                        network,
                        acquisition_authorization: Some(authorization),
                    },
                );
                assert_eq!(plan.retryability, Retryability::BlockedByPolicy);
                assert!(
                    plan.action
                        .expect("acquisition action")
                        .prerequisites
                        .contains(&"recovery.policy.grant_consent".to_owned())
                );
            }
        }
    }

    #[test]
    fn acquisition_target_must_be_bound_and_well_formed() {
        let unbound = plan(
            SemanticReadiness::ModelMissing,
            explicit(RetrievalTopology::FastOnly),
            permissive(),
            None,
        )
        .expect("missing binding yields a non-executable plan");
        assert_eq!(unbound.retryability, Retryability::BlockedByPolicy);
        let action = unbound.action.expect("acquisition action");
        assert!(action.required_authorization.is_none());
        assert_eq!(action.prerequisites, ["recovery.policy.bind_model"]);

        let mut malformed = target();
        malformed.manifest_fingerprint = " ".to_owned();
        assert!(matches!(
            plan(
                SemanticReadiness::ModelMissing,
                explicit(RetrievalTopology::FastOnly),
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
                SemanticReadiness::ModelMissing,
                explicit(RetrievalTopology::FastOnly),
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
                SemanticReadiness::ModelMissing,
                explicit(RetrievalTopology::FastOnly),
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
                SemanticReadiness::ModelMissing,
                explicit(RetrievalTopology::FastOnly),
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
                SemanticReadiness::ModelMissing,
                explicit(RetrievalTopology::FastOnly),
                permissive(),
                Some(&malformed),
            ),
            Err(RecoveryContractError::ZeroAcquisitionByteBudget)
        );

        let mut missing_host = target();
        missing_host.network_source_hosts.clear();
        assert_eq!(
            plan(
                SemanticReadiness::ModelMissing,
                explicit(RetrievalTopology::FastOnly),
                permissive(),
                Some(&missing_host),
            ),
            Err(RecoveryContractError::MissingNetworkSourceHosts)
        );
        let offline = plan(
            SemanticReadiness::ModelMissing,
            explicit(RetrievalTopology::FastOnly),
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
                SemanticReadiness::ModelMissing,
                explicit(RetrievalTopology::FastOnly),
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
                    SemanticReadiness::ModelMissing,
                    explicit(RetrievalTopology::FastOnly),
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
        let mut plan_json = serde_json::to_value(planned(
            SemanticReadiness::ModelMissing,
            hybrid(RetrievalTopology::FullProgressive),
            permissive(),
        ))
        .expect("serialize plan");

        let mut changed = plan_json.clone();
        changed["schema_version"] =
            serde_json::Value::String("frankensearch.recovery_plan.v1".to_owned());
        assert!(serde_json::from_value::<RecoveryPlan>(changed).is_err());

        let mut changed = plan_json.clone();
        changed["unknown_contract_field"] = serde_json::Value::Bool(true);
        assert!(serde_json::from_value::<RecoveryPlan>(changed).is_err());

        let mut changed = plan_json.clone();
        changed["state_code"] = serde_json::Value::String("recovery.state.ready".to_owned());
        assert!(serde_json::from_value::<RecoveryPlan>(changed).is_err());

        let mut changed = plan_json.clone();
        changed["provenance"] = serde_json::Value::String("verified_local".to_owned());
        assert!(serde_json::from_value::<RecoveryPlan>(changed).is_err());

        let mut changed = plan_json.clone();
        changed["response_contract"]["admitted_semantic_scores"] = serde_json::Value::from(1);
        assert!(serde_json::from_value::<RecoveryPlan>(changed).is_err());

        let authorization = plan_json["action"]["required_authorization"].clone();
        let mut changed = authorization.clone();
        changed["schema_version"] = serde_json::Value::String(
            "frankensearch.model_acquisition_authorization.v2".to_owned(),
        );
        assert!(serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err());

        let mut changed = authorization.clone();
        changed["source"]["kind"] = serde_json::Value::String("ambient".to_owned());
        assert!(serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err());

        let mut changed = authorization.clone();
        changed["destination_class"] = serde_json::Value::String("unbounded_path".to_owned());
        assert!(serde_json::from_value::<ModelAcquisitionAuthorization>(changed).is_err());

        let mut changed = authorization.clone();
        changed["byte_budget"] = serde_json::Value::from(0);
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

        plan_json["action"]["required_authorization"]["source"] = serde_json::json!({
            "kind": "local_bundle",
            "source_hosts": ["must-not-be-ignored.example.test"]
        });
        assert!(serde_json::from_value::<RecoveryPlan>(plan_json).is_err());
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
                    let original = plan(state.clone(), request, policy, Some(&target()))
                        .expect("valid representative plan");
                    let json = serde_json::to_string(&original).expect("serialize plan");
                    assert!(json.contains(RECOVERY_PLAN_SCHEMA_VERSION));
                    let decoded: RecoveryPlan =
                        serde_json::from_str(&json).expect("deserialize plan");
                    assert_eq!(decoded, original);
                }
            }
        }
        let hash = planned(SemanticReadiness::HashControl, hash_control(), permissive());
        let json = serde_json::to_string(&hash).expect("serialize hash plan");
        let decoded: RecoveryPlan = serde_json::from_str(&json).expect("deserialize hash plan");
        assert_eq!(decoded, hash);
    }

    #[test]
    fn golden_plan_json_for_model_missing_offline_hybrid() {
        let plan = planned(
            SemanticReadiness::ModelMissing,
            hybrid(RetrievalTopology::FullProgressive),
            RecoveryPolicy {
                interaction: InteractionPolicy::NonInteractive,
                network: NetworkPolicy::Offline,
                acquisition_authorization: None,
            },
        );
        let json = serde_json::to_value(&plan).expect("serialize plan");
        assert_eq!(json["schema_version"], "frankensearch.recovery_plan.v2");
        assert_eq!(json["state"]["state"], "model_missing");
        assert_eq!(json["state_code"], "recovery.state.model_missing");
        assert_eq!(json["provenance"], "unavailable");
        assert_eq!(json["mode"], "hybrid");
        assert_eq!(json["requested_topology"]["topology"], "full_progressive");
        assert_eq!(json["policy"]["interaction"], "non_interactive");
        assert_eq!(json["policy"]["network"], "offline");
        assert!(json["policy"]["acquisition_authorization"].is_null());
        assert_eq!(json["semantic_available"], false);
        assert_eq!(json["retryability"], "blocked_by_policy");
        assert_eq!(json["action"]["code"], "recovery.action.acquire_model");
        assert_eq!(
            json["action"]["argv"],
            serde_json::json!([
                "fsfs",
                "download-models",
                "--from-file",
                "ARG_MODEL_BUNDLE",
                "--verify"
            ])
        );
        assert_eq!(json["action"]["network_required"], false);
        assert_eq!(json["action"]["consent_required"], true);
        assert_eq!(
            json["action"]["prerequisites"][0],
            "recovery.policy.grant_consent"
        );
        assert_eq!(
            json["action"]["required_authorization"]["schema_version"],
            "frankensearch.model_acquisition_authorization.v1"
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
            SemanticReadiness::ModelMissing,
            SemanticReadiness::ModelUnloadable,
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
                    Retryability::BlockedByPolicy => "blocked",
                };
                let (realized, coverage) = plan
                    .response_contract
                    .as_ref()
                    .map_or(("none", 0), |response| {
                        (response.realized_topology.code(), response.coverage_ppm)
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
            hash_response.realized_topology.code(),
            hash_response.coverage_ppm
        ));
        let expected: Vec<&str> = vec![
            "recovery.state.ready|semantic => none,false,false,not_needed,full_progressive,1000000",
            "recovery.state.ready|hybrid => none,false,false,not_needed,full_progressive,1000000",
            "recovery.state.model_missing|semantic => recovery.action.acquire_model,true,true,after_action,none,0",
            "recovery.state.model_missing|hybrid => recovery.action.acquire_model,true,true,after_action,lexical_only,0",
            "recovery.state.model_unloadable|semantic => recovery.action.reacquire_model,true,true,after_action,none,0",
            "recovery.state.model_unloadable|hybrid => recovery.action.reacquire_model,true,true,after_action,lexical_only,0",
            "recovery.state.index_absent|semantic => recovery.action.build_index,false,false,after_action,none,0",
            "recovery.state.index_absent|hybrid => recovery.action.build_index,false,false,after_action,lexical_only,0",
            "recovery.state.identity_mismatch|semantic => recovery.action.reindex_full,false,true,after_action,none,0",
            "recovery.state.identity_mismatch|hybrid => recovery.action.reindex_full,false,true,after_action,lexical_only,0",
            "recovery.state.daemon_mismatch|semantic => recovery.action.restart_daemon,false,false,after_action,none,0",
            "recovery.state.daemon_mismatch|hybrid => recovery.action.restart_daemon,false,false,after_action,lexical_only,0",
            "recovery.state.index_empty|semantic => recovery.action.ingest_content,false,false,after_action,none,0",
            "recovery.state.index_empty|hybrid => recovery.action.ingest_content,false,false,after_action,lexical_only,0",
            "recovery.state.manifest_unsafe|semantic => recovery.action.reindex_full,false,true,after_action,none,0",
            "recovery.state.manifest_unsafe|hybrid => recovery.action.reindex_full,false,true,after_action,lexical_only,0",
            "recovery.state.ann_stale|semantic => recovery.action.rebuild_ann,false,false,after_action,none,0",
            "recovery.state.ann_stale|hybrid => recovery.action.rebuild_ann,false,false,after_action,lexical_only,0",
            "recovery.state.generation_incomplete|semantic => recovery.action.resume_index,false,false,after_action,none,0",
            "recovery.state.generation_incomplete|hybrid => recovery.action.resume_index,false,false,after_action,lexical_only,0",
            "recovery.state.partial_quality_coverage|semantic => recovery.action.backfill_quality,false,false,after_action,partial_quality,750000",
            "recovery.state.partial_quality_coverage|hybrid => recovery.action.backfill_quality,false,false,after_action,partial_quality,750000",
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
