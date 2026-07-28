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
//!   but only with explicit [`ResponseDegradation`] metadata.
//! - The planner is pure and exhaustive: every enumerated state is matched
//!   without wildcards, so a new readiness state fails compilation until it
//!   is planned for.
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

use crate::config::ZeroSignalReason;

/// Schema version for serialized [`RecoveryPlan`] payloads.
pub const RECOVERY_PLAN_SCHEMA_VERSION: &str = "frankensearch.recovery_plan.v1";

/// Placeholder token integrators substitute with the resolved index
/// directory.
///
/// The pure planner never sees real user paths (they are redacted from
/// telemetry); rendering a runnable command is the integrator's job.
pub const ARG_INDEX_DIR: &str = "<index-dir>";

/// Placeholder token integrators substitute with the corpus source
/// directory to (re-)ingest.
pub const ARG_SOURCE_DIR: &str = "<source-dir>";

/// Why the caller's request cannot be served semantically right now.
///
/// This is the planner's input state, produced by readiness probes
/// (model manifest checks, index census, generation binding). Variants
/// mirror the states bd-vmv7 enumerates; [`ZeroSignalReason`] carries the
/// finer classification for empty-index states so the two vocabularies
/// stay aligned rather than diverging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "state", content = "detail")]
pub enum SemanticReadiness {
    /// The semantic lane is fully usable: verified model, loadable, and a
    /// compatible index with usable live vectors.
    Ready,
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
    PartialQualityCoverage,
}

impl SemanticReadiness {
    /// Every planner-input state, with representative zero-signal reasons
    /// covering each classification family (benign state, request-scoped,
    /// availability failure, ANN anomaly).
    pub const ALL_REPRESENTATIVE: &'static [Self] = &[
        Self::Ready,
        Self::ModelMissing,
        Self::ModelUnloadable,
        Self::IndexAbsent,
        Self::IdentityMismatch,
        Self::DaemonMismatch,
        Self::IndexEmpty(ZeroSignalReason::NewlyCreatedEmpty),
        Self::IndexEmpty(ZeroSignalReason::AllTombstoned),
        Self::IndexEmpty(ZeroSignalReason::WalOnlyNoLiveRecords),
        Self::IndexEmpty(ZeroSignalReason::CallerRequestedZeroK),
        Self::IndexEmpty(ZeroSignalReason::NoUsableVectors),
        Self::IndexEmpty(ZeroSignalReason::AnnReturnedEmptyDespiteUsableVectors),
        Self::ManifestUnsafe,
        Self::AnnStale,
        Self::GenerationIncomplete,
        Self::PartialQualityCoverage,
    ];

    /// Stable three-segment state code.
    #[must_use]
    pub const fn state_code(self) -> &'static str {
        match self {
            Self::Ready => "recovery.state.ready",
            Self::ModelMissing => "recovery.state.model_missing",
            Self::ModelUnloadable => "recovery.state.model_unloadable",
            Self::IndexAbsent => "recovery.state.index_absent",
            Self::IdentityMismatch => "recovery.state.identity_mismatch",
            Self::DaemonMismatch => "recovery.state.daemon_mismatch",
            Self::IndexEmpty(_) => "recovery.state.index_empty",
            Self::ManifestUnsafe => "recovery.state.manifest_unsafe",
            Self::AnnStale => "recovery.state.ann_stale",
            Self::GenerationIncomplete => "recovery.state.generation_incomplete",
            Self::PartialQualityCoverage => "recovery.state.partial_quality_coverage",
        }
    }

    /// True when semantic results can be served (possibly with reduced
    /// refinement quality). Only [`Self::Ready`] and
    /// [`Self::PartialQualityCoverage`] qualify: partial coverage degrades
    /// refinement, not availability.
    #[must_use]
    pub const fn semantic_available(self) -> bool {
        matches!(self, Self::Ready | Self::PartialQualityCoverage)
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryPolicy {
    pub interaction: InteractionPolicy,
    pub network: NetworkPolicy,
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

/// Metadata attached to a hybrid response that proceeded without the
/// semantic lane. Absent for explicit-semantic requests (which fail
/// closed) and for ready lanes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseDegradation {
    /// Always `"lexical_only"` in schema v1.
    pub mode: String,
    /// The state code that forced the degradation.
    pub reason_code: String,
}

/// The full plan: current state, verdict for the requested mode, and the
/// truthful next action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryPlan {
    /// [`RECOVERY_PLAN_SCHEMA_VERSION`].
    pub schema_version: String,
    /// The readiness state the plan was computed from.
    pub state: SemanticReadiness,
    /// Stable code for `state` (denormalized for consumers that do not
    /// decode the enum).
    pub state_code: String,
    /// The mode the caller requested.
    pub mode: RequestMode,
    /// The policy the plan was computed under.
    pub policy: RecoveryPolicy,
    /// Whether semantic results can be served right now.
    pub semantic_available: bool,
    /// Whether retrying can succeed, and under what condition.
    pub retryability: Retryability,
    /// The truthful next action; `None` when the state needs none (ready,
    /// or the emptiness was request-scoped).
    pub action: Option<RecoveryAction>,
    /// Present exactly when a hybrid request may proceed lexical-only.
    pub degraded_response: Option<ResponseDegradation>,
}

/// Compute the truthful plan for a readiness state under a mode and
/// policy. Pure and deterministic: identical inputs yield identical plans.
#[must_use]
pub fn plan(state: SemanticReadiness, mode: RequestMode, policy: RecoveryPolicy) -> RecoveryPlan {
    let action = action_for(state);
    let (action, retryability) = match action {
        None => (
            None,
            if state.semantic_available() {
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
                action
                    .prerequisites
                    .push("recovery.policy.allow_network".to_owned());
            }
            let consent_blocked = action.consent_required
                && matches!(policy.interaction, InteractionPolicy::NonInteractive);
            if consent_blocked {
                action
                    .prerequisites
                    .push("recovery.policy.grant_consent".to_owned());
            }
            let retryability = if network_blocked || consent_blocked {
                Retryability::BlockedByPolicy
            } else {
                Retryability::AfterAction
            };
            (Some(action), retryability)
        }
    };

    let semantic_available = state.semantic_available();
    let degraded_response = match (mode, semantic_available) {
        (RequestMode::Hybrid, false) => Some(ResponseDegradation {
            mode: "lexical_only".to_owned(),
            reason_code: state.state_code().to_owned(),
        }),
        _ => None,
    };

    RecoveryPlan {
        schema_version: RECOVERY_PLAN_SCHEMA_VERSION.to_owned(),
        state,
        state_code: state.state_code().to_owned(),
        mode,
        policy,
        semantic_available,
        retryability,
        action,
        degraded_response,
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
    }
}

/// The exhaustive state → action table. No wildcard arm: adding a
/// readiness state without planning for it is a compile error.
fn action_for(state: SemanticReadiness) -> Option<RecoveryAction> {
    match state {
        SemanticReadiness::Ready => None,
        SemanticReadiness::ModelMissing => Some(simple_action(
            "recovery.action.acquire_model",
            "Download and verify the configured semantic models. Acquisition alone does not \
             make the lane searchable: readiness additionally requires the load self-test and \
             a compatible non-empty index.",
            &["fsfs", "download-models", "--verify"],
            true,
            false,
            true,
            false,
            "recovery.post.model_acquired_unverified",
        )),
        SemanticReadiness::ModelUnloadable => Some(simple_action(
            "recovery.action.reacquire_model",
            "The cached model failed verification or its load self-test; re-download and \
             re-verify it. Index data is untouched.",
            &["fsfs", "download-models", "--verify", "--force"],
            true,
            false,
            true,
            false,
            "recovery.post.model_acquired_unverified",
        )),
        SemanticReadiness::IndexAbsent => Some(simple_action(
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
        )),
        SemanticReadiness::IdentityMismatch => Some(simple_action(
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
        )),
        SemanticReadiness::DaemonMismatch => Some(simple_action(
            "recovery.action.restart_daemon",
            "The embedding daemon serves a different space than the local configuration; \
             restart it so both sides agree before trusting daemon vectors.",
            &["fsfs", "daemon", "restart"],
            false,
            false,
            true,
            false,
            "recovery.post.daemon_aligned",
        )),
        SemanticReadiness::IndexEmpty(reason) => plan_for_empty(reason),
        SemanticReadiness::ManifestUnsafe => Some(simple_action(
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
        )),
        SemanticReadiness::AnnStale => Some(simple_action(
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
        )),
        SemanticReadiness::GenerationIncomplete => Some(simple_action(
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
        )),
        SemanticReadiness::PartialQualityCoverage => Some(simple_action(
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
        )),
    }
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

    const ALL_MODES: &[RequestMode] = &[RequestMode::ExplicitSemantic, RequestMode::Hybrid];

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
                });
            }
        }
        out
    }

    fn permissive() -> RecoveryPolicy {
        RecoveryPolicy {
            interaction: InteractionPolicy::Interactive,
            network: NetworkPolicy::Allowed,
        }
    }

    #[test]
    fn every_stable_code_is_valid_and_unique() {
        let mut codes = Vec::new();
        for &state in SemanticReadiness::ALL_REPRESENTATIVE {
            codes.push(state.state_code().to_owned());
            for &mode in ALL_MODES {
                for policy in all_policies() {
                    let plan = plan(state, mode, policy);
                    if let Some(action) = plan.action {
                        codes.push(action.code.clone());
                        codes.push(action.expected_postcondition.clone());
                        codes.extend(action.prerequisites);
                    }
                }
            }
        }
        for code in &codes {
            assert!(
                ReasonCode::new(code.as_str()).is_valid(),
                "invalid stable code format: {code}"
            );
        }
        // Distinct states never share a code with distinct actions.
        let states: std::collections::HashSet<_> = SemanticReadiness::ALL_REPRESENTATIVE
            .iter()
            .map(|s| s.state_code())
            .collect();
        assert_eq!(states.len(), 11, "one code per state variant");
    }

    #[test]
    fn explicit_semantic_fails_closed_for_every_unready_state() {
        for &state in SemanticReadiness::ALL_REPRESENTATIVE {
            let plan = plan(state, RequestMode::ExplicitSemantic, permissive());
            assert_eq!(plan.semantic_available, state.semantic_available());
            assert!(
                plan.degraded_response.is_none(),
                "explicit semantic never degrades to lexical: {state:?}"
            );
        }
    }

    #[test]
    fn hybrid_degrades_with_metadata_exactly_when_unavailable() {
        for &state in SemanticReadiness::ALL_REPRESENTATIVE {
            let plan = plan(state, RequestMode::Hybrid, permissive());
            match plan.degraded_response {
                Some(degraded) => {
                    assert!(!state.semantic_available());
                    assert_eq!(degraded.mode, "lexical_only");
                    assert_eq!(degraded.reason_code, state.state_code());
                }
                None => assert!(
                    state.semantic_available(),
                    "unavailable hybrid state must carry degradation metadata: {state:?}"
                ),
            }
        }
    }

    #[test]
    fn acquisition_never_claims_readiness() {
        for state in [
            SemanticReadiness::ModelMissing,
            SemanticReadiness::ModelUnloadable,
        ] {
            let plan = plan(state, RequestMode::ExplicitSemantic, permissive());
            let action = plan.action.expect("acquisition state has an action");
            assert_eq!(
                action.expected_postcondition,
                "recovery.post.model_acquired_unverified"
            );
            assert_ne!(action.expected_postcondition, "recovery.state.ready");
            assert!(action.network_required);
        }
    }

    #[test]
    fn offline_policy_blocks_network_actions_with_prerequisite() {
        let policy = RecoveryPolicy {
            interaction: InteractionPolicy::Interactive,
            network: NetworkPolicy::Offline,
        };
        let plan = plan(SemanticReadiness::ModelMissing, RequestMode::Hybrid, policy);
        assert_eq!(plan.retryability, Retryability::BlockedByPolicy);
        let action = plan.action.expect("action still recommended");
        assert!(
            action
                .prerequisites
                .contains(&"recovery.policy.allow_network".to_owned())
        );
    }

    #[test]
    fn noninteractive_policy_blocks_consent_actions_with_prerequisite() {
        let policy = RecoveryPolicy {
            interaction: InteractionPolicy::NonInteractive,
            network: NetworkPolicy::Allowed,
        };
        for state in [
            SemanticReadiness::IdentityMismatch,
            SemanticReadiness::ManifestUnsafe,
            SemanticReadiness::IndexEmpty(ZeroSignalReason::NoUsableVectors),
        ] {
            let plan = plan(state, RequestMode::ExplicitSemantic, policy);
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
            let plan = plan(
                SemanticReadiness::IndexEmpty(reason),
                RequestMode::ExplicitSemantic,
                permissive(),
            );
            assert!(plan.action.is_none(), "{reason:?}");
            assert_eq!(plan.retryability, Retryability::AfterAction);
            assert!(!plan.semantic_available);
        }
    }

    #[test]
    fn partial_coverage_is_available_with_a_backfill_action() {
        let plan = plan(
            SemanticReadiness::PartialQualityCoverage,
            RequestMode::Hybrid,
            permissive(),
        );
        assert!(plan.semantic_available);
        assert!(plan.degraded_response.is_none());
        let action = plan.action.expect("backfill recommended");
        assert_eq!(action.code, "recovery.action.backfill_quality");
        assert!(action.preserves_old_data);
        // An action exists, so retrying after it is the truthful verdict
        // even though the lane already serves.
        assert_eq!(plan.retryability, Retryability::AfterAction);
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
        };
        assert_eq!(
            action.shell_command(),
            "fsfs index --index-dir '/data/My Projects/it'\\''s here' <source-dir>"
        );
    }

    #[test]
    fn serialization_roundtrips_and_locks_schema_version() {
        for &state in SemanticReadiness::ALL_REPRESENTATIVE {
            for &mode in ALL_MODES {
                for policy in all_policies() {
                    let original = plan(state, mode, policy);
                    let json = serde_json::to_string(&original).expect("serialize plan");
                    assert!(json.contains(RECOVERY_PLAN_SCHEMA_VERSION));
                    let decoded: RecoveryPlan =
                        serde_json::from_str(&json).expect("deserialize plan");
                    assert_eq!(decoded, original);
                }
            }
        }
    }

    #[test]
    fn golden_plan_json_for_model_missing_offline_hybrid() {
        let plan = plan(
            SemanticReadiness::ModelMissing,
            RequestMode::Hybrid,
            RecoveryPolicy {
                interaction: InteractionPolicy::NonInteractive,
                network: NetworkPolicy::Offline,
            },
        );
        let json = serde_json::to_value(&plan).expect("serialize plan");
        assert_eq!(json["schema_version"], "frankensearch.recovery_plan.v1");
        assert_eq!(json["state"]["state"], "model_missing");
        assert_eq!(json["state_code"], "recovery.state.model_missing");
        assert_eq!(json["mode"], "hybrid");
        assert_eq!(json["policy"]["interaction"], "non_interactive");
        assert_eq!(json["policy"]["network"], "offline");
        assert_eq!(json["semantic_available"], false);
        assert_eq!(json["retryability"], "blocked_by_policy");
        assert_eq!(json["action"]["code"], "recovery.action.acquire_model");
        assert_eq!(json["action"]["network_required"], true);
        assert_eq!(
            json["action"]["prerequisites"][0],
            "recovery.policy.allow_network"
        );
        assert_eq!(json["degraded_response"]["mode"], "lexical_only");
        assert_eq!(
            json["degraded_response"]["reason_code"],
            "recovery.state.model_missing"
        );
    }

    #[test]
    fn golden_transition_table_is_stable() {
        // One compact row per (state, mode) under the permissive policy:
        // "state_code|mode => action_code,net,consent,retry". Append-only
        // within schema v1; a diff here is a contract change.
        let mut rows = Vec::new();
        for &state in SemanticReadiness::ALL_REPRESENTATIVE {
            for &mode in ALL_MODES {
                let plan = plan(state, mode, permissive());
                let mode_tag = match mode {
                    RequestMode::ExplicitSemantic => "semantic",
                    RequestMode::Hybrid => "hybrid",
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
                let empty_tag = match state {
                    SemanticReadiness::IndexEmpty(reason) => {
                        format!("[{}]", reason.reason_code())
                    }
                    _ => String::new(),
                };
                rows.push(format!(
                    "{}{}|{} => {},{}",
                    plan.state_code, empty_tag, mode_tag, action_tag, retry_tag
                ));
            }
        }
        let expected: Vec<&str> = vec![
            "recovery.state.ready|semantic => none,false,false,not_needed",
            "recovery.state.ready|hybrid => none,false,false,not_needed",
            "recovery.state.model_missing|semantic => recovery.action.acquire_model,true,false,after_action",
            "recovery.state.model_missing|hybrid => recovery.action.acquire_model,true,false,after_action",
            "recovery.state.model_unloadable|semantic => recovery.action.reacquire_model,true,false,after_action",
            "recovery.state.model_unloadable|hybrid => recovery.action.reacquire_model,true,false,after_action",
            "recovery.state.index_absent|semantic => recovery.action.build_index,false,false,after_action",
            "recovery.state.index_absent|hybrid => recovery.action.build_index,false,false,after_action",
            "recovery.state.identity_mismatch|semantic => recovery.action.reindex_full,false,true,after_action",
            "recovery.state.identity_mismatch|hybrid => recovery.action.reindex_full,false,true,after_action",
            "recovery.state.daemon_mismatch|semantic => recovery.action.restart_daemon,false,false,after_action",
            "recovery.state.daemon_mismatch|hybrid => recovery.action.restart_daemon,false,false,after_action",
            "recovery.state.index_empty[zerosignal.state.newly_created_empty]|semantic => recovery.action.ingest_content,false,false,after_action",
            "recovery.state.index_empty[zerosignal.state.newly_created_empty]|hybrid => recovery.action.ingest_content,false,false,after_action",
            "recovery.state.index_empty[zerosignal.state.all_tombstoned]|semantic => recovery.action.ingest_content,false,false,after_action",
            "recovery.state.index_empty[zerosignal.state.all_tombstoned]|hybrid => recovery.action.ingest_content,false,false,after_action",
            "recovery.state.index_empty[zerosignal.state.wal_only_no_live_records]|semantic => recovery.action.ingest_content,false,false,after_action",
            "recovery.state.index_empty[zerosignal.state.wal_only_no_live_records]|hybrid => recovery.action.ingest_content,false,false,after_action",
            "recovery.state.index_empty[zerosignal.request.zero_k]|semantic => none,false,false,after_action",
            "recovery.state.index_empty[zerosignal.request.zero_k]|hybrid => none,false,false,after_action",
            "recovery.state.index_empty[zerosignal.availability.no_usable_vectors]|semantic => recovery.action.reindex_full,false,true,after_action",
            "recovery.state.index_empty[zerosignal.availability.no_usable_vectors]|hybrid => recovery.action.reindex_full,false,true,after_action",
            "recovery.state.index_empty[zerosignal.availability.ann_empty_despite_usable_vectors]|semantic => recovery.action.rebuild_ann,false,false,after_action",
            "recovery.state.index_empty[zerosignal.availability.ann_empty_despite_usable_vectors]|hybrid => recovery.action.rebuild_ann,false,false,after_action",
            "recovery.state.manifest_unsafe|semantic => recovery.action.reindex_full,false,true,after_action",
            "recovery.state.manifest_unsafe|hybrid => recovery.action.reindex_full,false,true,after_action",
            "recovery.state.ann_stale|semantic => recovery.action.rebuild_ann,false,false,after_action",
            "recovery.state.ann_stale|hybrid => recovery.action.rebuild_ann,false,false,after_action",
            "recovery.state.generation_incomplete|semantic => recovery.action.resume_index,false,false,after_action",
            "recovery.state.generation_incomplete|hybrid => recovery.action.resume_index,false,false,after_action",
            "recovery.state.partial_quality_coverage|semantic => recovery.action.backfill_quality,false,false,after_action",
            "recovery.state.partial_quality_coverage|hybrid => recovery.action.backfill_quality,false,false,after_action",
        ];
        assert_eq!(
            rows, expected,
            "transition table changed: contract review required"
        );
    }
}
