//! Typed operator advice for graceful search degradation.

use std::path::Path;

use frankensearch_core::SearchError;
use frankensearch_core::config::ZeroSignalReason;
use frankensearch_core::recovery_plan::{
    ModelAcquisitionTarget, RecoveryContractError, RecoveryPlan, RecoveryPolicy, RecoveryRequest,
    RequestMode, SemanticReadiness, TrustedRecoveryContext, VerifiedSemanticProvenance, plan,
};
use frankensearch_core::traits::ModelTier;
use frankensearch_core::types::RetrievalTopology;
use serde::{Deserialize, Serialize};

pub const DEGRADATION_ADVICE_SCHEMA_VERSION: &str = "fsfs.degradation.advice.v2";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DegradationFailureKind {
    RefinementFailed,
    LexicalFallback,
    MissingQualityModel,
    UnverifiableEmbeddingSpace,
    Timeout,
    CorruptIndex,
    CacheMiss,
    SemanticZeroSignal,
    HashControlGeneration,
}

impl DegradationFailureKind {
    #[must_use]
    pub const fn reason_code(self) -> &'static str {
        match self {
            Self::RefinementFailed => "degrade.advice.refinement_failed",
            Self::LexicalFallback => "degrade.advice.lexical_fallback",
            Self::MissingQualityModel => "degrade.advice.quality_model_missing",
            Self::UnverifiableEmbeddingSpace => "degrade.advice.embedding_space_unverifiable",
            Self::Timeout => "degrade.advice.timeout",
            Self::CorruptIndex => "degrade.advice.index_corrupt",
            Self::CacheMiss => "degrade.advice.cache_miss",
            Self::SemanticZeroSignal => "degrade.advice.semantic_zero_signal",
            Self::HashControlGeneration => "degrade.advice.hash_control",
        }
    }

    #[must_use]
    pub const fn summary(self) -> &'static str {
        match self {
            Self::RefinementFailed => "quality refinement failed; initial results remain usable",
            Self::LexicalFallback => "semantic retrieval fell back to lexical search",
            Self::MissingQualityModel => "quality model unavailable; refinement skipped",
            Self::UnverifiableEmbeddingSpace => {
                "semantic index and query embedding identities cannot be proven compatible"
            }
            Self::Timeout => "quality stage exceeded its latency budget",
            Self::CorruptIndex => "index artifact could not be read safely",
            Self::CacheMiss => "expected cache artifact was missing or stale",
            Self::SemanticZeroSignal => {
                "semantic lane produced zero signal despite live records; results may be lexical-only"
            }
            Self::HashControlGeneration => {
                "vector generation is a hash control artifact, not semantic search"
            }
        }
    }

    #[must_use]
    pub const fn preserves_initial_results(self) -> bool {
        match self {
            Self::RefinementFailed
            | Self::LexicalFallback
            | Self::MissingQualityModel
            | Self::UnverifiableEmbeddingSpace
            | Self::Timeout
            | Self::CacheMiss
            | Self::SemanticZeroSignal
            | Self::HashControlGeneration => true,
            Self::CorruptIndex => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DegradationAdviceSeverity {
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdviceOutputSurface {
    CliJson,
    CliJsonl,
    CliToon,
    Tui,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DegradationNextAction {
    pub order: u16,
    pub reason_code: String,
    pub action: String,
    /// Generated POSIX-shell command, absent when its arguments cannot be
    /// represented losslessly within the diagnostic limits. Never executed here.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DegradationAdvice {
    pub schema_version: String,
    pub failure: DegradationFailureKind,
    pub severity: DegradationAdviceSeverity,
    pub reason_code: String,
    pub operator_summary: String,
    /// Public error projection, not the provider's original exception text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_error: Option<String>,
    pub preserves_initial_results: bool,
    pub next_actions: Vec<DegradationNextAction>,
    /// Generated POSIX-shell replay; empty when bounded, lossless rendering is
    /// unavailable. Explicit caller overrides retain the caller's trust contract.
    pub replay_command: String,
    pub output_surfaces: Vec<AdviceOutputSurface>,
}

#[derive(Debug, Clone, Copy)]
pub struct DegradationAdviceInput<'a> {
    pub failure: DegradationFailureKind,
    pub query: &'a str,
    pub index_dir: Option<&'a Path>,
    pub original_error: Option<&'a SearchError>,
    /// Trusted, caller-authored command override, not untrusted provider text.
    pub replay_command: Option<&'a str>,
}

impl DegradationAdvice {
    #[must_use]
    pub fn from_input(input: DegradationAdviceInput<'_>) -> Self {
        let failure = input.failure;
        // Degradation advice travels inside successful result payloads as well
        // as TUI/stream output. Do not bypass the terminal error boundary here.
        let original_error = input
            .original_error
            .map(|error| crate::output_schema::output_error_from(error).message);
        let replay_command = input
            .replay_command
            .map(str::to_owned)
            .unwrap_or_else(|| replay_command_for(input.query, input.index_dir));

        Self {
            schema_version: DEGRADATION_ADVICE_SCHEMA_VERSION.to_owned(),
            failure,
            severity: severity_for(failure),
            reason_code: failure.reason_code().to_owned(),
            operator_summary: failure.summary().to_owned(),
            original_error,
            preserves_initial_results: failure.preserves_initial_results(),
            next_actions: next_actions_for(failure, input.index_dir),
            replay_command,
            output_surfaces: vec![
                AdviceOutputSurface::CliJson,
                AdviceOutputSurface::CliJsonl,
                AdviceOutputSurface::CliToon,
                AdviceOutputSurface::Tui,
            ],
        }
    }

    /// Map this advice's failure kind to its canonical [`SemanticReadiness`]
    /// state under the caller's requested mode and retrieval topology.
    #[must_use]
    pub const fn semantic_readiness(
        &self,
        mode: RequestMode,
        requested_topology: RetrievalTopology,
    ) -> SemanticReadiness {
        semantic_readiness_for_failure(self.failure, mode, requested_topology)
    }

    /// Compute the canonical [`RecoveryPlan`] for this degradation advice.
    ///
    /// # Errors
    ///
    /// Returns [`RecoveryContractError`] if the request topology, readiness
    /// state, or acquisition authorization fails validation.
    pub fn recovery_plan(
        &self,
        request: RecoveryRequest,
        policy: &RecoveryPolicy,
        acquisition_target: Option<&ModelAcquisitionTarget>,
        evaluation_time_unix_seconds: u64,
    ) -> Result<RecoveryPlan, RecoveryContractError> {
        let readiness = self.semantic_readiness(request.mode, request.requested_topology);
        recovery_plan_for_readiness(
            &readiness,
            request,
            policy,
            acquisition_target,
            evaluation_time_unix_seconds,
        )
    }
}

#[must_use]
pub fn advice_for_search_error(
    query: &str,
    index_dir: Option<&Path>,
    error: &SearchError,
) -> DegradationAdvice {
    DegradationAdvice::from_input(DegradationAdviceInput {
        failure: classify_search_error(error),
        query,
        index_dir,
        original_error: Some(error),
        replay_command: None,
    })
}

/// Operator advice for a typed semantic zero-signal classification
/// (bd-tqhc).
///
/// Only availability failures (`NoUsableVectors`,
/// `AnnReturnedEmptyDespiteUsableVectors`) warrant operator advice: every
/// other [`ZeroSignalReason`] is an expected outcome of the request or index
/// state and must not surface as degradation. The typed reason code rides in
/// `original_error` so the advice schema stays at v2; the machine-readable
/// channel remains `TwoTierMetrics::zero_signal`.
#[must_use]
pub fn advice_for_zero_signal(
    query: &str,
    index_dir: Option<&Path>,
    reason: ZeroSignalReason,
) -> Option<DegradationAdvice> {
    if !reason.is_availability_failure() {
        return None;
    }
    let mut advice = DegradationAdvice::from_input(DegradationAdviceInput {
        failure: DegradationFailureKind::SemanticZeroSignal,
        query,
        index_dir,
        original_error: None,
        replay_command: None,
    });
    advice.original_error = Some(format!("{}: {reason}", reason.reason_code()));
    Some(advice)
}

#[must_use]
pub fn classify_search_error(error: &SearchError) -> DegradationFailureKind {
    match error {
        SearchError::SearchTimeout { .. } => DegradationFailureKind::Timeout,
        SearchError::IndexCorrupted { .. }
        | SearchError::IndexVersionMismatch { .. }
        | SearchError::DimensionMismatch { .. } => DegradationFailureKind::CorruptIndex,
        SearchError::EmbedderUnavailable { .. }
        | SearchError::ModelNotFound { .. }
        | SearchError::ModelLoadFailed { .. } => DegradationFailureKind::MissingQualityModel,
        SearchError::UnverifiableRemoteSpace { .. } => {
            DegradationFailureKind::UnverifiableEmbeddingSpace
        }
        SearchError::InvalidConfig { field, value, .. }
            if field == "semantic.vector_generation" && value == "legacy_hash" =>
        {
            DegradationFailureKind::HashControlGeneration
        }
        SearchError::InvalidConfig { .. }
        | SearchError::IndexNotFound { .. }
        | SearchError::IndexCandidatesNotFound { .. } => DegradationFailureKind::CacheMiss,
        SearchError::EmbeddingFailed { .. }
        | SearchError::RerankerUnavailable { .. }
        | SearchError::RerankFailed { .. }
        | SearchError::SubsystemError { .. }
        | SearchError::Io { .. }
        | SearchError::Cancelled { .. }
        | SearchError::FederatedInsufficientResponses { .. }
        | SearchError::QueueFull { .. }
        | SearchError::DurabilityDisabled
        | SearchError::HashMismatch { .. }
        | SearchError::QueryParseError { .. } => DegradationFailureKind::RefinementFailed,
    }
}

/// Map a [`DegradationFailureKind`] to the canonical [`SemanticReadiness`]
/// state under the caller's requested mode and retrieval topology.
#[must_use]
pub const fn semantic_readiness_for_failure(
    failure: DegradationFailureKind,
    mode: RequestMode,
    requested_topology: RetrievalTopology,
) -> SemanticReadiness {
    if matches!(mode, RequestMode::HashControl) {
        return SemanticReadiness::HashControl;
    }
    let tier = if matches!(requested_topology, RetrievalTopology::FastOnly) {
        ModelTier::Fast
    } else {
        ModelTier::Quality
    };
    match failure {
        DegradationFailureKind::MissingQualityModel => SemanticReadiness::ModelMissing { tier },
        DegradationFailureKind::RefinementFailed | DegradationFailureKind::Timeout => {
            SemanticReadiness::PartialQualityCoverage {
                provenance: VerifiedSemanticProvenance::Local,
                coverage_ppm: 500_000,
            }
        }
        DegradationFailureKind::LexicalFallback | DegradationFailureKind::CacheMiss => {
            SemanticReadiness::IndexAbsent
        }
        DegradationFailureKind::UnverifiableEmbeddingSpace => SemanticReadiness::RemoteUnverified,
        DegradationFailureKind::CorruptIndex => SemanticReadiness::ManifestUnsafe,
        DegradationFailureKind::SemanticZeroSignal => {
            SemanticReadiness::IndexEmpty(ZeroSignalReason::NoUsableVectors)
        }
        DegradationFailureKind::HashControlGeneration => SemanticReadiness::IdentityMismatch,
    }
}

/// Map a [`SearchError`] to its canonical [`SemanticReadiness`] state without
/// formatting or inspecting opaque provider error bodies.
#[must_use]
pub fn semantic_readiness_for_search_error(
    error: &SearchError,
    mode: RequestMode,
    requested_topology: RetrievalTopology,
) -> SemanticReadiness {
    if matches!(mode, RequestMode::HashControl) {
        return SemanticReadiness::HashControl;
    }
    let tier = if matches!(requested_topology, RetrievalTopology::FastOnly) {
        ModelTier::Fast
    } else {
        ModelTier::Quality
    };
    match error {
        SearchError::ModelNotFound { .. } | SearchError::EmbedderUnavailable { .. } => {
            SemanticReadiness::ModelMissing { tier }
        }
        SearchError::ModelLoadFailed { .. } | SearchError::HashMismatch { .. } => {
            SemanticReadiness::ModelUnloadable { tier }
        }
        SearchError::DimensionMismatch { .. } | SearchError::IndexVersionMismatch { .. } => {
            SemanticReadiness::IdentityMismatch
        }
        SearchError::InvalidConfig { field, value, .. }
            if field == "semantic.vector_generation" && value == "legacy_hash" =>
        {
            SemanticReadiness::IdentityMismatch
        }
        other => {
            semantic_readiness_for_failure(classify_search_error(other), mode, requested_topology)
        }
    }
}

/// Map a [`ZeroSignalReason`] to its canonical [`SemanticReadiness`] state.
#[must_use]
pub const fn semantic_readiness_for_zero_signal(reason: ZeroSignalReason) -> SemanticReadiness {
    SemanticReadiness::IndexEmpty(reason)
}

/// Compute a canonical [`RecoveryPlan`] from a typed [`SemanticReadiness`]
/// state and trusted runtime context.
///
/// # Errors
///
/// Returns [`RecoveryContractError`] if the request topology, readiness state,
/// or acquisition authorization fails validation.
pub fn recovery_plan_for_readiness(
    state: &SemanticReadiness,
    request: RecoveryRequest,
    policy: &RecoveryPolicy,
    acquisition_target: Option<&ModelAcquisitionTarget>,
    evaluation_time_unix_seconds: u64,
) -> Result<RecoveryPlan, RecoveryContractError> {
    plan(TrustedRecoveryContext::new(
        state,
        request,
        policy,
        acquisition_target,
        evaluation_time_unix_seconds,
    ))
}

/// Compute a canonical [`RecoveryPlan`] for a [`SearchError`] without parsing
/// or formatting error strings.
///
/// # Errors
///
/// Returns [`RecoveryContractError`] if the request topology or acquisition
/// authorization is invalid.
pub fn recovery_plan_for_search_error(
    error: &SearchError,
    request: RecoveryRequest,
    policy: &RecoveryPolicy,
    acquisition_target: Option<&ModelAcquisitionTarget>,
    evaluation_time_unix_seconds: u64,
) -> Result<RecoveryPlan, RecoveryContractError> {
    let state =
        semantic_readiness_for_search_error(error, request.mode, request.requested_topology);
    recovery_plan_for_readiness(
        &state,
        request,
        policy,
        acquisition_target,
        evaluation_time_unix_seconds,
    )
}

/// Compute a canonical [`RecoveryPlan`] for a [`ZeroSignalReason`].
///
/// # Errors
///
/// Returns [`RecoveryContractError`] if the request topology is invalid.
pub fn recovery_plan_for_zero_signal(
    reason: ZeroSignalReason,
    request: RecoveryRequest,
    policy: &RecoveryPolicy,
    evaluation_time_unix_seconds: u64,
) -> Result<RecoveryPlan, RecoveryContractError> {
    let state = if matches!(request.mode, RequestMode::HashControl) {
        SemanticReadiness::HashControl
    } else {
        semantic_readiness_for_zero_signal(reason)
    };
    recovery_plan_for_readiness(&state, request, policy, None, evaluation_time_unix_seconds)
}

#[must_use]
pub fn synthetic_degradation_advice_fixture() -> Vec<DegradationAdvice> {
    [
        DegradationFailureKind::RefinementFailed,
        DegradationFailureKind::LexicalFallback,
        DegradationFailureKind::MissingQualityModel,
        DegradationFailureKind::UnverifiableEmbeddingSpace,
        DegradationFailureKind::Timeout,
        DegradationFailureKind::CorruptIndex,
        DegradationFailureKind::CacheMiss,
        DegradationFailureKind::SemanticZeroSignal,
        DegradationFailureKind::HashControlGeneration,
    ]
    .into_iter()
    .map(|failure| {
        DegradationAdvice::from_input(DegradationAdviceInput {
            failure,
            query: "authentication middleware",
            index_dir: Some(Path::new("/tmp/frankensearch-fixture/.frankensearch")),
            original_error: None,
            replay_command: None,
        })
    })
    .collect()
}

#[must_use]
const fn severity_for(failure: DegradationFailureKind) -> DegradationAdviceSeverity {
    match failure {
        DegradationFailureKind::RefinementFailed
        | DegradationFailureKind::MissingQualityModel
        | DegradationFailureKind::UnverifiableEmbeddingSpace
        | DegradationFailureKind::Timeout
        | DegradationFailureKind::CacheMiss
        | DegradationFailureKind::SemanticZeroSignal
        | DegradationFailureKind::HashControlGeneration => DegradationAdviceSeverity::Warn,
        DegradationFailureKind::LexicalFallback => DegradationAdviceSeverity::Info,
        DegradationFailureKind::CorruptIndex => DegradationAdviceSeverity::Error,
    }
}

fn next_actions_for(
    failure: DegradationFailureKind,
    index_dir: Option<&Path>,
) -> Vec<DegradationNextAction> {
    let index_dir = quoted_index_dir(index_dir);
    let with_index = |prefix: &str, suffix: &str| {
        index_dir
            .as_ref()
            .map(|index| format!("{prefix} --index-dir {index}{suffix}"))
    };
    match failure {
        DegradationFailureKind::RefinementFailed => vec![
            action(
                1,
                "degrade.action.keep_initial",
                "Keep displaying the initial result set; refinement failure is graceful.",
                None,
            ),
            action(
                2,
                "degrade.action.inspect_status",
                "Inspect model, index, and degraded-mode status before retrying.",
                with_index("fsfs status", " --format json"),
            ),
        ],
        DegradationFailureKind::LexicalFallback => vec![
            action(
                1,
                "degrade.action.verify_vector_index",
                "Verify the vector index and semantic embedder before depending on semantic scores.",
                with_index("fsfs doctor", " --format json"),
            ),
            action(
                2,
                "degrade.action.rebuild_vector_index",
                "Rebuild index artifacts in place when the semantic index is stale or unreadable.",
                with_index("fsfs index . --full", ""),
            ),
        ],
        DegradationFailureKind::MissingQualityModel => vec![
            action(
                1,
                "degrade.action.check_model_cache",
                "Check model cache path, revision, and offline/download settings.",
                Some("fsfs status --format json".to_owned()),
            ),
            action(
                2,
                "degrade.action.download_models",
                "Populate the configured model cache when quality refinement is required.",
                Some("fsfs download-models --verify".to_owned()),
            ),
        ],
        DegradationFailureKind::UnverifiableEmbeddingSpace => vec![
            action(
                1,
                "degrade.action.keep_lexical_only",
                "Keep lexical results, but do not admit semantic scores from an unidentified embedding space.",
                None,
            ),
            action(
                2,
                "degrade.action.configure_verified_identity",
                "Select a producer-authenticated immutable identity that matches the existing index, or use a verified local embedder.",
                None,
            ),
            action(
                3,
                "degrade.action.reindex_after_identity_change",
                "Rebuild in place only if you intentionally change the canonical embedding identity.",
                with_index("fsfs index . --full", ""),
            ),
        ],
        DegradationFailureKind::Timeout => vec![
            action(
                1,
                "degrade.action.use_fast_only",
                "Use fast-only results for latency-sensitive workflows.",
                Some("fsfs search --fast-only --format json -- '<query>'".to_owned()),
            ),
            action(
                2,
                "degrade.action.raise_quality_timeout",
                "Increase the quality timeout only after confirming the host has spare capacity.",
                Some("fsfs config set search.quality_timeout_ms 1000".to_owned()),
            ),
        ],
        DegradationFailureKind::CorruptIndex => vec![
            action(
                1,
                "degrade.action.stop_trusting_artifact",
                "Do not trust semantic hits from the unreadable artifact; use lexical fallback or rebuild.",
                None,
            ),
            action(
                2,
                "degrade.action.reindex_in_place",
                "Rebuild index artifacts in place from source content; no cleanup is required.",
                with_index("fsfs index . --full", ""),
            ),
        ],
        DegradationFailureKind::CacheMiss => vec![
            action(
                1,
                "degrade.action.verify_cache_key",
                "Verify index-dir, config source, and cache key before assuming there are no results.",
                with_index("fsfs status", " --format json"),
            ),
            action(
                2,
                "degrade.action.replay_search",
                "Replay the search with explicit index-dir and machine-readable output.",
                with_index("fsfs search", " --format json -- '<query>'"),
            ),
        ],
        DegradationFailureKind::SemanticZeroSignal => vec![
            action(
                1,
                "degrade.action.inspect_zero_signal_census",
                "Run doctor to see the typed zero-signal census (records, live, tombstones, usable vectors).",
                with_index("fsfs doctor", " --format json"),
            ),
            action(
                2,
                "degrade.action.rebuild_unusable_vectors",
                "Rebuild in place if the census shows live records without usable vectors (zero-norm or corrupt embeddings).",
                with_index("fsfs index . --full", ""),
            ),
        ],
        DegradationFailureKind::HashControlGeneration => vec![
            action(
                1,
                "degrade.action.do_not_treat_hash_as_semantic",
                "Do not treat this result set as semantic search; the published FSVI is a hash/fnv control artifact.",
                with_index("fsfs status", " --format json"),
            ),
            action(
                2,
                "degrade.action.rebuild_semantic_generation",
                "Rebuild with a verified semantic embedder; installing a model alone cannot repair hash vectors.",
                with_index("fsfs index . --full", ""),
            ),
        ],
    }
}

fn action(
    order: u16,
    reason_code: impl Into<String>,
    action: impl Into<String>,
    command: Option<String>,
) -> DegradationNextAction {
    DegradationNextAction {
        order,
        reason_code: reason_code.into(),
        action: action.into(),
        command,
    }
}

fn replay_command_for(query: &str, index_dir: Option<&Path>) -> String {
    let Some(index_dir) = quoted_index_dir(index_dir) else {
        return String::new();
    };
    let query = if query.is_empty() { "<query>" } else { query };
    let Some(query) = quote_shell_word(query, 64 * 1024) else {
        return String::new();
    };
    format!("fsfs search --index-dir {index_dir} --format json -- {query}")
}

fn quoted_index_dir(index_dir: Option<&Path>) -> Option<String> {
    match index_dir {
        Some(path) => {
            let text = path.to_str().filter(|text| !text.is_empty())?;
            quote_shell_word(text, 4 * 1024)
        }
        // Quote the expansion, and refuse an unset/empty variable before fsfs
        // starts. An absent path must not accidentally select the current root.
        None => Some("\"${FSFS_INDEX_DIR:?set FSFS_INDEX_DIR}\"".to_owned()),
    }
}

/// Render exactly one POSIX-shell argument. Reject rather than truncate: a
/// shortened path or query would operate on different data. Check before any
/// output allocation; quoting grows by at most four bytes per input byte.
fn quote_shell_word(text: &str, max_bytes: usize) -> Option<String> {
    if text.len() > max_bytes
        || text.chars().any(|ch| {
            ch.is_control()
                || matches!(ch, '\u{061c}' | '\u{200e}' | '\u{200f}'
                    | '\u{2028}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
        })
    {
        return None;
    }
    let mut quoted = String::with_capacity(text.len() + 2);
    quoted.push('\'');
    for ch in text.chars() {
        if ch == '\'' {
            quoted.push_str("'\\''");
        } else {
            quoted.push(ch);
        }
    }
    quoted.push('\'');
    Some(quoted)
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use frankensearch_core::SearchError;

    use frankensearch_core::config::ZeroSignalReason;

    use super::{
        AdviceOutputSurface, DEGRADATION_ADVICE_SCHEMA_VERSION, DegradationAdviceSeverity,
        DegradationFailureKind, advice_for_search_error, advice_for_zero_signal,
        synthetic_degradation_advice_fixture,
    };

    #[test]
    fn synthetic_fixture_covers_every_failure_kind_with_stable_codes() {
        let advice = synthetic_degradation_advice_fixture();
        let reasons = advice
            .iter()
            .map(|item| item.reason_code.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            reasons,
            vec![
                "degrade.advice.refinement_failed",
                "degrade.advice.lexical_fallback",
                "degrade.advice.quality_model_missing",
                "degrade.advice.embedding_space_unverifiable",
                "degrade.advice.timeout",
                "degrade.advice.index_corrupt",
                "degrade.advice.cache_miss",
                "degrade.advice.semantic_zero_signal",
                "degrade.advice.hash_control",
            ]
        );
        assert!(
            advice
                .iter()
                .all(|item| item.schema_version == DEGRADATION_ADVICE_SCHEMA_VERSION)
        );
        assert!(advice.iter().all(|item| !item.next_actions.is_empty()));
        assert!(
            advice
                .iter()
                .all(|item| item.replay_command.contains("--format json"))
        );
        assert!(advice.iter().all(|item| {
            item.output_surfaces
                .contains(&AdviceOutputSurface::CliJsonl)
                && item.output_surfaces.contains(&AdviceOutputSurface::CliToon)
        }));
    }

    #[test]
    fn zero_signal_advice_only_for_availability_failures() {
        let benign = [
            ZeroSignalReason::CallerRequestedZeroK,
            ZeroSignalReason::FilterEliminatedAll,
            ZeroSignalReason::NonFiniteQuery,
            ZeroSignalReason::ZeroNormQuery,
            ZeroSignalReason::NewlyCreatedEmpty,
            ZeroSignalReason::AllTombstoned,
            ZeroSignalReason::WalOnlyNoLiveRecords,
        ];
        for reason in benign {
            assert!(
                advice_for_zero_signal("auth", None, reason).is_none(),
                "benign reason {reason:?} must not produce operator advice"
            );
        }

        for reason in [
            ZeroSignalReason::NoUsableVectors,
            ZeroSignalReason::AnnReturnedEmptyDespiteUsableVectors,
        ] {
            let advice = advice_for_zero_signal(
                "auth",
                Some(Path::new("/tmp/project/.frankensearch")),
                reason,
            )
            .expect("availability failure must produce advice");
            assert_eq!(advice.failure, DegradationFailureKind::SemanticZeroSignal);
            assert_eq!(advice.severity, DegradationAdviceSeverity::Warn);
            assert_eq!(advice.reason_code, "degrade.advice.semantic_zero_signal");
            assert!(advice.preserves_initial_results);
            assert!(
                advice
                    .original_error
                    .as_deref()
                    .is_some_and(|text| text.starts_with(reason.reason_code())),
                "typed reason code must ride in original_error"
            );
            let commands = advice
                .next_actions
                .iter()
                .filter_map(|action| action.command.as_deref())
                .collect::<Vec<_>>()
                .join("\n");
            assert!(commands.contains("fsfs doctor"));
            assert!(!commands.to_ascii_lowercase().contains("delete"));
        }
    }

    #[test]
    fn search_error_classifier_preserves_original_error_text() {
        let error = SearchError::SearchTimeout {
            elapsed_ms: 750,
            budget_ms: 500,
        };
        let advice = advice_for_search_error("latency budget", None, &error);
        assert_eq!(advice.failure, DegradationFailureKind::Timeout);
        assert_eq!(advice.reason_code, "degrade.advice.timeout");
        assert!(
            advice
                .original_error
                .as_deref()
                .is_some_and(|text| text.contains("Search timed out"))
        );
        assert!(advice.preserves_initial_results);
    }

    #[test]
    fn candidate_path_miss_is_classified_as_cache_miss() {
        let error = SearchError::IndexCandidatesNotFound {
            paths: vec!["vector.fast.idx".into(), "vector.idx".into()],
        };
        let advice = advice_for_search_error("index discovery", None, &error);

        assert_eq!(advice.failure, DegradationFailureKind::CacheMiss);
    }

    #[test]
    fn hash_control_invalid_config_is_not_a_cache_miss() {
        let error = SearchError::InvalidConfig {
            field: "semantic.vector_generation".to_owned(),
            value: "legacy_hash".to_owned(),
            reason: "vector generation `fnv1a-256` is a hash control artifact, not semantic search"
                .to_owned(),
        };
        let advice = advice_for_search_error("ownership", Some(Path::new("/tmp/index")), &error);
        assert_eq!(
            advice.failure,
            DegradationFailureKind::HashControlGeneration
        );
        assert_eq!(advice.reason_code, "degrade.advice.hash_control");
        assert!(
            advice
                .next_actions
                .iter()
                .any(|action| action.action.contains("hash/fnv"))
        );
        assert!(!advice.next_actions.iter().any(|action| {
            action
                .command
                .as_deref()
                .is_some_and(|command| command.contains("download-models"))
        }));
    }

    #[test]
    fn unverifiable_embedding_space_keeps_only_lexical_results_and_requires_identity_proof() {
        let error = SearchError::UnverifiableRemoteSpace {
            producer: "cass-indexer".to_owned(),
            reason: "missing canonical model fingerprint".to_owned(),
        };
        let advice = advice_for_search_error(
            "semantic query",
            Some(Path::new("/tmp/project/.frankensearch")),
            &error,
        );

        assert_eq!(
            advice.failure,
            DegradationFailureKind::UnverifiableEmbeddingSpace
        );
        assert_eq!(
            advice.reason_code,
            "degrade.advice.embedding_space_unverifiable"
        );
        assert!(advice.preserves_initial_results);
        let actions = advice
            .next_actions
            .iter()
            .map(|action| {
                format!(
                    "{} {}",
                    action.action,
                    action.command.as_deref().unwrap_or_default()
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(actions.contains("lexical results"));
        assert!(actions.contains("producer-authenticated immutable identity"));
        assert!(actions.contains("only if you intentionally change"));
        assert!(actions.contains("fsfs index . --full --index-dir"));
        assert!(!actions.to_ascii_lowercase().contains("delete"));
    }

    #[test]
    fn corrupt_index_advice_uses_in_place_reindex_without_delete_instruction()
    -> serde_json::Result<()> {
        let error = SearchError::IndexCorrupted {
            path: PathBuf::from("/tmp/index.fsvi"),
            detail: "bad magic".to_owned(),
        };
        let advice = advice_for_search_error(
            "auth",
            Some(Path::new("/tmp/project/.frankensearch")),
            &error,
        );
        let rendered = serde_json::to_string(&advice)?;
        let action_text = advice
            .next_actions
            .iter()
            .flat_map(|next_action| {
                [
                    next_action.action.as_str(),
                    next_action.command.as_deref().unwrap_or_default(),
                ]
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(advice.failure, DegradationFailureKind::CorruptIndex);
        assert!(!advice.preserves_initial_results);
        assert!(rendered.contains("fsfs index . --full"));
        assert!(!action_text.to_ascii_lowercase().contains("delete"));
        Ok(())
    }

    #[test]
    fn typed_recovery_plan_covers_every_failure_and_zero_signal_state() {
        use frankensearch_core::recovery_plan::{
            InteractionPolicy, NetworkPolicy, RecoveryPolicy, RecoveryRequest, RequestMode,
        };
        use frankensearch_core::types::RetrievalTopology;

        let policy = RecoveryPolicy {
            interaction: InteractionPolicy::NonInteractive,
            network: NetworkPolicy::Offline,
            acquisition_authorization: None,
        };
        for mode in [RequestMode::ExplicitSemantic, RequestMode::Hybrid] {
            for topology in [
                RetrievalTopology::FastOnly,
                RetrievalTopology::QualityOnly,
                RetrievalTopology::FullProgressive,
            ] {
                let request = RecoveryRequest {
                    mode,
                    requested_topology: topology,
                };
                for fixture in synthetic_degradation_advice_fixture() {
                    let readiness =
                        super::semantic_readiness_for_failure(fixture.failure, mode, topology);
                    assert_eq!(fixture.semantic_readiness(mode, topology), readiness);
                    let plan = super::recovery_plan_for_readiness(
                        &readiness,
                        request,
                        &policy,
                        None,
                        1_750_000_000,
                    )
                    .expect("every DegradationFailureKind must produce a valid RecoveryPlan");
                    let advice_plan = fixture
                        .recovery_plan(request, &policy, None, 1_750_000_000)
                        .expect("DegradationAdvice::recovery_plan must match canonical plan");
                    assert_eq!(advice_plan, plan);
                    assert_eq!(plan.state(), &readiness);
                    assert_eq!(plan.mode(), mode);
                    assert_eq!(plan.requested_topology(), topology);
                }
                for reason in [
                    ZeroSignalReason::NewlyCreatedEmpty,
                    ZeroSignalReason::AllTombstoned,
                    ZeroSignalReason::WalOnlyNoLiveRecords,
                    ZeroSignalReason::NoUsableVectors,
                    ZeroSignalReason::AnnReturnedEmptyDespiteUsableVectors,
                    ZeroSignalReason::CallerRequestedZeroK,
                    ZeroSignalReason::FilterEliminatedAll,
                    ZeroSignalReason::NonFiniteQuery,
                    ZeroSignalReason::ZeroNormQuery,
                ] {
                    let plan = super::recovery_plan_for_zero_signal(
                        reason,
                        request,
                        &policy,
                        1_750_000_000,
                    )
                    .expect("every ZeroSignalReason must produce a valid RecoveryPlan");
                    assert_eq!(plan.state_code(), "recovery.state.index_empty");
                    assert!(!plan.semantic_available());
                }
            }
        }
    }
}

#[cfg(test)]
mod public_advice_tests {
    use super::*;
    use crate::output_schema::{
        OutputEnvelope, OutputMeta, SearchHitPayload, SearchOutputPhase, SearchPayload,
        decode_envelope_toon, encode_envelope_toon, output_error_from,
    };
    use crate::stream_protocol::{
        StreamEvent, StreamFrame, StreamResultEvent, decode_stream_frame_ndjson,
        decode_stream_frame_toon, encode_stream_frame_ndjson, encode_stream_frame_toon,
    };
    use std::fmt;
    use std::io;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const PRIVATE: &str = "private-degradation-provider-body";
    const TS: &str = "2026-10-13T19:00:00Z";

    #[derive(Clone)]
    struct Opaque {
        body: Arc<str>,
        formatted: Arc<AtomicUsize>,
        reject_format: bool,
    }

    impl fmt::Display for Opaque {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.formatted.fetch_add(1, Ordering::Relaxed);
            if self.reject_format {
                Err(fmt::Error)
            } else {
                formatter.write_str(&self.body)
            }
        }
    }

    impl fmt::Debug for Opaque {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            fmt::Display::fmt(self, formatter)
        }
    }

    impl std::error::Error for Opaque {}

    fn probe(reject_format: bool) -> Opaque {
        Opaque {
            body: format!("{PRIVATE}\n\u{001b}[2J{}", "x".repeat(1 << 20)).into(),
            formatted: Arc::new(AtomicUsize::new(0)),
            reject_format,
        }
    }

    fn failures(source: &Opaque) -> [SearchError; 5] {
        [
            SearchError::EmbeddingFailed {
                model: "test-embedder".to_owned(),
                source: Box::new(source.clone()),
            },
            SearchError::ModelLoadFailed {
                path: "/models/test/model.onnx".into(),
                source: Box::new(source.clone()),
            },
            SearchError::RerankFailed {
                model: "test-reranker".to_owned(),
                source: Box::new(source.clone()),
            },
            SearchError::Io(io::Error::new(
                io::ErrorKind::PermissionDenied,
                source.clone(),
            )),
            SearchError::SubsystemError {
                subsystem: PRIVATE,
                source: Box::new(source.clone()),
            },
        ]
    }

    fn assert_public(text: &str) {
        assert!(!text.contains(PRIVATE));
        assert!(!text.contains("[2J"));
        assert!(!text.contains("xxxxxxxxxxxxxxxx"));
        assert!(text.len() < 16 * 1024);
    }

    #[test]
    fn advice_never_formats_opaque_exceptions_even_when_formatting_would_fail() {
        for reject_format in [false, true] {
            let source = probe(reject_format);
            for error in failures(&source) {
                let advice =
                    advice_for_search_error("public query", Some(Path::new("/index")), &error);
                let expected_kind = if matches!(&error, SearchError::ModelLoadFailed { .. }) {
                    DegradationFailureKind::MissingQualityModel
                } else {
                    DegradationFailureKind::RefinementFailed
                };
                assert_eq!(advice.failure, expected_kind);
                assert_eq!(advice.reason_code, expected_kind.reason_code());
                assert!(advice.preserves_initial_results);
                assert_eq!(
                    advice.original_error,
                    Some(output_error_from(&error).message)
                );
                assert_public(&serde_json::to_string(&advice).unwrap());

                let policy = frankensearch_core::recovery_plan::RecoveryPolicy {
                    interaction:
                        frankensearch_core::recovery_plan::InteractionPolicy::NonInteractive,
                    network: frankensearch_core::recovery_plan::NetworkPolicy::Offline,
                    acquisition_authorization: None,
                };
                let plan = recovery_plan_for_search_error(
                    &error,
                    frankensearch_core::recovery_plan::RecoveryRequest {
                        mode: frankensearch_core::recovery_plan::RequestMode::Hybrid,
                        requested_topology:
                            frankensearch_core::types::RetrievalTopology::FullProgressive,
                    },
                    &policy,
                    None,
                    1_750_000_000,
                )
                .expect("recovery_plan_for_search_error succeeds without formatting opaque source");
                assert_public(&serde_json::to_string(&plan).unwrap());
            }
            assert_eq!(source.formatted.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn degraded_success_envelopes_and_result_streams_keep_hits_without_provider_bodies() {
        let source = probe(false);
        for error in failures(&source) {
            let advice = advice_for_search_error("public query", Some(Path::new("/index")), &error);
            let hit = SearchHitPayload {
                rank: 1,
                path: "retained.md".to_owned(),
                line: Some(7),
                score: 0.5,
                snippet: Some("original snippet".to_owned()),
                lexical_rank: Some(0),
                semantic_rank: Some(1),
                hash_rank: None,
                in_both_sources: true,
            };
            let payload = SearchPayload::new(
                "public query",
                SearchOutputPhase::RefinementFailed,
                3,
                vec![hit.clone()],
            )
            .with_degradation_advice([advice]);
            let envelope =
                OutputEnvelope::success(payload.clone(), OutputMeta::new("search", "json"), TS);
            let json = serde_json::to_string(&envelope).unwrap();
            assert_public(&json);
            let decoded: OutputEnvelope<SearchPayload> = serde_json::from_str(&json).unwrap();
            assert!(decoded.ok && decoded.error.is_none());
            assert_eq!(decoded.data.as_ref().unwrap().hits, vec![hit]);
            let toon = encode_envelope_toon(&envelope).unwrap();
            assert_public(&toon);
            let decoded: OutputEnvelope<SearchPayload> = decode_envelope_toon(&toon).unwrap();
            assert_eq!(decoded.data.as_ref(), Some(&payload));
            let frame = StreamFrame::new(
                "degraded-stream",
                1,
                TS,
                "search",
                StreamEvent::Result(StreamResultEvent {
                    rank: 1,
                    item: payload,
                }),
            );
            let ndjson = encode_stream_frame_ndjson(&frame).unwrap();
            assert_public(&ndjson);
            assert_eq!(ndjson.lines().count(), 1);
            let decoded: StreamFrame<SearchPayload> = decode_stream_frame_ndjson(&ndjson).unwrap();
            assert_eq!(decoded, frame);
            let toon = encode_stream_frame_toon(&frame).unwrap();
            assert_public(&toon);
            let decoded: StreamFrame<SearchPayload> = decode_stream_frame_toon(&toon).unwrap();
            assert_eq!(decoded, frame);
        }
        assert_eq!(source.formatted.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn explicit_constructor_does_not_bypass_the_error_boundary() {
        let source = probe(true);
        let error = SearchError::EmbeddingFailed {
            model: "test-embedder".to_owned(),
            source: Box::new(source.clone()),
        };
        let advice = DegradationAdvice::from_input(DegradationAdviceInput {
            failure: DegradationFailureKind::LexicalFallback,
            query: "public query",
            index_dir: None,
            original_error: Some(&error),
            replay_command: Some("trusted caller command"),
        });
        assert_eq!(advice.failure, DegradationFailureKind::LexicalFallback);
        assert_eq!(advice.replay_command, "trusted caller command");
        assert_eq!(
            advice.original_error.as_deref(),
            Some("Embedding inference failed for test-embedder.")
        );
        assert_eq!(source.formatted.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn shell_words_are_lossless_and_bounded_before_escaping() {
        assert_eq!(quote_shell_word("", 0).as_deref(), Some("''"));
        assert_eq!(quote_shell_word("a'b", 3).as_deref(), Some("'a'\\''b'"));
        assert_eq!(
            quote_shell_word("雪 $HOME *", 32).as_deref(),
            Some("'雪 $HOME *'")
        );
        assert!(quote_shell_word("é", 1).is_none());
        assert_eq!(quote_shell_word("é", 2).as_deref(), Some("'é'"));
        for unsafe_text in [
            "a\nb",
            "\0",
            "\t",
            "\u{001b}[2J",
            "\u{202e}",
            "\u{2066}",
            "\u{061c}",
        ] {
            assert!(quote_shell_word(unsafe_text, 4096).is_none());
        }
        let worst = "'".repeat(64 * 1024);
        assert_eq!(
            quote_shell_word(&worst, 64 * 1024).unwrap().len(),
            4 * worst.len() + 2
        );
        assert!(quote_shell_word(&format!("{worst}x"), 64 * 1024).is_none());
    }

    #[cfg(unix)]
    fn shell(command: &str, index_dir: Option<&str>) -> std::process::Output {
        // Execute only an inert argv-recording shell function, never fsfs or a
        // suggested mutation. The adversarial inputs also use only printf.
        let mut process = std::process::Command::new("/bin/sh");
        process
            .arg("-c")
            .arg(format!("fsfs() {{ printf '%s\\000' \"$@\"; }}\n{command}"))
            .env_remove("FSFS_INDEX_DIR")
            .stdin(std::process::Stdio::null());
        if let Some(index_dir) = index_dir {
            process.env("FSFS_INDEX_DIR", index_dir);
        }
        process.output().unwrap()
    }

    #[cfg(unix)]
    fn argv(output: &std::process::Output) -> Vec<String> {
        assert!(output.status.success());
        assert!(output.stderr.is_empty());
        output
            .stdout
            .strip_suffix(&[0])
            .expect("terminated argv")
            .split(|byte| *byte == 0)
            .map(|word| String::from_utf8(word.to_vec()).unwrap())
            .collect()
    }

    #[cfg(unix)]
    #[test]
    fn replay_round_trips_actual_posix_arguments_without_command_substitution() {
        let values = [
            "ordinary words",
            "owner's index",
            "$(printf INJECTED >&2)",
            "; printf INJECTED >&2; #",
            "`printf INJECTED >&2`",
            "雪 $HOME * ? [a] \\",
        ];
        for path in values {
            for query in values.into_iter().chain(["--apply", ""]) {
                let command = replay_command_for(query, Some(Path::new(path)));
                let observed = argv(&shell(&command, None));
                assert_eq!(
                    observed,
                    [
                        "search",
                        "--index-dir",
                        path,
                        "--format",
                        "json",
                        "--",
                        if query.is_empty() { "<query>" } else { query },
                    ]
                );
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn every_generated_action_quotes_its_index_and_query_placeholder() {
        let path = "owner's index/$(printf INJECTED >&2); *";
        let mut checked = 0;
        for fixture in synthetic_degradation_advice_fixture() {
            let advice = DegradationAdvice::from_input(DegradationAdviceInput {
                failure: fixture.failure,
                query: "public query",
                index_dir: Some(Path::new(path)),
                original_error: None,
                replay_command: None,
            });
            for command in advice
                .next_actions
                .iter()
                .filter_map(|action| action.command.as_deref())
            {
                let observed = argv(&shell(command, None));
                if let Some(index) = observed.iter().position(|word| word == "--index-dir") {
                    assert_eq!(observed[index + 1], path);
                    checked += 1;
                }
                if observed[0] == "search" {
                    assert_eq!(observed.last().unwrap(), "<query>");
                }
            }
        }
        assert_eq!(checked, 11);
    }

    #[cfg(unix)]
    #[test]
    fn absent_index_requires_a_nonempty_environment_without_word_splitting() {
        let command = replay_command_for("two words", None);
        for index in [None, Some("")] {
            let output = shell(&command, index);
            assert!(!output.status.success());
            assert!(
                output.stdout.is_empty(),
                "fsfs must not start without a selected path"
            );
        }
        let path = "space dir/$(printf INJECTED >&2)";
        assert_eq!(
            argv(&shell(&command, Some(path))),
            [
                "search",
                "--index-dir",
                path,
                "--format",
                "json",
                "--",
                "two words",
            ]
        );
    }

    #[test]
    fn unrepresentable_inputs_omit_commands_instead_of_retargeting_or_truncating() {
        for path in [
            String::new(),
            "x".repeat(4097),
            "index\nsecond line".to_owned(),
            "\u{202e}index".to_owned(),
        ] {
            let advice = DegradationAdvice::from_input(DegradationAdviceInput {
                failure: DegradationFailureKind::HashControlGeneration,
                query: "public query",
                index_dir: Some(Path::new(&path)),
                original_error: None,
                replay_command: None,
            });
            assert!(advice.replay_command.is_empty());
            assert!(
                advice
                    .next_actions
                    .iter()
                    .all(|action| action.command.is_none())
            );
            assert!(!advice.operator_summary.is_empty());
        }
        for query in ["x".repeat(65537), "query\nsecond line".to_owned()] {
            assert!(replay_command_for(&query, Some(Path::new("/index"))).is_empty());
        }
        let boundary_path = "é".repeat(2048);
        assert!(quoted_index_dir(Some(Path::new(&boundary_path))).is_some());
        assert!(quoted_index_dir(Some(Path::new(&format!("{boundary_path}x")))).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_paths_are_not_lossily_rendered_as_another_index() {
        use std::os::unix::ffi::OsStringExt;
        let path = std::path::PathBuf::from(std::ffi::OsString::from_vec(b"/index/\xff".to_vec()));
        assert!(quoted_index_dir(Some(&path)).is_none());
        assert!(replay_command_for("public query", Some(&path)).is_empty());
    }
}
