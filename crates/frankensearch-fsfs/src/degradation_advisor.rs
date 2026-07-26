//! Typed operator advice for graceful search degradation.

use std::path::Path;

use frankensearch_core::SearchError;
use serde::{Deserialize, Serialize};

pub const DEGRADATION_ADVICE_SCHEMA_VERSION: &str = "fsfs.degradation.advice.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DegradationFailureKind {
    RefinementFailed,
    LexicalFallback,
    MissingQualityModel,
    Timeout,
    CorruptIndex,
    CacheMiss,
}

impl DegradationFailureKind {
    #[must_use]
    pub const fn reason_code(self) -> &'static str {
        match self {
            Self::RefinementFailed => "degrade.advice.refinement_failed",
            Self::LexicalFallback => "degrade.advice.lexical_fallback",
            Self::MissingQualityModel => "degrade.advice.quality_model_missing",
            Self::Timeout => "degrade.advice.timeout",
            Self::CorruptIndex => "degrade.advice.index_corrupt",
            Self::CacheMiss => "degrade.advice.cache_miss",
        }
    }

    #[must_use]
    pub const fn summary(self) -> &'static str {
        match self {
            Self::RefinementFailed => "quality refinement failed; initial results remain usable",
            Self::LexicalFallback => "semantic retrieval fell back to lexical search",
            Self::MissingQualityModel => "quality model unavailable; refinement skipped",
            Self::Timeout => "quality stage exceeded its latency budget",
            Self::CorruptIndex => "index artifact could not be read safely",
            Self::CacheMiss => "expected cache artifact was missing or stale",
        }
    }

    #[must_use]
    pub const fn preserves_initial_results(self) -> bool {
        match self {
            Self::RefinementFailed
            | Self::LexicalFallback
            | Self::MissingQualityModel
            | Self::Timeout
            | Self::CacheMiss => true,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_error: Option<String>,
    pub preserves_initial_results: bool,
    pub next_actions: Vec<DegradationNextAction>,
    pub replay_command: String,
    pub output_surfaces: Vec<AdviceOutputSurface>,
}

#[derive(Debug, Clone, Copy)]
pub struct DegradationAdviceInput<'a> {
    pub failure: DegradationFailureKind,
    pub query: &'a str,
    pub index_dir: Option<&'a Path>,
    pub original_error: Option<&'a SearchError>,
    pub replay_command: Option<&'a str>,
}

impl DegradationAdvice {
    #[must_use]
    pub fn from_input(input: DegradationAdviceInput<'_>) -> Self {
        let failure = input.failure;
        let original_error = input.original_error.map(ToString::to_string);
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

#[must_use]
pub const fn classify_search_error(error: &SearchError) -> DegradationFailureKind {
    match error {
        SearchError::SearchTimeout { .. } => DegradationFailureKind::Timeout,
        SearchError::IndexCorrupted { .. }
        | SearchError::IndexVersionMismatch { .. }
        | SearchError::DimensionMismatch { .. } => DegradationFailureKind::CorruptIndex,
        SearchError::EmbedderUnavailable { .. }
        | SearchError::ModelNotFound { .. }
        | SearchError::ModelLoadFailed { .. } => DegradationFailureKind::MissingQualityModel,
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

#[must_use]
pub fn synthetic_degradation_advice_fixture() -> Vec<DegradationAdvice> {
    [
        DegradationFailureKind::RefinementFailed,
        DegradationFailureKind::LexicalFallback,
        DegradationFailureKind::MissingQualityModel,
        DegradationFailureKind::Timeout,
        DegradationFailureKind::CorruptIndex,
        DegradationFailureKind::CacheMiss,
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
        | DegradationFailureKind::Timeout
        | DegradationFailureKind::CacheMiss => DegradationAdviceSeverity::Warn,
        DegradationFailureKind::LexicalFallback => DegradationAdviceSeverity::Info,
        DegradationFailureKind::CorruptIndex => DegradationAdviceSeverity::Error,
    }
}

fn next_actions_for(
    failure: DegradationFailureKind,
    index_dir: Option<&Path>,
) -> Vec<DegradationNextAction> {
    let index_dir = index_dir
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "$FSFS_INDEX_DIR".to_owned());
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
                Some(format!("fsfs status --index-dir {index_dir} --format json")),
            ),
        ],
        DegradationFailureKind::LexicalFallback => vec![
            action(
                1,
                "degrade.action.verify_vector_index",
                "Verify the vector index and semantic embedder before depending on semantic scores.",
                Some(format!("fsfs doctor --index-dir {index_dir} --format json")),
            ),
            action(
                2,
                "degrade.action.rebuild_vector_index",
                "Rebuild index artifacts in place when the semantic index is stale or unreadable.",
                Some(format!("fsfs index --full --index-dir {index_dir} .")),
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
        DegradationFailureKind::Timeout => vec![
            action(
                1,
                "degrade.action.use_fast_only",
                "Use fast-only results for latency-sensitive workflows.",
                Some("fsfs search --fast-only --format json -- <query>".to_owned()),
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
                Some(format!("fsfs index --full --index-dir {index_dir} .")),
            ),
        ],
        DegradationFailureKind::CacheMiss => vec![
            action(
                1,
                "degrade.action.verify_cache_key",
                "Verify index-dir, config source, and cache key before assuming there are no results.",
                Some(format!("fsfs status --index-dir {index_dir} --format json")),
            ),
            action(
                2,
                "degrade.action.replay_search",
                "Replay the search with explicit index-dir and machine-readable output.",
                Some(format!(
                    "fsfs search --index-dir {index_dir} --format json -- <query>"
                )),
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
    let index_dir = index_dir
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "$FSFS_INDEX_DIR".to_owned());
    let query = if query.is_empty() { "<query>" } else { query };
    format!("fsfs search --index-dir {index_dir} --format json -- {query}")
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use frankensearch_core::SearchError;

    use super::{
        AdviceOutputSurface, DEGRADATION_ADVICE_SCHEMA_VERSION, DegradationFailureKind,
        advice_for_search_error, synthetic_degradation_advice_fixture,
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
                "degrade.advice.timeout",
                "degrade.advice.index_corrupt",
                "degrade.advice.cache_miss",
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
        assert_eq!(advice.reason_code, "degrade.advice.cache_miss");
        assert!(
            advice
                .original_error
                .as_deref()
                .is_some_and(|text| text.contains("vector.fast.idx"))
        );
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
        assert!(rendered.contains("fsfs index --full"));
        assert!(!action_text.to_ascii_lowercase().contains("delete"));
        Ok(())
    }
}
