//! CLI-mode E2E scenario and diagnostic artifact contracts.
//!
//! This module codifies the fsfs CLI end-to-end lane for:
//! - index/search/explain/degrade scenarios,
//! - structured diagnostic artifacts,
//! - deterministic replay guidance.

use std::collections::BTreeMap;

use frankensearch_core::{
    ArtifactEmissionInput, ArtifactEntry, ClockMode, Correlation, DeterminismTier,
    E2E_ARTIFACT_ARTIFACTS_INDEX_JSON, E2E_ARTIFACT_REPLAY_COMMAND_TXT,
    E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL, E2E_SCHEMA_EVENT, E2E_SCHEMA_MANIFEST, E2E_SCHEMA_REPLAY,
    E2eEnvelope, E2eEventType, E2eOutcome, E2eSeverity, EventBody, ExitStatus, ManifestBody,
    ModelVersion, Platform, ReplayBody, ReplayEventType, Suite, build_artifact_entries,
    render_artifacts_index, sha256_checksum, validate_event_envelope, validate_manifest_envelope,
};
use serde::{Deserialize, Serialize};

/// Schema discriminator for fsfs CLI E2E contracts.
pub const CLI_E2E_SCHEMA_VERSION: &str = "fsfs.cli.e2e.v1";

/// Stable reason code for scenario start events.
pub const CLI_E2E_REASON_SCENARIO_START: &str = "e2e.cli.scenario_start";
/// Stable reason code for successful scenario assertions.
pub const CLI_E2E_REASON_SCENARIO_PASS: &str = "e2e.cli.scenario_pass";
/// Stable reason code for degraded scenario assertions.
pub const CLI_E2E_REASON_SCENARIO_DEGRADE: &str = "e2e.cli.degrade_path";
/// Stable reason code for permission-denied chaos assertions.
pub const CLI_E2E_REASON_FILESYSTEM_PERMISSION_DENIED: &str = "e2e.fs_permission.denied";
/// Stable reason code for symlink-loop chaos assertions.
pub const CLI_E2E_REASON_FILESYSTEM_SYMLINK_LOOP: &str = "e2e.fs_symlink.loop";
/// Stable reason code for mount-boundary chaos assertions.
pub const CLI_E2E_REASON_FILESYSTEM_MOUNT_BOUNDARY: &str = "e2e.fs_mount.boundary";
/// Stable reason code for giant-log chaos assertions.
pub const CLI_E2E_REASON_FILESYSTEM_GIANT_LOG_SKIPPED: &str = "e2e.fs_skip.giant_log";
/// Stable reason code for binary-blob chaos assertions.
pub const CLI_E2E_REASON_FILESYSTEM_BINARY_BLOB_SKIPPED: &str = "e2e.fs_skip.binary_blob";

/// CLI scenario category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CliE2eScenarioKind {
    Index,
    Search,
    Explain,
    Degrade,
}

impl std::fmt::Display for CliE2eScenarioKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Index => write!(f, "index"),
            Self::Search => write!(f, "search"),
            Self::Explain => write!(f, "explain"),
            Self::Degrade => write!(f, "degrade"),
        }
    }
}

/// One deterministic CLI E2E scenario definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CliE2eScenario {
    /// Stable scenario identifier.
    pub id: String,
    /// Scenario kind.
    pub kind: CliE2eScenarioKind,
    /// Command argv vector (without binary name).
    pub args: Vec<String>,
    /// Expected terminal exit code.
    pub expected_exit_code: i32,
    /// Stable expected reason code.
    pub expected_reason_code: String,
    /// Human-readable scenario summary.
    pub summary: String,
}

/// Run-level configuration shared across scenario executions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CliE2eRunConfig {
    /// Stable run identifier (ULID format).
    pub run_id: String,
    /// RFC 3339 UTC timestamp.
    pub ts: String,
    /// Deterministic scenario seed.
    pub seed: u64,
    /// Config fingerprint used for this run.
    pub config_hash: String,
    /// Platform metadata.
    pub platform: Platform,
}

impl Default for CliE2eRunConfig {
    fn default() -> Self {
        Self {
            run_id: "01JABCD3EFGHJKMNPQRSTVWXYZ".to_owned(),
            ts: "2026-02-14T00:00:00Z".to_owned(),
            seed: 42,
            config_hash: "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                .to_owned(),
            platform: Platform {
                os: std::env::consts::OS.to_owned(),
                arch: std::env::consts::ARCH.to_owned(),
                rustc: "nightly-2026-02-14".to_owned(),
            },
        }
    }
}

/// Structured artifact bundle emitted for one scenario run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CliE2eArtifactBundle {
    /// Contract schema version for this bundle.
    pub schema_version: String,
    /// Scenario metadata.
    pub scenario: CliE2eScenario,
    /// Manifest artifact envelope.
    pub manifest: E2eEnvelope<ManifestBody>,
    /// Structured events JSONL lines represented as envelopes.
    pub events: Vec<E2eEnvelope<EventBody>>,
    /// Replay stream entries.
    pub replay: Vec<E2eEnvelope<ReplayBody>>,
    /// Deterministic replay command for triage.
    pub replay_command: String,
}

impl CliE2eArtifactBundle {
    /// Build a deterministic CLI E2E artifact bundle for one scenario.
    #[must_use]
    pub fn build(
        config: &CliE2eRunConfig,
        scenario: &CliE2eScenario,
        exit_status: ExitStatus,
    ) -> Self {
        let event_bodies = scenario_event_bodies(scenario, exit_status);
        let events: Vec<E2eEnvelope<EventBody>> = event_bodies
            .iter()
            .enumerate()
            .map(|(index, event)| {
                E2eEnvelope::new(
                    E2E_SCHEMA_EVENT,
                    &config.run_id,
                    &config.ts,
                    EventBody {
                        correlation: Correlation {
                            event_id: format!("{}-evt-{index:02}", scenario.id),
                            root_request_id: format!("{}-root", scenario.id),
                            parent_event_id: None,
                        },
                        ..event.clone()
                    },
                )
            })
            .collect();

        let replay = vec![E2eEnvelope::new(
            E2E_SCHEMA_REPLAY,
            &config.run_id,
            &config.ts,
            ReplayBody {
                replay_type: ReplayEventType::Query,
                offset_ms: 0,
                seq: 1,
                payload: serde_json::json!({
                    "scenario_id": scenario.id,
                    "scenario_kind": scenario.kind,
                    "args": scenario.args,
                    "expected_reason_code": scenario.expected_reason_code,
                }),
            },
        )];

        let replay_command = replay_command_for_scenario(scenario);
        let manifest = E2eEnvelope::new(
            E2E_SCHEMA_MANIFEST,
            &config.run_id,
            &config.ts,
            ManifestBody {
                suite: Suite::Fsfs,
                determinism_tier: DeterminismTier::Semantic,
                seed: config.seed,
                config_hash: config.config_hash.clone(),
                index_version: Some("fsvi-v1".to_owned()),
                model_versions: vec![
                    ModelVersion {
                        name: "potion-128M".to_owned(),
                        revision: "contract-v1".to_owned(),
                        digest: None,
                    },
                    ModelVersion {
                        name: "all-MiniLM-L6-v2".to_owned(),
                        revision: "contract-v1".to_owned(),
                        digest: None,
                    },
                ],
                platform: config.platform.clone(),
                clock_mode: ClockMode::Simulated,
                tie_break_policy: "doc_id_lexical".to_owned(),
                artifacts: artifact_entries(&replay_command, &events),
                duration_ms: 250,
                exit_status,
            },
        );

        Self {
            schema_version: CLI_E2E_SCHEMA_VERSION.to_owned(),
            scenario: scenario.clone(),
            manifest,
            events,
            replay,
            replay_command,
        }
    }

    /// Validate manifest + event envelope invariants.
    ///
    /// # Errors
    ///
    /// Returns a stringified validation error when either the manifest or any
    /// event envelope violates the unified E2E artifact contract.
    pub fn validate(&self) -> Result<(), String> {
        validate_manifest_envelope(&self.manifest).map_err(|err| err.to_string())?;
        for event in &self.events {
            validate_event_envelope(event).map_err(|err| err.to_string())?;
        }
        Ok(())
    }
}

/// Deterministic default scenario catalog for CLI mode E2E.
#[must_use]
pub fn default_cli_e2e_scenarios() -> Vec<CliE2eScenario> {
    vec![
        CliE2eScenario {
            id: "cli-index-baseline".to_owned(),
            kind: CliE2eScenarioKind::Index,
            args: vec![
                "index".to_owned(),
                "--roots".to_owned(),
                "/tmp/fsfs-e2e-corpus".to_owned(),
                "--format".to_owned(),
                "json".to_owned(),
            ],
            expected_exit_code: 0,
            expected_reason_code: CLI_E2E_REASON_SCENARIO_PASS.to_owned(),
            summary: "Index fixture corpus and emit structured diagnostics".to_owned(),
        },
        CliE2eScenario {
            id: "cli-search-stream".to_owned(),
            kind: CliE2eScenarioKind::Search,
            args: vec![
                "search".to_owned(),
                "rust async cancellation".to_owned(),
                "--stream".to_owned(),
                "--format".to_owned(),
                "jsonl".to_owned(),
            ],
            expected_exit_code: 0,
            expected_reason_code: CLI_E2E_REASON_SCENARIO_PASS.to_owned(),
            summary: "Run progressive search stream with NDJSON framing".to_owned(),
        },
        CliE2eScenario {
            id: "cli-explain-hit".to_owned(),
            kind: CliE2eScenarioKind::Explain,
            args: vec![
                "explain".to_owned(),
                "--format".to_owned(),
                "toon".to_owned(),
                "--query".to_owned(),
                "embedding budget".to_owned(),
            ],
            expected_exit_code: 0,
            expected_reason_code: CLI_E2E_REASON_SCENARIO_PASS.to_owned(),
            summary: "Emit explainability payload in TOON format".to_owned(),
        },
        CliE2eScenario {
            id: "cli-degrade-path".to_owned(),
            kind: CliE2eScenarioKind::Degrade,
            args: vec![
                "search".to_owned(),
                "latency fallback lane".to_owned(),
                "--profile".to_owned(),
                "degraded".to_owned(),
                "--format".to_owned(),
                "json".to_owned(),
            ],
            expected_exit_code: 0,
            expected_reason_code: CLI_E2E_REASON_SCENARIO_DEGRADE.to_owned(),
            summary: "Exercise degraded query path and reason-coded status".to_owned(),
        },
    ]
}

/// Build bundles for all default scenarios.
#[must_use]
pub fn build_default_cli_e2e_bundles(config: &CliE2eRunConfig) -> Vec<CliE2eArtifactBundle> {
    default_cli_e2e_scenarios()
        .into_iter()
        .map(|scenario| {
            let exit_status = scenario_exit_status(&scenario);
            CliE2eArtifactBundle::build(config, &scenario, exit_status)
        })
        .collect()
}

/// Deterministic filesystem-chaos scenario catalog for CLI mode E2E.
///
/// These scenarios encode expected reason-code behavior for filesystem edge
/// cases (permissions, symlink loops, mount boundaries, and high-cost skips).
#[must_use]
pub fn default_cli_e2e_filesystem_chaos_scenarios() -> Vec<CliE2eScenario> {
    vec![
        CliE2eScenario {
            id: "cli-chaos-permission-denied".to_owned(),
            kind: CliE2eScenarioKind::Degrade,
            args: vec![
                "index".to_owned(),
                "--roots".to_owned(),
                "/tmp/fsfs-chaos/permission-denied".to_owned(),
                "--format".to_owned(),
                "json".to_owned(),
            ],
            expected_exit_code: 0,
            expected_reason_code: CLI_E2E_REASON_FILESYSTEM_PERMISSION_DENIED.to_owned(),
            summary: "Handle unreadable directories with deterministic reason codes".to_owned(),
        },
        CliE2eScenario {
            id: "cli-chaos-symlink-loop".to_owned(),
            kind: CliE2eScenarioKind::Degrade,
            args: vec![
                "index".to_owned(),
                "--roots".to_owned(),
                "/tmp/fsfs-chaos/symlink-loop".to_owned(),
                "--format".to_owned(),
                "json".to_owned(),
            ],
            expected_exit_code: 0,
            expected_reason_code: CLI_E2E_REASON_FILESYSTEM_SYMLINK_LOOP.to_owned(),
            summary: "Detect symlink loops without non-deterministic traversal".to_owned(),
        },
        CliE2eScenario {
            id: "cli-chaos-mount-boundary".to_owned(),
            kind: CliE2eScenarioKind::Degrade,
            args: vec![
                "index".to_owned(),
                "--roots".to_owned(),
                "/tmp/fsfs-chaos/mount-boundary".to_owned(),
                "--format".to_owned(),
                "json".to_owned(),
            ],
            expected_exit_code: 0,
            expected_reason_code: CLI_E2E_REASON_FILESYSTEM_MOUNT_BOUNDARY.to_owned(),
            summary: "Respect configured mount boundaries during discovery".to_owned(),
        },
        CliE2eScenario {
            id: "cli-chaos-giant-log-skip".to_owned(),
            kind: CliE2eScenarioKind::Degrade,
            args: vec![
                "index".to_owned(),
                "--roots".to_owned(),
                "/tmp/fsfs-chaos/giant-logs".to_owned(),
                "--format".to_owned(),
                "json".to_owned(),
            ],
            expected_exit_code: 0,
            expected_reason_code: CLI_E2E_REASON_FILESYSTEM_GIANT_LOG_SKIPPED.to_owned(),
            summary: "Skip giant log artifacts with explicit degradation reason".to_owned(),
        },
        CliE2eScenario {
            id: "cli-chaos-binary-blob-skip".to_owned(),
            kind: CliE2eScenarioKind::Degrade,
            args: vec![
                "index".to_owned(),
                "--roots".to_owned(),
                "/tmp/fsfs-chaos/binary-blobs".to_owned(),
                "--format".to_owned(),
                "json".to_owned(),
            ],
            expected_exit_code: 0,
            expected_reason_code: CLI_E2E_REASON_FILESYSTEM_BINARY_BLOB_SKIPPED.to_owned(),
            summary: "Skip binary blobs with deterministic policy evidence".to_owned(),
        },
    ]
}

/// Build bundles for the filesystem-chaos scenario catalog.
#[must_use]
pub fn build_cli_e2e_filesystem_chaos_bundles(
    config: &CliE2eRunConfig,
) -> Vec<CliE2eArtifactBundle> {
    default_cli_e2e_filesystem_chaos_scenarios()
        .into_iter()
        .map(|scenario| {
            let exit_status = scenario_exit_status(&scenario);
            CliE2eArtifactBundle::build(config, &scenario, exit_status)
        })
        .collect()
}

#[must_use]
const fn scenario_exit_status(scenario: &CliE2eScenario) -> ExitStatus {
    if scenario.expected_exit_code == 0 {
        ExitStatus::Pass
    } else {
        ExitStatus::Fail
    }
}

fn artifact_entries(replay_command: &str, events: &[E2eEnvelope<EventBody>]) -> Vec<ArtifactEntry> {
    let structured_events_jsonl = render_events_jsonl(events);
    #[allow(clippy::cast_possible_truncation)]
    let line_count = u64::try_from(events.len()).expect("event count must fit in u64");
    let mut entries = build_artifact_entries([
        ArtifactEmissionInput {
            file: E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL,
            bytes: structured_events_jsonl.as_bytes(),
            line_count: Some(line_count),
        },
        ArtifactEmissionInput {
            file: E2E_ARTIFACT_REPLAY_COMMAND_TXT,
            bytes: replay_command.as_bytes(),
            line_count: None,
        },
    ])
    .expect("cli e2e manifest artifacts must satisfy contract");
    let artifacts_index_json =
        render_artifacts_index(&entries).expect("artifacts index payload must render");
    entries.push(ArtifactEntry {
        file: E2E_ARTIFACT_ARTIFACTS_INDEX_JSON.to_owned(),
        checksum: sha256_checksum(artifacts_index_json.as_bytes()),
        line_count: None,
    });
    entries.sort_by(|left, right| left.file.cmp(&right.file));
    entries
}

fn render_events_jsonl(events: &[E2eEnvelope<EventBody>]) -> String {
    events
        .iter()
        .map(|event| serde_json::to_string(event).expect("event envelope serialization must work"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn scenario_event_bodies(scenario: &CliE2eScenario, exit_status: ExitStatus) -> Vec<EventBody> {
    let mut metrics = BTreeMap::new();
    let argv_len = u32::try_from(scenario.args.len()).unwrap_or(u32::MAX);
    metrics.insert("argv_len".to_owned(), f64::from(argv_len));

    let assertion_outcome = if matches!(exit_status, ExitStatus::Pass) {
        E2eOutcome::Pass
    } else {
        E2eOutcome::Fail
    };
    let assertion_reason = scenario.expected_reason_code.as_str();

    vec![
        EventBody {
            event_type: E2eEventType::E2eStart,
            correlation: empty_correlation(),
            severity: E2eSeverity::Info,
            lane_id: None,
            oracle_id: None,
            outcome: None,
            reason_code: Some(CLI_E2E_REASON_SCENARIO_START.to_owned()),
            context: Some(format!("starting {}", scenario.id)),
            metrics: None,
        },
        EventBody {
            event_type: E2eEventType::LaneStart,
            correlation: empty_correlation(),
            severity: E2eSeverity::Info,
            lane_id: Some(scenario.kind.to_string()),
            oracle_id: None,
            outcome: None,
            reason_code: Some(CLI_E2E_REASON_SCENARIO_START.to_owned()),
            context: Some(scenario.summary.clone()),
            metrics: Some(metrics),
        },
        EventBody {
            event_type: E2eEventType::Assertion,
            correlation: empty_correlation(),
            severity: if matches!(assertion_outcome, E2eOutcome::Pass) {
                E2eSeverity::Info
            } else {
                E2eSeverity::Warn
            },
            lane_id: None,
            oracle_id: Some("cli.exit_status".to_owned()),
            outcome: Some(assertion_outcome),
            reason_code: Some(assertion_reason.to_owned()),
            context: Some(format!("expected_exit={}", scenario.expected_exit_code)),
            metrics: None,
        },
        EventBody {
            event_type: E2eEventType::LaneEnd,
            correlation: empty_correlation(),
            severity: E2eSeverity::Info,
            lane_id: Some(scenario.kind.to_string()),
            oracle_id: None,
            outcome: Some(assertion_outcome),
            reason_code: Some(assertion_reason.to_owned()),
            context: None,
            metrics: None,
        },
        EventBody {
            event_type: E2eEventType::E2eEnd,
            correlation: empty_correlation(),
            severity: if matches!(assertion_outcome, E2eOutcome::Pass) {
                E2eSeverity::Info
            } else {
                E2eSeverity::Warn
            },
            lane_id: None,
            oracle_id: None,
            outcome: Some(assertion_outcome),
            reason_code: Some(assertion_reason.to_owned()),
            context: None,
            metrics: None,
        },
    ]
}

const fn empty_correlation() -> Correlation {
    Correlation {
        event_id: String::new(),
        root_request_id: String::new(),
        parent_event_id: None,
    }
}

/// Deterministic replay command for a single scenario.
#[must_use]
pub fn replay_command_for_scenario(scenario: &CliE2eScenario) -> String {
    let test_target = if scenario.id.starts_with("cli-chaos-") {
        "filesystem_chaos"
    } else {
        "cli_e2e_contract"
    };
    format!(
        "cargo test -p frankensearch-fsfs --test {test_target} -- --nocapture --exact scenario_{}",
        scenario.id.replace('-', "_")
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use frankensearch_core::{
        ArtifactEntry, E2E_ARTIFACT_ARTIFACTS_INDEX_JSON, E2E_ARTIFACT_REPLAY_COMMAND_TXT,
        E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL, E2eOutcome, render_artifacts_index, sha256_checksum,
    };

    use super::{
        CLI_E2E_REASON_FILESYSTEM_BINARY_BLOB_SKIPPED, CLI_E2E_REASON_FILESYSTEM_GIANT_LOG_SKIPPED,
        CLI_E2E_REASON_FILESYSTEM_MOUNT_BOUNDARY, CLI_E2E_REASON_FILESYSTEM_PERMISSION_DENIED,
        CLI_E2E_REASON_FILESYSTEM_SYMLINK_LOOP, CLI_E2E_SCHEMA_VERSION, CliE2eRunConfig,
        CliE2eScenarioKind, ExitStatus, build_cli_e2e_filesystem_chaos_bundles,
        build_default_cli_e2e_bundles, default_cli_e2e_filesystem_chaos_scenarios,
        default_cli_e2e_scenarios,
    };

    #[test]
    fn scenario_catalog_covers_index_search_explain_degrade() {
        let scenarios = default_cli_e2e_scenarios();
        let kinds: Vec<CliE2eScenarioKind> = scenarios.iter().map(|item| item.kind).collect();
        assert_eq!(scenarios.len(), 4);
        assert_eq!(
            kinds,
            vec![
                CliE2eScenarioKind::Index,
                CliE2eScenarioKind::Search,
                CliE2eScenarioKind::Explain,
                CliE2eScenarioKind::Degrade
            ]
        );
    }

    #[test]
    fn bundles_validate_against_unified_e2e_contract() {
        let bundles = build_default_cli_e2e_bundles(&CliE2eRunConfig::default());
        assert_eq!(bundles.len(), 4);
        for bundle in bundles {
            assert_eq!(bundle.schema_version, CLI_E2E_SCHEMA_VERSION);
            bundle
                .validate()
                .expect("bundle must satisfy e2e artifact validators");
        }
    }

    #[test]
    fn degraded_scenario_produces_passed_manifest_with_required_artifacts() {
        let config = CliE2eRunConfig::default();
        let degrade = default_cli_e2e_scenarios()
            .into_iter()
            .find(|scenario| scenario.kind == CliE2eScenarioKind::Degrade)
            .expect("degrade scenario");
        let bundle = super::CliE2eArtifactBundle::build(&config, &degrade, ExitStatus::Pass);
        let artifact_files: Vec<&str> = bundle
            .manifest
            .body
            .artifacts
            .iter()
            .map(|artifact| artifact.file.as_str())
            .collect();

        assert!(artifact_files.contains(&"structured_events.jsonl"));
        assert!(artifact_files.contains(&"artifacts_index.json"));
        assert!(artifact_files.contains(&"replay_command.txt"));
        assert!(
            bundle
                .replay_command
                .contains("--exact scenario_cli_degrade_path")
        );
    }

    #[test]
    fn manifest_artifact_checksums_match_serialized_payloads() {
        let config = CliE2eRunConfig::default();
        let degrade = default_cli_e2e_scenarios()
            .into_iter()
            .find(|scenario| scenario.kind == CliE2eScenarioKind::Degrade)
            .expect("degrade scenario");
        let bundle = super::CliE2eArtifactBundle::build(&config, &degrade, ExitStatus::Pass);
        let artifact_checksums: BTreeMap<&str, &str> = bundle
            .manifest
            .body
            .artifacts
            .iter()
            .map(|artifact| (artifact.file.as_str(), artifact.checksum.as_str()))
            .collect();

        let events_jsonl = super::render_events_jsonl(&bundle.events);
        let expected_events_checksum = sha256_checksum(events_jsonl.as_bytes());
        assert_eq!(
            artifact_checksums
                .get(E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL)
                .copied(),
            Some(expected_events_checksum.as_str())
        );

        let expected_replay_checksum = sha256_checksum(bundle.replay_command.as_bytes());
        assert_eq!(
            artifact_checksums
                .get(E2E_ARTIFACT_REPLAY_COMMAND_TXT)
                .copied(),
            Some(expected_replay_checksum.as_str())
        );

        let mut index_inputs: Vec<ArtifactEntry> = bundle
            .manifest
            .body
            .artifacts
            .iter()
            .filter(|artifact| artifact.file != E2E_ARTIFACT_ARTIFACTS_INDEX_JSON)
            .cloned()
            .collect();
        index_inputs.sort_by(|left, right| left.file.cmp(&right.file));
        let artifacts_index_json =
            render_artifacts_index(&index_inputs).expect("render artifacts index payload");
        let expected_index_checksum = sha256_checksum(artifacts_index_json.as_bytes());
        assert_eq!(
            artifact_checksums
                .get(E2E_ARTIFACT_ARTIFACTS_INDEX_JSON)
                .copied(),
            Some(expected_index_checksum.as_str())
        );
    }

    #[test]
    fn filesystem_chaos_catalog_covers_reason_taxonomy() {
        let scenarios = default_cli_e2e_filesystem_chaos_scenarios();
        let ids: Vec<&str> = scenarios
            .iter()
            .map(|scenario| scenario.id.as_str())
            .collect();
        let reasons: Vec<&str> = scenarios
            .iter()
            .map(|scenario| scenario.expected_reason_code.as_str())
            .collect();

        assert_eq!(scenarios.len(), 5);
        assert!(ids.contains(&"cli-chaos-permission-denied"));
        assert!(ids.contains(&"cli-chaos-symlink-loop"));
        assert!(ids.contains(&"cli-chaos-mount-boundary"));
        assert!(ids.contains(&"cli-chaos-giant-log-skip"));
        assert!(ids.contains(&"cli-chaos-binary-blob-skip"));
        assert!(reasons.contains(&CLI_E2E_REASON_FILESYSTEM_PERMISSION_DENIED));
        assert!(reasons.contains(&CLI_E2E_REASON_FILESYSTEM_SYMLINK_LOOP));
        assert!(reasons.contains(&CLI_E2E_REASON_FILESYSTEM_MOUNT_BOUNDARY));
        assert!(reasons.contains(&CLI_E2E_REASON_FILESYSTEM_GIANT_LOG_SKIPPED));
        assert!(reasons.contains(&CLI_E2E_REASON_FILESYSTEM_BINARY_BLOB_SKIPPED));
    }

    #[test]
    fn filesystem_chaos_bundles_validate_and_target_filesystem_harness() {
        let bundles = build_cli_e2e_filesystem_chaos_bundles(&CliE2eRunConfig::default());
        assert_eq!(bundles.len(), 5);
        for bundle in bundles {
            bundle
                .validate()
                .expect("chaos bundle must satisfy e2e artifact validators");
            assert!(bundle.replay_command.contains("--test filesystem_chaos"));
            let outcomes = bundle
                .events
                .iter()
                .filter_map(|event| event.body.outcome)
                .collect::<Vec<_>>();
            assert!(
                !outcomes.is_empty(),
                "chaos bundles must include outcome-bearing assertion events"
            );
            assert!(
                outcomes.iter().all(|outcome| *outcome == E2eOutcome::Pass),
                "filesystem chaos bundles should report pass outcomes when expected_exit_code is zero"
            );
            assert!(bundle.events.iter().any(|event| {
                event
                    .body
                    .reason_code
                    .as_deref()
                    .is_some_and(|code| code == bundle.scenario.expected_reason_code)
            }));
        }
    }
}
