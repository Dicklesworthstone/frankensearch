use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use frankensearch_core::{
    ArtifactEmissionInput, ArtifactEntry, ClockMode, Correlation, DeterminismTier, DiffEntry,
    E2E_ARTIFACT_ARTIFACTS_INDEX_JSON, E2E_ARTIFACT_REPLAY_COMMAND_TXT,
    E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL, E2E_ARTIFACT_TERMINAL_TRANSCRIPT_TXT, E2E_SCHEMA_EVENT,
    E2E_SCHEMA_MANIFEST, E2E_SCHEMA_REPLAY, E2E_SCHEMA_SNAPSHOT_DIFF, E2eEnvelope, E2eEventType,
    E2eOutcome, E2eSeverity, EventBody, ExitStatus, ManifestBody, ModelVersion, Platform,
    ReplayBody, ReplayEventType, SnapshotDiffBody, Suite, build_artifact_entries,
    normalize_replay_command, render_artifacts_index, sha256_checksum, validate_event_envelope,
    validate_manifest_envelope,
};
use frankensearch_ops::data_source::TimeWindow;
use frankensearch_ops::state::{FleetSnapshot, ResourceMetrics, SearchMetrics};
use frankensearch_ops::{
    ControlPlaneMetrics, DataSource, InstanceAttribution, InstanceLifecycle, MockDataSource,
    OpsApp, ViewPreset,
};
use frankensearch_tui::{InputEvent, ReplayPlayer, ReplayRecorder};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use serde::{Deserialize, Serialize};

const OPS_RUN_ID: &str = "01HQXG5M7P3KZFV9N2RSTW6YAB";
const OPS_TS: &str = "2026-02-15T01:20:00Z";
const OPS_SEED: u64 = 0x5EED_2830_0001;
const OPS_ROOT_REQUEST_ID: &str = "01HQXG5M7Q3KZFV9N2RSTW6YAC";
const OPS_BASELINE_RUN_ID: &str = "01HQXG5M7R3KZFV9N2RSTW6YAD";
const OPS_REASON_DISCOVERY: &str = "e2e.ops.discovery";
const OPS_REASON_TRIAGE: &str = "e2e.ops.triage";
const OPS_REASON_DRILLDOWN: &str = "e2e.ops.drilldown";
const OPS_REASON_RECOVERY: &str = "e2e.ops.recovery";
const OPS_REASON_DIFF_FAIL: &str = "e2e.diff.tolerance_exceeded";
const OPS_REASON_PASS: &str = "e2e.ops.pass";

#[derive(Debug)]
struct SequencedOpsSource {
    snapshots: Vec<FleetSnapshot>,
    control_plane: Vec<ControlPlaneMetrics>,
    cursor: AtomicUsize,
}

impl SequencedOpsSource {
    fn new(snapshots: Vec<FleetSnapshot>, control_plane: Vec<ControlPlaneMetrics>) -> Self {
        assert!(
            !snapshots.is_empty() && snapshots.len() == control_plane.len(),
            "sequenced source requires aligned non-empty snapshots and control-plane metrics"
        );
        Self {
            snapshots,
            control_plane,
            cursor: AtomicUsize::new(0),
        }
    }

    fn current_index(&self) -> usize {
        let current = self.cursor.load(Ordering::Relaxed);
        if current == 0 {
            0
        } else {
            (current - 1).min(self.control_plane.len().saturating_sub(1))
        }
    }
}

impl DataSource for SequencedOpsSource {
    fn fleet_snapshot(&self) -> FleetSnapshot {
        let idx = self.cursor.fetch_add(1, Ordering::Relaxed);
        let bounded = idx.min(self.snapshots.len().saturating_sub(1));
        self.snapshots[bounded].clone()
    }

    fn search_metrics(&self, _instance_id: &str, _window: TimeWindow) -> Option<SearchMetrics> {
        None
    }

    fn resource_metrics(&self, _instance_id: &str) -> Option<ResourceMetrics> {
        None
    }

    fn control_plane_metrics(&self) -> ControlPlaneMetrics {
        self.control_plane[self.current_index()].clone()
    }

    fn attribution(&self, _instance_id: &str) -> Option<InstanceAttribution> {
        None
    }

    fn lifecycle(&self, _instance_id: &str) -> Option<InstanceLifecycle> {
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SnapshotCapture {
    label: String,
    width: u16,
    height: u16,
    active_screen: String,
    health: String,
    density: String,
    contrast: String,
    checksum: String,
}

#[derive(Debug)]
struct ScenarioOutput {
    manifest: E2eEnvelope<ManifestBody>,
    events: Vec<E2eEnvelope<EventBody>>,
    replay: Vec<E2eEnvelope<ReplayBody>>,
    snapshots: Vec<SnapshotCapture>,
    replay_json: String,
    replay_command: String,
    artifacts_index_json: String,
    terminal_transcript: String,
    snapshot_diff: Option<E2eEnvelope<SnapshotDiffBody>>,
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3_u64);
    }
    hash
}

fn render_app_text(app: &mut OpsApp, width: u16, height: u16) -> String {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test backend should initialize");
    terminal
        .draw(|frame| app.render(frame))
        .expect("ops app should render in test backend");

    let buffer = terminal.backend().buffer();
    let mut lines = Vec::with_capacity(usize::from(height));
    for y in 0..height {
        let mut line = String::new();
        for x in 0..width {
            let cell = buffer
                .cell((x, y))
                .expect("buffer coordinates should exist in declared viewport");
            line.push_str(cell.symbol());
        }
        lines.push(line.trim_end().to_owned());
    }
    lines.join("\n")
}

fn capture_snapshot(app: &mut OpsApp, label: &str, width: u16, height: u16) -> SnapshotCapture {
    let rendered = render_app_text(app, width, height);
    let checksum = format!("{:016x}", fnv1a64(rendered.as_bytes()));
    let active_screen = app
        .shell
        .active_screen
        .as_ref()
        .map_or_else(|| "none".to_owned(), |id| id.0.clone());

    SnapshotCapture {
        label: label.to_owned(),
        width,
        height,
        active_screen,
        health: app.state.control_plane_health().to_string(),
        density: app.view.density.to_string(),
        contrast: app.preferences.contrast.label().to_owned(),
        checksum,
    }
}

struct EventSpec<'a> {
    event_type: E2eEventType,
    severity: E2eSeverity,
    lane_id: Option<&'a str>,
    outcome: Option<E2eOutcome>,
    reason_code: Option<&'a str>,
    context: Option<String>,
    metrics: Option<BTreeMap<String, f64>>,
}

fn make_event(seq: u64, spec: EventSpec<'_>) -> E2eEnvelope<EventBody> {
    let event_id = format!("01HQXG5M7S3KZFV9N2RSTW{seq:04X}");
    E2eEnvelope::new(
        E2E_SCHEMA_EVENT,
        OPS_RUN_ID,
        OPS_TS,
        EventBody {
            event_type: spec.event_type,
            correlation: Correlation {
                event_id,
                root_request_id: OPS_ROOT_REQUEST_ID.to_owned(),
                parent_event_id: None,
            },
            severity: spec.severity,
            lane_id: spec.lane_id.map(std::borrow::ToOwned::to_owned),
            oracle_id: None,
            outcome: spec.outcome,
            reason_code: spec.reason_code.map(std::borrow::ToOwned::to_owned),
            context: spec.context,
            metrics: spec.metrics,
        },
    )
}

fn to_jsonl<T: Serialize>(items: &[T]) -> String {
    items
        .iter()
        .map(|item| serde_json::to_string(item).expect("jsonl item should serialize"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_evidence_jsonl(snapshot: &FleetSnapshot) -> (String, u64) {
    let lines: Vec<String> = snapshot
        .lifecycle_events()
        .iter()
        .map(|event| {
            serde_json::json!({
                "ts_ms": event.at_ms,
                "instance_id": event.instance_id,
                "reason_code": event.reason_code,
                "confidence": event.attribution_confidence_score,
                "collision": event.attribution_collision,
            })
            .to_string()
        })
        .collect();
    let line_count = usize_to_u64(lines.len());
    (lines.join("\n"), line_count)
}

fn replay_command_for(test_name: &str) -> String {
    normalize_replay_command(&format!(
        "cargo test -p frankensearch-ops --test ops_pty_snapshot_e2e -- --exact {test_name}"
    ))
}

#[allow(clippy::too_many_lines)]
fn run_ops_scenario(test_name: &str, force_snapshot_diff_failure: bool) -> ScenarioOutput {
    let base_snapshot = MockDataSource::sample().fleet_snapshot();
    let source = SequencedOpsSource::new(
        vec![base_snapshot.clone(), base_snapshot.clone(), base_snapshot],
        vec![
            ControlPlaneMetrics::default(),
            ControlPlaneMetrics {
                ingestion_lag_events: 3_100,
                event_throughput_eps: 0.75,
                ..ControlPlaneMetrics::default()
            },
            ControlPlaneMetrics::default(),
        ],
    );

    let mut app = OpsApp::new(Box::new(source));
    let mut recorder = ReplayRecorder::new();
    recorder.start();

    let mut transcript_lines = vec![format!(
        "run_id={OPS_RUN_ID} seed={OPS_SEED} lane=ops.pty_snapshot"
    )];
    let mut snapshots = Vec::new();
    let mut events = Vec::new();
    let mut event_seq = 0_u64;

    events.push(make_event(
        event_seq,
        EventSpec {
            event_type: E2eEventType::E2eStart,
            severity: E2eSeverity::Info,
            lane_id: Some("ops.pty_snapshot"),
            outcome: Some(E2eOutcome::Pass),
            reason_code: None,
            context: Some("begin ops PTY + snapshot scenario".to_owned()),
            metrics: None,
        },
    ));
    event_seq = event_seq.saturating_add(1);

    let desktop_resize = InputEvent::Resize(132, 40);
    recorder.record(&desktop_resize);
    let _ = app.handle_input(&desktop_resize);
    app.refresh_data();
    transcript_lines.push(format!(
        "phase=discovery screen={} health={} density={} contrast={}",
        app.shell
            .active_screen
            .as_ref()
            .map_or_else(|| "none".to_owned(), |id| id.0.clone()),
        app.state.control_plane_health(),
        app.view.density,
        app.preferences.contrast.label()
    ));
    snapshots.push(capture_snapshot(&mut app, "desktop.discovery", 132, 40));
    let mut discovery_metrics = BTreeMap::new();
    discovery_metrics.insert(
        "instance_count".to_owned(),
        f64::from(u32::try_from(app.state.fleet().instance_count()).unwrap_or(u32::MAX)),
    );
    events.push(make_event(
        event_seq,
        EventSpec {
            event_type: E2eEventType::PhaseTransition,
            severity: E2eSeverity::Info,
            lane_id: Some("ops.discovery"),
            outcome: Some(E2eOutcome::Pass),
            reason_code: Some(OPS_REASON_DISCOVERY),
            context: Some("fleet snapshot loaded for discovery triage".to_owned()),
            metrics: Some(discovery_metrics),
        },
    ));
    event_seq = event_seq.saturating_add(1);

    let goto_timeline = InputEvent::Key(
        crossterm::event::KeyCode::Char('t'),
        crossterm::event::KeyModifiers::NONE,
    );
    recorder.record(&goto_timeline);
    let _ = app.handle_input(&goto_timeline);
    snapshots.push(capture_snapshot(&mut app, "desktop.triage", 132, 40));
    events.push(make_event(
        event_seq,
        EventSpec {
            event_type: E2eEventType::PhaseTransition,
            severity: E2eSeverity::Info,
            lane_id: Some("ops.triage"),
            outcome: Some(E2eOutcome::Pass),
            reason_code: Some(OPS_REASON_TRIAGE),
            context: Some("timeline triage navigation from fleet view".to_owned()),
            metrics: None,
        },
    ));
    event_seq = event_seq.saturating_add(1);

    let drilldown_project = InputEvent::Key(
        crossterm::event::KeyCode::Char('g'),
        crossterm::event::KeyModifiers::NONE,
    );
    recorder.record(&drilldown_project);
    let _ = app.handle_input(&drilldown_project);

    app.view.apply_preset(ViewPreset::ProjectDeepDive);
    app.preferences.toggle_contrast();
    app.preferences.toggle_focus_visibility();
    app.refresh_data();

    transcript_lines.push(format!(
        "phase=drilldown screen={} health={} density={} contrast={}",
        app.shell
            .active_screen
            .as_ref()
            .map_or_else(|| "none".to_owned(), |id| id.0.clone()),
        app.state.control_plane_health(),
        app.view.density,
        app.preferences.contrast.label()
    ));
    snapshots.push(capture_snapshot(&mut app, "desktop.drilldown", 132, 40));
    events.push(make_event(
        event_seq,
        EventSpec {
            event_type: E2eEventType::PhaseTransition,
            severity: E2eSeverity::Warn,
            lane_id: Some("ops.drilldown"),
            outcome: Some(E2eOutcome::Pass),
            reason_code: Some(OPS_REASON_DRILLDOWN),
            context: Some("project drilldown under degraded control-plane health".to_owned()),
            metrics: None,
        },
    ));
    event_seq = event_seq.saturating_add(1);

    let back_to_fleet = InputEvent::Key(
        crossterm::event::KeyCode::Esc,
        crossterm::event::KeyModifiers::NONE,
    );
    recorder.record(&back_to_fleet);
    let _ = app.handle_input(&back_to_fleet);

    let compact_resize = InputEvent::Resize(96, 28);
    recorder.record(&compact_resize);
    let _ = app.handle_input(&compact_resize);

    app.view.apply_preset(ViewPreset::LowNoise);
    app.preferences.toggle_contrast();
    app.refresh_data();

    transcript_lines.push(format!(
        "phase=recovery screen={} health={} density={} contrast={}",
        app.shell
            .active_screen
            .as_ref()
            .map_or_else(|| "none".to_owned(), |id| id.0.clone()),
        app.state.control_plane_health(),
        app.view.density,
        app.preferences.contrast.label()
    ));
    snapshots.push(capture_snapshot(&mut app, "compact.recovery", 96, 28));
    events.push(make_event(
        event_seq,
        EventSpec {
            event_type: E2eEventType::PhaseTransition,
            severity: E2eSeverity::Info,
            lane_id: Some("ops.recovery"),
            outcome: Some(E2eOutcome::Pass),
            reason_code: Some(OPS_REASON_RECOVERY),
            context: Some("fleet recovered after degraded control-plane interval".to_owned()),
            metrics: None,
        },
    ));
    event_seq = event_seq.saturating_add(1);

    recorder.stop();
    let replay_json = recorder
        .export_json()
        .expect("recorded replay should serialize");
    let mut replay_player =
        ReplayPlayer::from_json(&replay_json).expect("recorded replay should deserialize");
    replay_player.play();

    let mut replay = Vec::new();
    let mut replay_line_count = 0_u64;
    while let Some((offset, event)) = replay_player.advance_input() {
        replay.push(E2eEnvelope::new(
            E2E_SCHEMA_REPLAY,
            OPS_RUN_ID,
            OPS_TS,
            ReplayBody {
                replay_type: ReplayEventType::Signal,
                offset_ms: u64::try_from(offset.as_millis()).unwrap_or(u64::MAX),
                seq: replay_line_count,
                payload: serde_json::json!({
                    "event": format!("{event:?}"),
                }),
            },
        ));
        replay_line_count = replay_line_count.saturating_add(1);
    }
    assert!(
        replay_line_count > 0,
        "PTY replay stream should capture input events"
    );

    let replay_jsonl = to_jsonl(&replay);
    let structured_events_jsonl = to_jsonl(&events);
    let structured_event_line_count = usize_to_u64(events.len());
    let terminal_transcript = transcript_lines.join("\n");
    let replay_command = replay_command_for(test_name);

    let (evidence_jsonl, evidence_line_count) = render_evidence_jsonl(app.state.fleet());
    let snapshot_matrix_json =
        serde_json::to_vec_pretty(&snapshots).expect("snapshot matrix should serialize");
    let replay_seed_json = serde_json::to_string_pretty(&serde_json::json!({
        "seed": OPS_SEED,
        "run_id": OPS_RUN_ID,
    }))
    .expect("replay seed payload should serialize");

    let snapshot_diff = if force_snapshot_diff_failure {
        let baseline = snapshots.first().map_or_else(
            || "0000000000000000".to_owned(),
            |snapshot| snapshot.checksum.clone(),
        );
        let observed = format!(
            "{:016x}",
            fnv1a64(format!("{baseline}:mismatch").as_bytes())
        );
        Some(E2eEnvelope::new(
            E2E_SCHEMA_SNAPSHOT_DIFF,
            OPS_RUN_ID,
            OPS_TS,
            SnapshotDiffBody {
                comparison_mode: DeterminismTier::BitExact,
                baseline_run_id: OPS_BASELINE_RUN_ID.to_owned(),
                diffs: vec![DiffEntry {
                    field_path: "snapshots[3].checksum".to_owned(),
                    baseline,
                    current: observed,
                    delta: Some("non_zero".to_owned()),
                    within_tolerance: false,
                    tolerance: Some("exact".to_owned()),
                }],
                pass: false,
                mismatch_count: 1,
            },
        ))
    } else {
        None
    };

    let snapshot_diff_json = snapshot_diff
        .as_ref()
        .map(|payload| serde_json::to_string_pretty(payload).expect("snapshot diff should encode"));

    let mut emission_inputs = vec![
        ArtifactEmissionInput {
            file: E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL,
            bytes: structured_events_jsonl.as_bytes(),
            line_count: Some(structured_event_line_count),
        },
        ArtifactEmissionInput {
            file: "ops_replay_trace.jsonl",
            bytes: replay_jsonl.as_bytes(),
            line_count: Some(replay_line_count),
        },
        ArtifactEmissionInput {
            file: "ops_evidence.jsonl",
            bytes: evidence_jsonl.as_bytes(),
            line_count: Some(evidence_line_count),
        },
        ArtifactEmissionInput {
            file: "ops_snapshot_matrix.json",
            bytes: &snapshot_matrix_json,
            line_count: None,
        },
        ArtifactEmissionInput {
            file: "replay_seed.json",
            bytes: replay_seed_json.as_bytes(),
            line_count: None,
        },
        ArtifactEmissionInput {
            file: E2E_ARTIFACT_REPLAY_COMMAND_TXT,
            bytes: replay_command.as_bytes(),
            line_count: None,
        },
        ArtifactEmissionInput {
            file: E2E_ARTIFACT_TERMINAL_TRANSCRIPT_TXT,
            bytes: terminal_transcript.as_bytes(),
            line_count: None,
        },
    ];
    if let Some(payload) = snapshot_diff_json.as_ref() {
        emission_inputs.push(ArtifactEmissionInput {
            file: "ops_snapshot_diff.json",
            bytes: payload.as_bytes(),
            line_count: None,
        });
    }

    let mut artifacts = build_artifact_entries(emission_inputs)
        .expect("ops scenario artifacts should satisfy canonical emission contracts");
    let artifacts_index_json =
        render_artifacts_index(&artifacts).expect("artifact index should serialize");
    artifacts.push(ArtifactEntry {
        file: E2E_ARTIFACT_ARTIFACTS_INDEX_JSON.to_owned(),
        checksum: sha256_checksum(artifacts_index_json.as_bytes()),
        line_count: None,
    });
    artifacts.sort_by(|left, right| left.file.cmp(&right.file));

    let config_hash = sha256_checksum(
        serde_json::to_string(&serde_json::json!({
            "seed": OPS_SEED,
            "snapshot_sizes": snapshots
                .iter()
                .map(|snapshot| format!("{}x{}", snapshot.width, snapshot.height))
                .collect::<Vec<_>>(),
        }))
        .expect("config payload should serialize")
        .as_bytes(),
    );

    let exit_status = if force_snapshot_diff_failure {
        ExitStatus::Fail
    } else {
        ExitStatus::Pass
    };

    let end_outcome = if force_snapshot_diff_failure {
        E2eOutcome::Fail
    } else {
        E2eOutcome::Pass
    };
    let end_reason = if force_snapshot_diff_failure {
        OPS_REASON_DIFF_FAIL
    } else {
        OPS_REASON_PASS
    };
    events.push(make_event(
        event_seq,
        EventSpec {
            event_type: E2eEventType::E2eEnd,
            severity: if force_snapshot_diff_failure {
                E2eSeverity::Error
            } else {
                E2eSeverity::Info
            },
            lane_id: Some("ops.pty_snapshot"),
            outcome: Some(end_outcome),
            reason_code: Some(end_reason),
            context: Some("ops PTY + snapshot scenario finished".to_owned()),
            metrics: None,
        },
    ));

    for event in &events {
        validate_event_envelope(event).expect("event envelope should validate");
    }

    let manifest = E2eEnvelope::new(
        E2E_SCHEMA_MANIFEST,
        OPS_RUN_ID,
        OPS_TS,
        ManifestBody {
            suite: Suite::Ops,
            determinism_tier: DeterminismTier::BitExact,
            seed: OPS_SEED,
            config_hash,
            index_version: Some("ops-v1".to_owned()),
            model_versions: vec![ModelVersion {
                name: "frankensearch-ops".to_owned(),
                revision: "0.1.0".to_owned(),
                digest: None,
            }],
            platform: Platform {
                os: std::env::consts::OS.to_owned(),
                arch: std::env::consts::ARCH.to_owned(),
                rustc: "nightly".to_owned(),
            },
            clock_mode: ClockMode::Simulated,
            tie_break_policy: "ops.screen_id.lexical".to_owned(),
            artifacts,
            duration_ms: 150,
            exit_status,
        },
    );
    validate_manifest_envelope(&manifest).expect("ops manifest should validate");

    ScenarioOutput {
        manifest,
        events,
        replay,
        snapshots,
        replay_json,
        replay_command,
        artifacts_index_json,
        terminal_transcript,
        snapshot_diff,
    }
}

#[test]
fn ops_snapshot_matrix_covers_multi_size_accessibility_and_density_modes() {
    let output = run_ops_scenario(
        "ops_snapshot_matrix_covers_multi_size_accessibility_and_density_modes",
        false,
    );

    assert_eq!(output.manifest.body.suite, Suite::Ops);
    assert_eq!(output.manifest.body.exit_status, ExitStatus::Pass);
    assert!(output.snapshot_diff.is_none());

    assert!(
        output
            .snapshots
            .iter()
            .any(|snapshot| snapshot.width >= 120 && snapshot.height >= 40),
        "snapshot matrix must include desktop viewport"
    );
    assert!(
        output
            .snapshots
            .iter()
            .any(|snapshot| snapshot.width <= 100 && snapshot.height <= 30),
        "snapshot matrix must include compact viewport"
    );
    assert!(
        output
            .snapshots
            .iter()
            .any(|snapshot| snapshot.contrast == "High Contrast"),
        "snapshot matrix must include an accessibility high-contrast capture"
    );
    assert!(
        output
            .snapshots
            .iter()
            .any(|snapshot| snapshot.density == "Expanded")
            && output
                .snapshots
                .iter()
                .any(|snapshot| snapshot.density == "Compact"),
        "snapshot matrix must include multiple density modes"
    );

    let reason_codes: Vec<&str> = output
        .events
        .iter()
        .filter_map(|envelope| envelope.body.reason_code.as_deref())
        .collect();
    assert!(reason_codes.contains(&OPS_REASON_DISCOVERY));
    assert!(reason_codes.contains(&OPS_REASON_TRIAGE));
    assert!(reason_codes.contains(&OPS_REASON_DRILLDOWN));
    assert!(reason_codes.contains(&OPS_REASON_RECOVERY));
    assert!(reason_codes.contains(&OPS_REASON_PASS));
}

#[test]
fn ops_failure_bundle_includes_transcript_snapshot_diff_and_replay_entrypoint() {
    let output = run_ops_scenario(
        "ops_failure_bundle_includes_transcript_snapshot_diff_and_replay_entrypoint",
        true,
    );

    assert_eq!(output.manifest.body.suite, Suite::Ops);
    assert_eq!(output.manifest.body.exit_status, ExitStatus::Fail);
    assert!(output.snapshot_diff.is_some());

    let artifact_files: Vec<&str> = output
        .manifest
        .body
        .artifacts
        .iter()
        .map(|entry| entry.file.as_str())
        .collect();
    assert!(artifact_files.contains(&E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL));
    assert!(artifact_files.contains(&E2E_ARTIFACT_REPLAY_COMMAND_TXT));
    assert!(artifact_files.contains(&E2E_ARTIFACT_ARTIFACTS_INDEX_JSON));
    assert!(artifact_files.contains(&E2E_ARTIFACT_TERMINAL_TRANSCRIPT_TXT));
    assert!(artifact_files.contains(&"ops_snapshot_diff.json"));

    assert!(
        output
            .replay_command
            .contains("cargo test -p frankensearch-ops"),
        "replay entrypoint should be a copy/paste-able cargo command"
    );
    assert!(
        output.terminal_transcript.contains("phase=discovery")
            && output.terminal_transcript.contains("phase=recovery"),
        "terminal transcript should capture end-to-end phase transitions"
    );

    let mut replay_player =
        ReplayPlayer::from_json(&output.replay_json).expect("replay payload should decode");
    replay_player.play();
    let mut replay_events = 0_usize;
    while replay_player.advance_input().is_some() {
        replay_events = replay_events.saturating_add(1);
    }
    assert_eq!(
        replay_events,
        output.replay.len(),
        "replay stream should roundtrip deterministically"
    );
    assert!(
        output
            .artifacts_index_json
            .contains("ops_snapshot_diff.json"),
        "artifact index must include failing snapshot diff payload"
    );

    let end_event = output
        .events
        .iter()
        .find(|event| event.body.event_type == E2eEventType::E2eEnd)
        .expect("e2e end event should be present");
    assert_eq!(end_event.body.outcome, Some(E2eOutcome::Fail));
    assert_eq!(
        end_event.body.reason_code.as_deref(),
        Some(OPS_REASON_DIFF_FAIL)
    );
}
