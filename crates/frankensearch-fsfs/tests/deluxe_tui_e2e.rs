use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::time::Duration;

use frankensearch_fsfs::interaction_primitives::{
    InteractionBudget, InteractionCycleTiming, InteractionSnapshot, LatencyBucket, LatencyPhase,
    PhaseTiming, ScreenAction, SearchInteractionDispatch, SearchInteractionEvent,
    SearchInteractionState, SearchResultEntry,
};
use frankensearch_fsfs::{DegradedRetrievalMode, FsfsScreen};
use frankensearch_tui::{InputEvent, ReplayPlayer, ReplayRecorder};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SnapshotArtifact {
    frame_seq: u64,
    checksum: u64,
    snapshot_ref: String,
    latency_bucket: String,
    visible_window: (usize, usize),
    visible_count: usize,
    selected_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct DeluxeTuiE2eArtifact {
    scenario: String,
    viewport_height: usize,
    mode: String,
    action_trace: Vec<String>,
    snapshots: Vec<SnapshotArtifact>,
    replay_json: String,
    replay_command: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ReplayFailureArtifact {
    scenario: String,
    failure_phase: String,
    expected_fingerprint: u64,
    observed_fingerprint: u64,
    expected_len: usize,
    observed_len: usize,
    mismatch_index: usize,
    snapshot_ref: String,
    replay_command: String,
}

const fn mode_label(mode: DegradedRetrievalMode) -> &'static str {
    match mode {
        DegradedRetrievalMode::Normal => "normal",
        DegradedRetrievalMode::EmbedDeferred => "embed_deferred",
        DegradedRetrievalMode::LexicalOnly => "lexical_only",
        DegradedRetrievalMode::MetadataOnly => "metadata_only",
        DegradedRetrievalMode::Paused => "paused",
    }
}

fn replay_command_for_test(test_name: &str) -> String {
    format!("cargo test -p frankensearch-fsfs --test deluxe_tui_e2e -- --exact {test_name}")
}

fn sample_results(count: usize) -> Vec<SearchResultEntry> {
    (0..count)
        .map(|idx| {
            SearchResultEntry::new(
                format!("doc-{idx:03}"),
                format!("src/module_{idx:03}.rs"),
                format!("snippet-{idx:03}"),
            )
        })
        .collect()
}

fn fixed_cycle_timing(frame_seq: u64, budget: &InteractionBudget) -> InteractionCycleTiming {
    let bounded = |duration: Duration, upper: Duration| {
        if duration <= upper { duration } else { upper }
    };

    InteractionCycleTiming {
        frame_seq,
        input: PhaseTiming {
            phase: LatencyPhase::Input,
            duration: bounded(Duration::from_millis(1), budget.input_budget),
            budget: budget.input_budget,
        },
        update: PhaseTiming {
            phase: LatencyPhase::Update,
            duration: bounded(Duration::from_millis(4), budget.update_budget),
            budget: budget.update_budget,
        },
        render: PhaseTiming {
            phase: LatencyPhase::Render,
            duration: bounded(Duration::from_millis(7), budget.render_budget),
            budget: budget.render_budget,
        },
    }
}

fn snapshot_from_state(
    state: &SearchInteractionState,
    frame_seq: u64,
    mode: DegradedRetrievalMode,
) -> InteractionSnapshot {
    InteractionSnapshot {
        seq: frame_seq,
        screen: FsfsScreen::Search,
        tick: frame_seq,
        focused_panel: state.focus.focused(),
        selected_index: Some(state.list.selected),
        scroll_offset: Some(state.list.scroll_offset),
        visible_count: Some(state.visible_results().len()),
        query_text: Some(state.query_input.clone()),
        active_filters: vec![format!("mode:{}", mode_label(mode))],
        follow_mode: None,
        degradation_mode: mode,
        checksum: 0,
    }
    .with_checksum()
}

fn replay_roundtrip_events(json: &str) -> Vec<InputEvent> {
    let mut player = ReplayPlayer::from_json(json).expect("replay JSON should decode");
    player.play();

    let mut events = Vec::new();
    while let Some((_offset, event)) = player.advance_input() {
        events.push(event);
    }
    events
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3_u64);
    }
    hash
}

fn replay_fingerprint(events: &[InputEvent]) -> u64 {
    let mut digest_input = String::new();
    for event in events {
        write!(&mut digest_input, "{event:?}|").expect("writing to String must not fail");
    }
    fnv1a64(digest_input.as_bytes())
}

fn first_mismatch_index(expected: &[InputEvent], observed: &[InputEvent]) -> Option<usize> {
    expected
        .iter()
        .zip(observed.iter())
        .position(|(left, right)| left != right)
        .or_else(|| {
            if expected.len() == observed.len() {
                None
            } else {
                Some(expected.len().min(observed.len()))
            }
        })
}

fn replay_failure_artifact(
    scenario: &str,
    snapshot_ref: &str,
    replay_command: &str,
    expected: &[InputEvent],
    observed: &[InputEvent],
) -> Option<ReplayFailureArtifact> {
    let mismatch_index = first_mismatch_index(expected, observed)?;
    Some(ReplayFailureArtifact {
        scenario: scenario.to_owned(),
        failure_phase: "replay_event_mismatch".to_owned(),
        expected_fingerprint: replay_fingerprint(expected),
        observed_fingerprint: replay_fingerprint(observed),
        expected_len: expected.len(),
        observed_len: observed.len(),
        mismatch_index,
        snapshot_ref: snapshot_ref.to_owned(),
        replay_command: replay_command.to_owned(),
    })
}

fn run_scenario(
    scenario: &str,
    viewport_height: usize,
    mode: DegradedRetrievalMode,
) -> DeluxeTuiE2eArtifact {
    let mut state = SearchInteractionState::new(viewport_height);

    state.apply_incremental_query("how does degraded retrieval mode work");
    let submit = state.apply_palette_action_id("search.submit_query");
    match submit {
        SearchInteractionDispatch::AppliedWithEvent(SearchInteractionEvent::QuerySubmitted(q)) => {
            assert_eq!(q, "how does degraded retrieval mode work");
        }
        other => panic!("expected query submit event, got {other:?}"),
    }

    state.set_results(sample_results(24));

    let mut action_trace = vec!["search.submit_query".to_owned()];

    let _ = state.apply_action(&ScreenAction::SelectDown);
    action_trace.push("search.select_down".to_owned());

    let _ = state.apply_action(&ScreenAction::PageDown);
    action_trace.push("search.page_down".to_owned());

    let _ = state.apply_action(&ScreenAction::ToggleDetailPanel);
    action_trace.push("search.toggle_explain".to_owned());

    let open = state.apply_action(&ScreenAction::OpenSelectedResult);
    action_trace.push("search.open_selected".to_owned());
    match open {
        Some(SearchInteractionEvent::OpenSelected {
            doc_id,
            source_path,
        }) => {
            assert!(doc_id.starts_with("doc-"));
            assert!(source_path.starts_with("src/"));
        }
        other => panic!("expected open-selected event, got {other:?}"),
    }

    let budget = InteractionBudget::degraded(mode);

    let cycle1 = fixed_cycle_timing(1, &budget);
    let telemetry1 = state.telemetry_sample(&cycle1, &budget);
    let snapshot1 = snapshot_from_state(&state, cycle1.frame_seq, mode);
    assert!(snapshot1.verify_checksum());

    let _ = state.apply_action(&ScreenAction::SelectDown);
    action_trace.push("search.select_down".to_owned());

    let cycle2 = fixed_cycle_timing(2, &budget);
    let telemetry2 = state.telemetry_sample(&cycle2, &budget);
    let snapshot2 = snapshot_from_state(&state, cycle2.frame_seq, mode);
    assert!(snapshot2.verify_checksum());

    let mut recorder = ReplayRecorder::new();
    recorder.start();
    let scripted_inputs = vec![
        InputEvent::Resize(120, 30),
        InputEvent::Resize(120, 32),
        InputEvent::Resize(120, 30),
    ];
    for event in &scripted_inputs {
        recorder.record(event);
    }
    recorder.stop();

    let replay_json = recorder
        .export_json()
        .expect("replay events should serialize");
    let replayed = replay_roundtrip_events(&replay_json);
    assert_eq!(replayed, scripted_inputs);

    let snapshot_artifacts = vec![
        SnapshotArtifact {
            frame_seq: cycle1.frame_seq,
            checksum: snapshot1.checksum,
            snapshot_ref: format!("snapshot-{:016x}", snapshot1.checksum),
            latency_bucket: latency_bucket_str(telemetry1.latency_bucket),
            visible_window: telemetry1.visible_window,
            visible_count: telemetry1.visible_count,
            selected_index: telemetry1.selected_index,
        },
        SnapshotArtifact {
            frame_seq: cycle2.frame_seq,
            checksum: snapshot2.checksum,
            snapshot_ref: format!("snapshot-{:016x}", snapshot2.checksum),
            latency_bucket: latency_bucket_str(telemetry2.latency_bucket),
            visible_window: telemetry2.visible_window,
            visible_count: telemetry2.visible_count,
            selected_index: telemetry2.selected_index,
        },
    ];

    DeluxeTuiE2eArtifact {
        scenario: scenario.to_owned(),
        viewport_height,
        mode: mode_label(mode).to_owned(),
        action_trace,
        snapshots: snapshot_artifacts,
        replay_json,
        replay_command: replay_command_for_test(scenario),
    }
}

fn latency_bucket_str(bucket: LatencyBucket) -> String {
    match bucket {
        LatencyBucket::UnderBudget => "under_budget".to_owned(),
        LatencyBucket::NearBudget => "near_budget".to_owned(),
        LatencyBucket::OverBudget => "over_budget".to_owned(),
    }
}

#[test]
fn scenario_tui_search_navigation_explain_flow_is_replayable() {
    let first = run_scenario(
        "scenario_tui_search_navigation_explain_flow_is_replayable",
        9,
        DegradedRetrievalMode::Normal,
    );
    let second = run_scenario(
        "scenario_tui_search_navigation_explain_flow_is_replayable",
        9,
        DegradedRetrievalMode::Normal,
    );

    assert_eq!(first.snapshots, second.snapshots);
    assert_eq!(first.action_trace, second.action_trace);
    assert!(
        first
            .action_trace
            .contains(&"search.submit_query".to_owned())
    );
    assert!(
        first
            .action_trace
            .contains(&"search.toggle_explain".to_owned())
    );
    assert!(
        first
            .action_trace
            .contains(&"search.open_selected".to_owned())
    );
    assert!(
        first
            .replay_command
            .contains("--exact scenario_tui_search_navigation_explain_flow_is_replayable")
    );

    let replayed = replay_roundtrip_events(&first.replay_json);
    assert_eq!(replayed.len(), 3);
}

#[test]
fn scenario_tui_degraded_modes_capture_budgeted_snapshots() {
    let modes = [
        DegradedRetrievalMode::Normal,
        DegradedRetrievalMode::EmbedDeferred,
        DegradedRetrievalMode::LexicalOnly,
        DegradedRetrievalMode::Paused,
    ];

    let mut totals = Vec::new();
    for mode in modes {
        let artifact = run_scenario(
            "scenario_tui_degraded_modes_capture_budgeted_snapshots",
            8,
            mode,
        );
        assert!(!artifact.snapshots.is_empty());
        for snapshot in &artifact.snapshots {
            assert_ne!(snapshot.latency_bucket, "over_budget");
            assert!(snapshot.snapshot_ref.starts_with("snapshot-"));
        }
        totals.push(InteractionBudget::degraded(mode).total());
    }

    assert!(totals[0] <= totals[1]);
    assert!(totals[1] <= totals[2]);
    assert!(totals[2] <= totals[3]);
}

#[test]
fn scenario_tui_multi_size_windows_and_snapshot_checksums_are_explicit() {
    let viewports = [4_usize, 8, 16];
    let mut first_snapshot_checksums = BTreeSet::new();

    for viewport in viewports {
        let artifact = run_scenario(
            "scenario_tui_multi_size_windows_and_snapshot_checksums_are_explicit",
            viewport,
            DegradedRetrievalMode::LexicalOnly,
        );

        let first = artifact
            .snapshots
            .first()
            .expect("at least one snapshot is required");
        first_snapshot_checksums.insert(first.checksum);

        for snapshot in &artifact.snapshots {
            assert!(snapshot.visible_window.1 >= snapshot.visible_window.0);
            assert!(snapshot.visible_count <= viewport);
            if snapshot.visible_count > 0 {
                assert!(snapshot.selected_index >= snapshot.visible_window.0);
                assert!(snapshot.selected_index < snapshot.visible_window.1);
            }
        }
    }

    assert!(
        first_snapshot_checksums.len() > 1,
        "snapshot checksums should vary across viewport sizes"
    );
}

#[test]
fn scenario_tui_replay_failures_emit_reproducible_artifacts() {
    let scenario = "scenario_tui_replay_failures_emit_reproducible_artifacts";
    let artifact = run_scenario(scenario, 10, DegradedRetrievalMode::EmbedDeferred);
    let expected = replay_roundtrip_events(&artifact.replay_json);

    let mut observed = expected.clone();
    let _ = observed.pop();

    let failure = replay_failure_artifact(
        scenario,
        &artifact.snapshots[0].snapshot_ref,
        &artifact.replay_command,
        &expected,
        &observed,
    )
    .expect("mismatch should emit replay failure artifact");

    let failure_again = replay_failure_artifact(
        scenario,
        &artifact.snapshots[0].snapshot_ref,
        &artifact.replay_command,
        &expected,
        &observed,
    )
    .expect("artifact generation should be deterministic");

    assert_eq!(failure, failure_again);
    assert_eq!(failure.failure_phase, "replay_event_mismatch");
    assert!(failure.expected_len > failure.observed_len);
    assert!(failure.snapshot_ref.starts_with("snapshot-"));
    assert!(
        failure
            .replay_command
            .contains("--exact scenario_tui_replay_failures_emit_reproducible_artifacts")
    );

    let serialized = serde_json::to_string(&failure).expect("artifact should serialize");
    let decoded: ReplayFailureArtifact =
        serde_json::from_str(&serialized).expect("artifact should round-trip");
    assert_eq!(decoded, failure);
}
