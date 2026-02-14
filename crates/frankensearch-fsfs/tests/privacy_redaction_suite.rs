use std::collections::BTreeMap;
use std::time::Duration;

use frankensearch_core::{
    ArtifactEmissionInput, ArtifactEntry as E2eArtifactEntry, ClockMode, Correlation,
    DeterminismTier, E2E_ARTIFACT_ARTIFACTS_INDEX_JSON, E2E_ARTIFACT_REPLAY_COMMAND_TXT,
    E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL, E2E_SCHEMA_EVENT, E2E_SCHEMA_MANIFEST, E2E_SCHEMA_REPLAY,
    E2eEnvelope, E2eEventType, E2eOutcome, E2eSeverity, EventBody, EvidenceEventType, ExitStatus,
    ExplainedSource, ExplanationPhase, HitExplanation, ManifestBody, ModelVersion, PipelineState,
    Platform, ReplayBody, ReplayEventType, ScoreComponent, Severity, Suite, build_artifact_entries,
    render_artifacts_index, sha256_checksum, validate_event_envelope, validate_manifest_envelope,
};
use frankensearch_fsfs::interaction_primitives::{
    InteractionBudget, InteractionCycleTiming, LatencyPhase, PhaseTiming, SearchInteractionState,
    SearchResultEntry,
};
use frankensearch_fsfs::repro::{ReplayClientSurface, ReplayEntrypoint};
use frankensearch_fsfs::{
    ArtifactEntry as ReproArtifactEntry, CaptureReason, DataClass, FrameSeqRange, FsfsEventFamily,
    FsfsEvidenceEvent, FsfsExplanationPayload, FsfsReasonCode, OutputEnvelope, OutputError,
    OutputMeta, OutputSurface, REDACTION_POLICY_VERSION, RankingExplanation, RedactionPolicy,
    ReplayMeta, ReplayMode, ReproInstance, ReproManifest, RetentionTier, ScopeDecision,
    ScopeDecisionKind, StreamEvent, StreamFrame, StreamStartedEvent, TraceLink,
    encode_stream_frame_ndjson,
};
use serde::{Deserialize, Serialize};

const RAW_TOKEN: &str = "sk_live_SUPER_SECRET_12345";
const RAW_EMAIL: &str = "jane.doe@example.com";
const RAW_PRIVATE_KEY: &str = "-----BEGIN PRIVATE KEY-----";
const RAW_SESSION: &str = "sessionid=abcdef123";
const REDACTED_MARKER: &str = "<redacted>";
const PRIVACY_RUN_ID: &str = "01JAH9A2W8F8Q6GQ4C7M3N2P1R";
const PRIVACY_TS: &str = "2026-02-14T00:00:04Z";
const PRIVACY_REASON_PASS: &str = "e2e.privacy.redaction_pass";
const PRIVACY_REASON_LEAK: &str = "e2e.privacy.redaction_leak";
const PRIVACY_REPLAY_COMMAND: &str = "cargo test -p frankensearch-fsfs --test privacy_redaction_suite -- --nocapture --exact redaction_assertions_cover_logs_evidence_explain_and_streamed_outputs";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct LeakFinding {
    artifact: String,
    pattern: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct LeakReport {
    schema_version: String,
    policy_version: String,
    replay_handle: String,
    findings: Vec<LeakFinding>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct PrivacyLaneMeta {
    lane_id: String,
    policy_version: String,
    finding_count: usize,
    replay_handle: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct PrivacyUnifiedE2eBundle {
    manifest: E2eEnvelope<ManifestBody>,
    events: Vec<E2eEnvelope<EventBody>>,
    replay: Vec<E2eEnvelope<ReplayBody>>,
    lane_meta: PrivacyLaneMeta,
}

fn redacted_or_marker(
    policy: &RedactionPolicy,
    data_class: DataClass,
    surface: OutputSurface,
    value: &str,
) -> String {
    policy
        .apply(data_class, surface, value)
        .unwrap_or_else(|| REDACTED_MARKER.to_owned())
}

fn build_leak_report(
    policy_version: &str,
    replay_handle: &str,
    artifacts: &[(&str, String)],
) -> LeakReport {
    let patterns = [RAW_TOKEN, RAW_EMAIL, RAW_PRIVATE_KEY, RAW_SESSION];
    let mut findings = Vec::new();

    for (artifact, payload) in artifacts {
        for pattern in &patterns {
            if payload.contains(pattern) {
                findings.push(LeakFinding {
                    artifact: (*artifact).to_owned(),
                    pattern: (*pattern).to_owned(),
                });
            }
        }
    }

    LeakReport {
        schema_version: "fsfs.privacy.leak_report.v1".to_owned(),
        policy_version: policy_version.to_owned(),
        replay_handle: replay_handle.to_owned(),
        findings,
    }
}

fn sample_ranking() -> RankingExplanation {
    let hit = HitExplanation {
        final_score: 0.73,
        components: vec![ScoreComponent {
            source: ExplainedSource::LexicalBm25 {
                matched_terms: vec!["privacy".to_owned(), "redaction".to_owned()],
                tf: 2.0,
                idf: 3.0,
            },
            raw_score: 3.2,
            normalized_score: 0.8,
            rrf_contribution: 0.02,
            weight: 1.0,
        }],
        phase: ExplanationPhase::Initial,
        rank_movement: None,
    };

    RankingExplanation::from_hit_explanation(
        "doc-privacy-001",
        &hit,
        FsfsReasonCode::QUERY_EXPLAIN_ATTACHED,
        900,
    )
}

fn build_privacy_unified_bundle(
    report: &LeakReport,
    policy: &RedactionPolicy,
) -> PrivacyUnifiedE2eBundle {
    let lane_meta = PrivacyLaneMeta {
        lane_id: "privacy.redaction".to_owned(),
        policy_version: policy.version.clone(),
        finding_count: report.findings.len(),
        replay_handle: report.replay_handle.clone(),
    };

    let outcome = if report.findings.is_empty() {
        E2eOutcome::Pass
    } else {
        E2eOutcome::Fail
    };
    let exit_status = if report.findings.is_empty() {
        ExitStatus::Pass
    } else {
        ExitStatus::Fail
    };
    let reason = if report.findings.is_empty() {
        PRIVACY_REASON_PASS
    } else {
        PRIVACY_REASON_LEAK
    };

    let mut metrics = BTreeMap::new();
    metrics.insert("finding_count".to_owned(), lane_meta.finding_count as f64);

    let event_bodies = vec![
        EventBody {
            event_type: E2eEventType::E2eStart,
            correlation: Correlation {
                event_id: String::new(),
                root_request_id: String::new(),
                parent_event_id: None,
            },
            severity: E2eSeverity::Info,
            lane_id: None,
            oracle_id: None,
            outcome: None,
            reason_code: Some("e2e.privacy.start".to_owned()),
            context: Some("privacy redaction suite started".to_owned()),
            metrics: None,
        },
        EventBody {
            event_type: E2eEventType::LaneStart,
            correlation: Correlation {
                event_id: String::new(),
                root_request_id: String::new(),
                parent_event_id: None,
            },
            severity: E2eSeverity::Info,
            lane_id: Some(lane_meta.lane_id.clone()),
            oracle_id: None,
            outcome: None,
            reason_code: Some("e2e.privacy.lane_start".to_owned()),
            context: Some("validate redaction surfaces and replay metadata".to_owned()),
            metrics: Some(metrics.clone()),
        },
        EventBody {
            event_type: E2eEventType::Assertion,
            correlation: Correlation {
                event_id: String::new(),
                root_request_id: String::new(),
                parent_event_id: None,
            },
            severity: if matches!(outcome, E2eOutcome::Pass) {
                E2eSeverity::Info
            } else {
                E2eSeverity::Warn
            },
            lane_id: None,
            oracle_id: Some("privacy.leak_report".to_owned()),
            outcome: Some(outcome),
            reason_code: Some(reason.to_owned()),
            context: Some("privacy leak report evaluation".to_owned()),
            metrics: Some(metrics),
        },
        EventBody {
            event_type: E2eEventType::LaneEnd,
            correlation: Correlation {
                event_id: String::new(),
                root_request_id: String::new(),
                parent_event_id: None,
            },
            severity: E2eSeverity::Info,
            lane_id: Some(lane_meta.lane_id.clone()),
            oracle_id: None,
            outcome: Some(outcome),
            reason_code: Some(reason.to_owned()),
            context: None,
            metrics: None,
        },
        EventBody {
            event_type: E2eEventType::E2eEnd,
            correlation: Correlation {
                event_id: String::new(),
                root_request_id: String::new(),
                parent_event_id: None,
            },
            severity: if matches!(outcome, E2eOutcome::Pass) {
                E2eSeverity::Info
            } else {
                E2eSeverity::Warn
            },
            lane_id: None,
            oracle_id: None,
            outcome: Some(outcome),
            reason_code: Some(reason.to_owned()),
            context: None,
            metrics: None,
        },
    ];

    let events: Vec<E2eEnvelope<EventBody>> = event_bodies
        .iter()
        .enumerate()
        .map(|(index, body)| {
            E2eEnvelope::new(
                E2E_SCHEMA_EVENT,
                PRIVACY_RUN_ID,
                PRIVACY_TS,
                EventBody {
                    correlation: Correlation {
                        event_id: format!("privacy-evt-{index:02}"),
                        root_request_id: "privacy-root".to_owned(),
                        parent_event_id: None,
                    },
                    ..body.clone()
                },
            )
        })
        .collect();

    let events_jsonl = events
        .iter()
        .map(|event| serde_json::to_string(event).expect("serialize privacy event envelope"))
        .collect::<Vec<_>>()
        .join("\n");

    let lane_meta_json =
        serde_json::to_string(&lane_meta).expect("serialize privacy lane metadata artifact");
    let leak_report_json =
        serde_json::to_string(report).expect("serialize privacy leak-report artifact");
    let replay_payload = serde_json::json!({
        "lane_id": lane_meta.lane_id,
        "policy_version": lane_meta.policy_version,
        "replay_handle": lane_meta.replay_handle,
    });
    let replay = vec![E2eEnvelope::new(
        E2E_SCHEMA_REPLAY,
        PRIVACY_RUN_ID,
        PRIVACY_TS,
        ReplayBody {
            replay_type: ReplayEventType::Query,
            offset_ms: 0,
            seq: 1,
            payload: replay_payload,
        },
    )];

    let event_count = u64::try_from(events.len()).expect("event count fits in u64");
    let mut artifacts = build_artifact_entries([
        ArtifactEmissionInput {
            file: E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL,
            bytes: events_jsonl.as_bytes(),
            line_count: Some(event_count),
        },
        ArtifactEmissionInput {
            file: E2E_ARTIFACT_REPLAY_COMMAND_TXT,
            bytes: PRIVACY_REPLAY_COMMAND.as_bytes(),
            line_count: None,
        },
        ArtifactEmissionInput {
            file: "privacy/leak_report.json",
            bytes: leak_report_json.as_bytes(),
            line_count: None,
        },
        ArtifactEmissionInput {
            file: "privacy/lane_meta.json",
            bytes: lane_meta_json.as_bytes(),
            line_count: None,
        },
    ])
    .expect("privacy artifact entries should satisfy shared emitter contracts");
    let artifacts_index_json = render_artifacts_index(&artifacts).expect("render artifacts index");
    artifacts.push(E2eArtifactEntry {
        file: E2E_ARTIFACT_ARTIFACTS_INDEX_JSON.to_owned(),
        checksum: sha256_checksum(artifacts_index_json.as_bytes()),
        line_count: None,
    });
    artifacts.sort_by(|left, right| left.file.cmp(&right.file));

    let manifest = E2eEnvelope::new(
        E2E_SCHEMA_MANIFEST,
        PRIVACY_RUN_ID,
        PRIVACY_TS,
        ManifestBody {
            suite: Suite::Fsfs,
            determinism_tier: DeterminismTier::Semantic,
            seed: 7,
            config_hash: format!("privacy-policy:{}", policy.version),
            index_version: None,
            model_versions: vec![ModelVersion {
                name: "redaction-policy".to_owned(),
                revision: policy.version.clone(),
                digest: None,
            }],
            platform: Platform {
                os: std::env::consts::OS.to_owned(),
                arch: std::env::consts::ARCH.to_owned(),
                rustc: "nightly-2026-02-14".to_owned(),
            },
            clock_mode: ClockMode::Simulated,
            tie_break_policy: "doc_id_lexical".to_owned(),
            artifacts,
            duration_ms: 32,
            exit_status,
        },
    );

    PrivacyUnifiedE2eBundle {
        manifest,
        events,
        replay,
        lane_meta,
    }
}

fn assert_privacy_unified_bundle_is_valid(bundle: &PrivacyUnifiedE2eBundle) {
    validate_manifest_envelope(&bundle.manifest).expect("manifest envelope should validate");
    for event in &bundle.events {
        validate_event_envelope(event).expect("event envelope should validate");
    }
}

#[test]
fn privacy_unified_bundle_is_deterministic_for_fixed_inputs() {
    let policy = RedactionPolicy::default();
    let report = LeakReport {
        schema_version: "fsfs.privacy.leak_report.v1".to_owned(),
        policy_version: policy.version.clone(),
        replay_handle: "repro replay --trace-id trace-redaction-001".to_owned(),
        findings: Vec::new(),
    };

    let first = build_privacy_unified_bundle(&report, &policy);
    let second = build_privacy_unified_bundle(&report, &policy);

    assert_eq!(first, second);
    assert_privacy_unified_bundle_is_valid(&first);
}

#[test]
fn corpus_fixtures_cover_secrets_pii_and_policy_boundaries() {
    let policy = RedactionPolicy::default();

    let dropped_log = policy.apply(DataClass::Credential, OutputSurface::Log, RAW_TOKEN);
    assert!(dropped_log.is_none());

    let masked_evidence = policy
        .apply(DataClass::Credential, OutputSurface::Evidence, RAW_TOKEN)
        .expect("credentials must be masked in evidence");
    assert!(masked_evidence.starts_with("<MASKED:"));
    assert_ne!(masked_evidence, RAW_TOKEN);

    let pii_at_boundary = "a".repeat(policy.truncate_max_len);
    let pii_over_boundary = format!("{}{}", pii_at_boundary, "b");

    let unchanged = policy
        .apply(
            DataClass::PersonalData,
            OutputSurface::Explain,
            &pii_at_boundary,
        )
        .expect("explain transform should emit a value");
    assert_eq!(unchanged, pii_at_boundary);

    let truncated = policy
        .apply(
            DataClass::PersonalData,
            OutputSurface::Explain,
            &pii_over_boundary,
        )
        .expect("over-boundary personal data should be truncated");
    assert!(truncated.ends_with("..."));
    assert_ne!(truncated, pii_over_boundary);
    assert_eq!(truncated.len(), policy.truncate_max_len + 3);

    let classes = frankensearch_fsfs::classify_path("/home/alice/.ssh/id_ed25519");
    assert!(classes.contains(&DataClass::PrivateKey));
}

#[test]
#[allow(clippy::too_many_lines)]
fn redaction_assertions_cover_logs_evidence_explain_and_streamed_outputs() {
    let policy = RedactionPolicy::default();
    let replay_entrypoint = ReplayEntrypoint {
        trace_id: "trace-redaction-001".to_owned(),
        client_surface: ReplayClientSurface::Cli,
        manifest_path: Some("/tmp/repro/manifest.json".to_owned()),
        artifact_root: None,
        start_frame_seq: Some(10),
        end_frame_seq: Some(30),
        strict_reason_codes: true,
    };
    replay_entrypoint
        .validate()
        .expect("replay entrypoint should be valid");
    let replay_handle = replay_entrypoint.to_cli_args().join(" ");

    let redacted_query = redacted_or_marker(
        &policy,
        DataClass::Credential,
        OutputSurface::Explain,
        RAW_TOKEN,
    );
    let redacted_path = redacted_or_marker(
        &policy,
        DataClass::UserPath,
        OutputSurface::Evidence,
        "/home/alice/.ssh/id_ed25519",
    );
    let redacted_email = redacted_or_marker(
        &policy,
        DataClass::PersonalData,
        OutputSurface::Evidence,
        RAW_EMAIL,
    );

    let evidence = FsfsEvidenceEvent::new(
        TraceLink::root("trace-redaction-001", "event-redaction-001"),
        FsfsEventFamily::Privacy,
        frankensearch_core::EvidenceRecord::new(
            EvidenceEventType::Decision,
            FsfsReasonCode::PRIVACY_REDACT_APPLIED,
            "redaction policy applied",
            Severity::Info,
            PipelineState::Nominal,
            "privacy_suite",
        ),
        "2026-02-14T00:00:00Z",
    )
    .with_scope_decision(ScopeDecision {
        path: redacted_path,
        decision: ScopeDecisionKind::Exclude,
        reason_code: FsfsReasonCode::PRIVACY_SCOPE_HARD_DENY.to_owned(),
        sensitive_classes: vec!["private_key".to_owned(), "credential".to_owned()],
        persist_allowed: false,
        emit_allowed: false,
        display_allowed: false,
        redaction_profile: policy.version.clone(),
    });
    let evidence_json = serde_json::to_string(&evidence).expect("serialize evidence");

    let explanation = FsfsExplanationPayload::new(redacted_query.clone(), sample_ranking())
        .with_trace(TraceLink::root("trace-redaction-001", "event-explain-001"));
    let explanation_json = explanation.to_cli_json().expect("serialize explanation");

    let stream_frame = StreamFrame::<String>::new(
        "stream-redaction-001",
        1,
        "2026-02-14T00:00:01Z",
        "search",
        StreamEvent::<String>::Started(StreamStartedEvent {
            stream_id: "stream-redaction-001".to_owned(),
            query: format!("{redacted_query} {redacted_email}"),
            format: "jsonl".to_owned(),
        }),
    );
    let stream_ndjson = encode_stream_frame_ndjson(&stream_frame).expect("encode stream frame");

    let crash = OutputEnvelope::<()>::error(
        OutputError::new(
            "io_error",
            format!("failed while handling {redacted_query}"),
            1,
        ),
        OutputMeta::new("search", "json"),
        "2026-02-14T00:00:02Z",
    );
    let crash_json = serde_json::to_string(&crash).expect("serialize crash output");

    let replay_manifest = ReproManifest {
        schema_version: 1,
        pack_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
        trace_id: "trace-redaction-001".to_owned(),
        capture_reason: CaptureReason::Error,
        captured_at: "2026-02-14T00:00:03Z".to_owned(),
        instance: ReproInstance {
            instance_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1S".to_owned(),
            project_key: redacted_or_marker(
                &policy,
                DataClass::UserPath,
                OutputSurface::ReproPack,
                "/home/alice/projects/frankensearch",
            ),
            host_name: "host-a".to_owned(),
            pid: Some(42),
            version: "0.1.0".to_owned(),
            rust_version: Some("nightly".to_owned()),
        },
        artifacts: vec![ReproArtifactEntry {
            filename: "evidence.jsonl".to_owned(),
            size_bytes: 123,
            checksum_xxh3: "7f3a2c1b09de8899".to_owned(),
            content_type: "application/jsonl".to_owned(),
            present: true,
        }],
        retention_tier: RetentionTier::Hot,
        expires_at: None,
        redaction_policy_version: REDACTION_POLICY_VERSION.to_owned(),
    };
    let replay_manifest_json = serde_json::to_string(&replay_manifest).expect("serialize manifest");

    let replay_meta = ReplayMeta {
        mode: ReplayMode::Deterministic,
        seed: Some(7),
        tick_ms: Some(16),
        frame_seq_range: Some(FrameSeqRange {
            first: 10,
            last: 30,
        }),
        event_count: 12,
        trace_duration_ms: 256,
    };
    let replay_meta_json = serde_json::to_string(&replay_meta).expect("serialize replay meta");

    let mut state = SearchInteractionState::new(6);
    state.apply_incremental_query(&redacted_query);
    state.set_results(vec![SearchResultEntry::new(
        "doc-privacy-001",
        "src/privacy.rs",
        "no-secrets-here",
    )]);

    let budget = InteractionBudget::degraded(frankensearch_fsfs::DegradedRetrievalMode::Normal);
    let cycle = InteractionCycleTiming {
        frame_seq: 1,
        input: PhaseTiming {
            phase: LatencyPhase::Input,
            duration: Duration::from_millis(1),
            budget: budget.input_budget,
        },
        update: PhaseTiming {
            phase: LatencyPhase::Update,
            duration: Duration::from_millis(2),
            budget: budget.update_budget,
        },
        render: PhaseTiming {
            phase: LatencyPhase::Render,
            duration: Duration::from_millis(3),
            budget: budget.render_budget,
        },
    };
    let telemetry = state.telemetry_sample(&cycle, &budget);
    let telemetry_json = format!(
        "{{\"interaction_id\":\"{}\",\"frame_seq\":{},\"latency_bucket\":\"{:?}\",\"query_len\":{}}}",
        telemetry.interaction_id,
        telemetry.frame_seq,
        telemetry.latency_bucket,
        telemetry.query_len
    );

    let artifacts = vec![
        ("evidence.json", evidence_json),
        ("explain.json", explanation_json),
        ("stream.ndjson", stream_ndjson),
        ("crash.json", crash_json),
        ("manifest.json", replay_manifest_json),
        ("replay-meta.json", replay_meta_json),
        ("telemetry.json", telemetry_json),
    ];

    let report = build_leak_report(&policy.version, &replay_handle, &artifacts);
    assert!(
        report.findings.is_empty(),
        "leaks found: {:?}",
        report.findings
    );
    assert_eq!(report.policy_version, REDACTION_POLICY_VERSION);
    assert!(report.replay_handle.contains("repro replay --trace-id"));

    let bundle = build_privacy_unified_bundle(&report, &policy);
    assert_privacy_unified_bundle_is_valid(&bundle);
    let artifact_files: Vec<&str> = bundle
        .manifest
        .body
        .artifacts
        .iter()
        .map(|artifact| artifact.file.as_str())
        .collect();
    assert!(artifact_files.contains(&E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL));
    assert!(artifact_files.contains(&E2E_ARTIFACT_REPLAY_COMMAND_TXT));
    assert!(artifact_files.contains(&E2E_ARTIFACT_ARTIFACTS_INDEX_JSON));
    assert!(artifact_files.contains(&"leak_report.json"));
    assert!(artifact_files.contains(&"lane_meta.json"));
    assert!(
        bundle
            .events
            .iter()
            .any(|event| event.body.reason_code.as_deref() == Some(PRIVACY_REASON_PASS))
    );
    assert_eq!(bundle.lane_meta.policy_version, policy.version);
    assert_eq!(bundle.lane_meta.finding_count, 0);

    let report_json = serde_json::to_string(&report).expect("serialize leak report");
    let decoded: LeakReport = serde_json::from_str(&report_json).expect("roundtrip leak report");
    assert_eq!(decoded, report);
}
