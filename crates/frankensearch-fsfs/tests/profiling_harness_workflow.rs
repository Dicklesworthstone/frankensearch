use std::collections::BTreeSet;

use frankensearch_fsfs::{
    CRAWL_INGEST_OPT_TRACK_SCHEMA_VERSION, ITERATION_REASON_ACCEPTED,
    ITERATION_REASON_MULTI_CHANGE, ITERATION_REASON_NO_CHANGE, LeverSnapshot,
    OPPORTUNITY_MATRIX_SCHEMA_VERSION, OneLeverIterationProtocol, OpportunityCandidate,
    OpportunityMatrix, PROFILING_WORKFLOW_SCHEMA_VERSION, ProfileKind, ProfileWorkflow,
    crawl_ingest_optimization_track,
};

#[test]
fn profiling_workflow_covers_flamegraph_heap_and_syscall() {
    let workflow = ProfileWorkflow::for_dataset_profile("medium");

    assert_eq!(workflow.schema_version, PROFILING_WORKFLOW_SCHEMA_VERSION,);
    assert_eq!(workflow.dataset_profile, "medium");
    assert_eq!(workflow.steps.len(), 3);

    let lanes: BTreeSet<ProfileKind> = workflow.steps.iter().map(|step| step.kind).collect();
    assert_eq!(
        lanes,
        BTreeSet::from([
            ProfileKind::Flamegraph,
            ProfileKind::Heap,
            ProfileKind::Syscall,
        ])
    );
}

#[test]
fn opportunity_matrix_emits_deterministic_ranking_table() {
    let matrix = OpportunityMatrix::new(vec![
        OpportunityCandidate {
            id: "query-path".to_owned(),
            summary: "Reduce retrieval/fusion allocations".to_owned(),
            impact: 88,
            confidence: 82,
            effort: 22,
        },
        OpportunityCandidate {
            id: "crawl-path".to_owned(),
            summary: "Shrink syscall pressure during crawl".to_owned(),
            impact: 70,
            confidence: 90,
            effort: 18,
        },
        OpportunityCandidate {
            id: "tui-path".to_owned(),
            summary: "Cut frame overdraw and diff churn".to_owned(),
            impact: 65,
            confidence: 95,
            effort: 12,
        },
    ]);

    assert_eq!(matrix.schema_version, OPPORTUNITY_MATRIX_SCHEMA_VERSION);

    let ranked = matrix.ranked();
    let ordered_ids: Vec<&str> = ranked
        .iter()
        .map(|entry| entry.candidate.id.as_str())
        .collect();

    assert_eq!(ordered_ids, vec!["tui-path", "crawl-path", "query-path"]);
    assert_eq!(ranked[0].rank, 1);
    assert_eq!(ranked[1].rank, 2);
    assert_eq!(ranked[2].rank, 3);
    assert!(ranked[0].ice_score_per_mille >= ranked[1].ice_score_per_mille);
    assert!(ranked[1].ice_score_per_mille >= ranked[2].ice_score_per_mille);
}

#[test]
fn crawl_ingest_track_is_ranked_and_guarded() {
    let track = crawl_ingest_optimization_track();
    assert_eq!(track.schema_version, CRAWL_INGEST_OPT_TRACK_SCHEMA_VERSION);
    assert_eq!(track.hotspots.len(), 5);
    assert_eq!(track.hotspots[0].rank, 1);
    assert!(
        track.hotspots[0].expected_p95_gain_pct >= track.hotspots[0].expected_p50_gain_pct,
        "top hotspot p95 gain should be >= p50 gain"
    );

    let hotspot_ids: BTreeSet<&str> = track
        .hotspots
        .iter()
        .map(|hotspot| hotspot.lever_id.as_str())
        .collect();

    assert_eq!(track.proof_checklist.len(), hotspot_ids.len());
    assert_eq!(track.rollback_guardrails.len(), hotspot_ids.len());

    for proof in &track.proof_checklist {
        assert!(hotspot_ids.contains(proof.lever_id.as_str()));
        assert!(!proof.required_invariants.is_empty());
        assert!(proof.replay_command.contains("--lane ingest"));
    }

    for guardrail in &track.rollback_guardrails {
        assert!(hotspot_ids.contains(guardrail.lever_id.as_str()));
        assert!(
            guardrail.rollback_command.contains("fsfs profile rollback"),
            "rollback command must be explicit"
        );
        assert!(!guardrail.abort_reason_codes.is_empty());
    }
}

#[test]
fn one_lever_protocol_enforces_single_change_contract() {
    let baseline = LeverSnapshot::from_pairs([
        ("query.semantic_fanout", "64"),
        ("query.lexical_fanout", "120"),
        ("tui.frame_budget_ms", "16"),
    ]);

    let accepted = LeverSnapshot::from_pairs([
        ("query.semantic_fanout", "80"),
        ("query.lexical_fanout", "120"),
        ("tui.frame_budget_ms", "16"),
    ]);
    let accepted_validation = OneLeverIterationProtocol::validate(&baseline, &accepted);
    assert!(accepted_validation.accepted);
    assert_eq!(
        accepted_validation.changed_levers,
        vec!["query.semantic_fanout"]
    );
    assert_eq!(accepted_validation.reason_code, ITERATION_REASON_ACCEPTED);

    let unchanged = OneLeverIterationProtocol::validate(&baseline, &baseline);
    assert!(!unchanged.accepted);
    assert_eq!(unchanged.changed_levers, Vec::<String>::new());
    assert_eq!(unchanged.reason_code, ITERATION_REASON_NO_CHANGE);

    let invalid = LeverSnapshot::from_pairs([
        ("query.semantic_fanout", "96"),
        ("query.lexical_fanout", "96"),
        ("tui.frame_budget_ms", "16"),
    ]);
    let invalid_validation = OneLeverIterationProtocol::validate(&baseline, &invalid);
    assert!(!invalid_validation.accepted);
    assert_eq!(
        invalid_validation.changed_levers,
        vec!["query.lexical_fanout", "query.semantic_fanout"]
    );
    assert_eq!(
        invalid_validation.reason_code,
        ITERATION_REASON_MULTI_CHANGE
    );
}

#[test]
fn workflow_artifact_manifest_is_reproducible() {
    let workflow = ProfileWorkflow::for_dataset_profile("small");
    let artifacts = workflow.artifact_manifest("run-2026-02-14");

    assert_eq!(artifacts.len(), 3);
    assert_eq!(artifacts[0].kind, ProfileKind::Flamegraph);
    assert_eq!(artifacts[1].kind, ProfileKind::Heap);
    assert_eq!(artifacts[2].kind, ProfileKind::Syscall);
    assert_eq!(
        artifacts[0].artifact_path,
        "run-2026-02-14/profiles/small/flamegraph.svg"
    );
    assert!(artifacts[0].replay_command.contains("--kind flamegraph"));
    assert!(artifacts[1].replay_command.contains("--kind heap"));
    assert!(artifacts[2].replay_command.contains("--kind syscall"));
}
