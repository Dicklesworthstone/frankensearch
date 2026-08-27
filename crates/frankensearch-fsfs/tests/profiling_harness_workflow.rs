use std::collections::BTreeSet;

use frankensearch_fsfs::{
    CRAWL_INGEST_OPT_TRACK_SCHEMA_VERSION, CorpusProfileSnapshot, CorrectnessAssertion,
    CorrectnessProofKind, HostProfileSnapshot, ITERATION_REASON_ACCEPTED,
    ITERATION_REASON_MULTI_CHANGE, ITERATION_REASON_NO_CHANGE, LatencyDecomposition, LeverSnapshot,
    ModelCacheState, OPPORTUNITY_MATRIX_SCHEMA_VERSION, OneLeverIterationProtocol,
    OpportunityCandidate, OpportunityMatrix, PROFILING_WORKFLOW_SCHEMA_VERSION, PhaseObservation,
    ProfileKind, ProfileWorkflow, QUERY_LATENCY_OPT_SCHEMA_VERSION, QueryPhase,
    SELF_CALIBRATING_PROFILE_SCHEMA_VERSION, SearchProfileMeasurements,
    SelfCalibratingProfileArtifactKind, SelfCalibratingProfileInput, SelfCalibratingProfileReport,
    VerificationProtocol, VerificationResult, crawl_ingest_optimization_track,
    query_path_lever_catalog, query_path_opportunity_matrix,
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
        assert_ne!(proof.required_invariants, [] as [std::string::String; 0]);
        assert!(
            proof.replay_command.contains("benchmark_baseline_matrix"),
            "replay must name the deterministic baseline matrix, not a fictional subcommand"
        );
        assert!(
            !proof.replay_command.contains("fsfs profile"),
            "fsfs has no profile subcommand (bd-rh0t)"
        );
    }

    for guardrail in &track.rollback_guardrails {
        assert!(hotspot_ids.contains(guardrail.lever_id.as_str()));
        assert!(
            guardrail.rollback_command.contains("git revert")
                && guardrail
                    .rollback_command
                    .contains("benchmark_baseline_matrix"),
            "rollback command must be a source revert plus the baseline proof"
        );
        assert!(
            !guardrail.rollback_command.contains("fsfs profile"),
            "fsfs has no profile subcommand (bd-rh0t)"
        );
        assert_ne!(guardrail.abort_reason_codes, [] as [std::string::String; 0]);
    }
}

#[test]
fn query_latency_track_matrix_and_catalog_stay_in_sync() {
    let matrix = query_path_opportunity_matrix();
    assert_eq!(matrix.schema_version, OPPORTUNITY_MATRIX_SCHEMA_VERSION);
    assert!(
        !matrix.candidates.is_empty(),
        "query matrix must not be empty"
    );

    let ranked = matrix.ranked();
    assert_eq!(
        ranked[0].candidate.id, "vector_search.scratch_buffer_reuse",
        "top-ranked query lever drifted; revisit expected optimization order"
    );

    let matrix_ids: BTreeSet<&str> = matrix.candidates.iter().map(|c| c.id.as_str()).collect();
    let catalog = query_path_lever_catalog();
    let catalog_ids: BTreeSet<&str> = catalog.iter().map(|lever| lever.id.as_str()).collect();

    assert_eq!(matrix_ids, catalog_ids, "matrix and catalog should match");
    assert_eq!(catalog_ids.len(), 8, "unexpected query lever count");
}

#[test]
fn query_latency_decomposition_reports_over_budget_phase() {
    let decomposition = LatencyDecomposition::new(
        vec![
            PhaseObservation {
                phase: QueryPhase::Canonicalize,
                actual_us: 120,
                budget_us: QueryPhase::Canonicalize.default_budget_us(),
                skipped: false,
            },
            PhaseObservation {
                phase: QueryPhase::FastEmbed,
                actual_us: 700,
                budget_us: QueryPhase::FastEmbed.default_budget_us(),
                skipped: false,
            },
            PhaseObservation {
                phase: QueryPhase::FastVectorSearch,
                actual_us: 7_600,
                budget_us: QueryPhase::FastVectorSearch.default_budget_us(),
                skipped: false,
            },
            PhaseObservation {
                phase: QueryPhase::Fuse,
                actual_us: 480,
                budget_us: QueryPhase::Fuse.default_budget_us(),
                skipped: false,
            },
            PhaseObservation {
                phase: QueryPhase::QualityEmbed,
                actual_us: 0,
                budget_us: QueryPhase::QualityEmbed.default_budget_us(),
                skipped: true,
            },
        ],
        12,
        4_096,
    );

    assert_eq!(
        decomposition.schema_version,
        QUERY_LATENCY_OPT_SCHEMA_VERSION
    );
    assert_eq!(decomposition.initial_path_us(), 8_900);
    assert_eq!(decomposition.refinement_path_us(), 0);
    assert_eq!(
        decomposition.verdict_reason_code(),
        "query.latency.single_phase_over_budget"
    );

    let over_budget = decomposition.over_budget_phases();
    assert_eq!(over_budget.len(), 1);
    assert_eq!(over_budget[0].phase, QueryPhase::FastVectorSearch);
    assert_eq!(over_budget[0].overshoot_us(), 2_600);
}

#[test]
fn query_latency_verification_protocol_covers_catalog_and_rejects_failure() {
    let protocol = VerificationProtocol::default_protocol();
    assert_eq!(protocol.schema_version, QUERY_LATENCY_OPT_SCHEMA_VERSION);
    let epsilon: f64 = protocol
        .score_epsilon_str
        .parse()
        .expect("epsilon should parse");
    assert!(epsilon > 0.0);

    let catalog_ids: BTreeSet<String> = query_path_lever_catalog()
        .into_iter()
        .map(|lever| lever.id)
        .collect();
    let protocol_ids: BTreeSet<String> = protocol.required_lever_ids.iter().cloned().collect();
    assert_eq!(catalog_ids, protocol_ids);

    let verification = VerificationResult::from_assertions(
        "vector_search.parallel_threshold_tuning",
        vec![
            CorrectnessAssertion {
                lever_id: "vector_search.parallel_threshold_tuning".to_owned(),
                proof_kind: CorrectnessProofKind::RankPreserving,
                test_corpus_ids: vec!["golden_100".to_owned()],
                assertion: "ranking order preserved".to_owned(),
                passed: true,
                reason_code: "opt.assert.rank_preserving.passed".to_owned(),
            },
            CorrectnessAssertion {
                lever_id: "vector_search.parallel_threshold_tuning".to_owned(),
                proof_kind: CorrectnessProofKind::RankPreserving,
                test_corpus_ids: vec!["adversarial_unicode".to_owned()],
                assertion: "ranking order preserved".to_owned(),
                passed: false,
                reason_code: "opt.assert.rank_preserving.rank_swap_at_position_3".to_owned(),
            },
        ],
    );

    assert!(!verification.passed);
    assert_eq!(verification.failure_count(), 1);
    assert_eq!(verification.reason_code, "opt.verify.failed");
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
    // The replay command is the step's own profiler invocation (bd-rh0t):
    // re-running it deterministically reproduces the artifact, and no
    // fictional `fsfs profile` subcommand is ever advertised.
    assert!(artifacts[0].replay_command.contains("flamegraph"));
    assert!(artifacts[1].replay_command.contains("heaptrack"));
    assert!(artifacts[2].replay_command.contains("strace"));
    for artifact in &artifacts {
        assert!(!artifact.replay_command.contains("fsfs profile"));
    }
}

#[test]
fn self_calibrating_profile_report_exposes_recommendation_artifacts() {
    let report = SelfCalibratingProfileReport::from_input(SelfCalibratingProfileInput {
        run_id: "e2e-self-cal".to_owned(),
        host: HostProfileSnapshot {
            host_id: "csd".to_owned(),
            logical_cpus: 24,
            total_memory_mib: 128_000,
            available_memory_mib: 96_000,
            simd_lanes: 8,
            model_cache_state: ModelCacheState::Cold,
        },
        corpus: CorpusProfileSnapshot {
            corpus_id: "workspace_sample".to_owned(),
            document_count: 60_000,
            total_bytes: 480_000_000,
            cluster_count: 5,
            representative_query_count: 30,
        },
        measurements: SearchProfileMeasurements {
            phase1_p95_us: 11_600,
            phase2_p95_us: 171_000,
            vector_search_docs_per_ms: 760,
            peak_memory_mib: 2_048,
            observed_candidate_multiplier: 3,
            model_cache_warm_ms: 22,
            model_cache_cold_ms: 620,
            quality_uplift_basis_points: 900,
        },
        requested_limit: 12,
    });

    assert_eq!(
        report.schema_version,
        SELF_CALIBRATING_PROFILE_SCHEMA_VERSION
    );
    assert_eq!(report.recommended_config.candidate_multiplier, 2);
    assert_eq!(report.recommended_config.hnsw_threshold, 25_000);
    assert_eq!(report.recommended_config.quality_timeout_ms, 196);
    assert!(
        report
            .reason_codes
            .contains(&"profile.self_calibrating.phase2_over_budget".to_owned())
    );
    assert!(
        report
            .reason_codes
            .contains(&"profile.self_calibrating.model_cache_cold".to_owned())
    );
    assert!(report.artifacts.iter().any(|artifact| artifact.kind
        == SelfCalibratingProfileArtifactKind::StructuredJsonl
        && artifact.artifact_path.ends_with("profile-events.jsonl")
        && artifact.replay_command.contains("--run-id e2e-self-cal")));
}
