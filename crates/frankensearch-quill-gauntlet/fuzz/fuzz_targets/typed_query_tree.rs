#![no_main]

use asupersync::{Cx, test_utils::run_test_with_cx};
use frankensearch_quill::QuillIndexError;
use frankensearch_quill_gauntlet::{
    ComparatorConfig, ComparisonMode, ComparisonReport, ComparisonStatus, DifferentialCase,
    DifferentialHarness, DivergenceClass, GauntletEngine, GauntletError, RankClass,
    ScoreEpsilonReason, TYPED_QUERY_FUZZ_MAX_INPUT_BYTES, TYPED_QUERY_FUZZ_SHRINK_FUEL,
    TypedQueryFailureFingerprint, TypedQueryOracleBehavior, TypedQueryTree,
    materialize_typed_query_fuzz_workload, persist_typed_query_fuzz_replay, scalar_g1a_fuzz_pair,
    typed_query_failure_divergence_class,
};
use libfuzzer_sys::fuzz_target;

async fn run_supported_case(
    cx: &Cx,
    documents: &[frankensearch_core::IndexableDocument],
    case: &DifferentialCase,
    comparator_config: ComparatorConfig,
) -> Result<ComparisonReport, GauntletError> {
    let (subject, oracle) = scalar_g1a_fuzz_pair(cx, documents).await?;
    DifferentialHarness::new(ComparisonMode::CrossEngine, comparator_config)
        .run(cx, &subject, &oracle, case)
        .await
        .map(|run| run.comparison)
}

fn assert_exact_refusal(error: GauntletError, expected_detail: &str, context: &str) {
    match error {
        GauntletError::Quill(QuillIndexError::UnsupportedQuery { detail })
            if detail == expected_detail => {}
        other => panic!(
            "{context} must be the exact typed UnsupportedQuery({expected_detail:?}) refusal, got {other:?}"
        ),
    }
}

fn assert_nonfinite_oracle_refusal(
    error: GauntletError,
    documents: &[frankensearch_quill_gauntlet::GeneratedDocument],
) {
    let GauntletError::InvalidObservation { reason } = error else {
        panic!("non-finite boost must fail as an oracle observation refusal");
    };
    let prefix = "oracle.hits has a non-finite score for ";
    let Some(doc_id) = reason.strip_prefix(prefix) else {
        panic!("non-finite boost must retain the exact oracle hits refusal, got {reason:?}");
    };
    assert!(
        documents.iter().any(|document| document.id == doc_id),
        "non-finite oracle refusal must name an indexed document, got {doc_id:?}"
    );
}

fn assert_reviewed_oversized_lowering(report: &ComparisonReport) {
    assert_eq!(report.status, ComparisonStatus::Classified);
    assert_eq!(report.rank_class, RankClass::RankExact);
    assert_eq!(report.divergences.len(), 1);
    assert_eq!(
        report.divergences[0].class,
        DivergenceClass::OversizedQueryToken
    );
    assert_eq!(
        report.divergences[0].pointer,
        "/comparison/subject/ast_differences/0"
    );
}

async fn shrink_preserving_fingerprint(
    cx: &Cx,
    workload: &frankensearch_quill_gauntlet::TypedQueryFuzzWorkload,
    documents: &[frankensearch_core::IndexableDocument],
    tree: TypedQueryTree,
    case: DifferentialCase,
    report: ComparisonReport,
    comparator_config: ComparatorConfig,
) -> (TypedQueryTree, DifferentialCase, ComparisonReport, usize) {
    let original_fingerprint = TypedQueryFailureFingerprint::from_report(&report);
    let mut current_tree = tree;
    let mut current_case = case;
    let mut current_report = report;
    let mut candidates_evaluated = 0_usize;
    let mut reduction_steps = 0_usize;

    while candidates_evaluated < TYPED_QUERY_FUZZ_SHRINK_FUEL {
        let mut reduced = false;
        for candidate_tree in current_tree.shrink_candidates() {
            if candidates_evaluated == TYPED_QUERY_FUZZ_SHRINK_FUEL {
                break;
            }
            candidates_evaluated += 1;
            let candidate_case = workload.case_for_ast(candidate_tree);
            let candidate_report = run_supported_case(
                cx,
                documents,
                &candidate_case,
                comparator_config,
            )
            .await
            .unwrap_or_else(|error| {
                panic!(
                    "AST shrink candidate must execute before comparison: provenance={} candidate={candidate_tree:?} error={error}",
                    workload.provenance_for_ast(candidate_tree),
                )
            });
            if candidate_report.status == ComparisonStatus::Failed
                && TypedQueryFailureFingerprint::from_report(&candidate_report)
                    == original_fingerprint
            {
                current_tree = candidate_tree;
                current_case = candidate_case;
                current_report = candidate_report;
                reduction_steps += 1;
                reduced = true;
                break;
            }
        }
        if !reduced {
            break;
        }
    }

    assert_eq!(
        TypedQueryFailureFingerprint::from_report(&current_report),
        original_fingerprint,
        "AST shrink must retain the original failure class, direction, and signature"
    );
    (current_tree, current_case, current_report, reduction_steps)
}

fuzz_target!(|input: &[u8]| {
    if input.len() > TYPED_QUERY_FUZZ_MAX_INPUT_BYTES {
        return;
    }
    let input = input.to_vec();
    run_test_with_cx(move |cx| async move {
        let workload = materialize_typed_query_fuzz_workload(&input)
            .expect("bounded deterministic typed-query fuzz workload");
        let tree = workload.ast;
        let case = workload.case.clone();
        let indexable = workload
            .documents
            .iter()
            .cloned()
            .map(frankensearch_core::IndexableDocument::from)
            .collect::<Vec<_>>();
        let comparator_config = ComparatorConfig::default()
            .with_score_epsilon_reason(ScoreEpsilonReason::SummationAssociation);

        if let Some(expected_detail) = tree.exact_refusal_detail() {
            let (subject, oracle) = scalar_g1a_fuzz_pair(&cx, &indexable)
                .await
                .expect("fresh committed pair for the typed-refusal lane");
            let subject_error = GauntletEngine::observe(&subject, &cx, &case)
                .await
                .expect_err("unsupported AST must refuse in Quill before comparison");
            assert_exact_refusal(subject_error, expected_detail, "Quill observation");
            GauntletEngine::observe(&oracle, &cx, &case)
                .await
                .unwrap_or_else(|error| {
                    panic!(
                        "pinned Tantivy oracle must execute the declared unsupported AST: provenance={} error={error}",
                        workload.provenance_for_ast(tree),
                    )
                });
            let harness_error =
                DifferentialHarness::new(ComparisonMode::CrossEngine, comparator_config)
                    .run(&cx, &subject, &oracle, &case)
                    .await
                    .expect_err("typed Quill refusal must not create a one-sided comparison");
            assert_exact_refusal(harness_error, expected_detail, "harness observation");
            return;
        }

        let result = if tree.is_malformed() {
            let (subject, oracle) = scalar_g1a_fuzz_pair(&cx, &indexable)
                .await
                .expect("fresh committed pair for the lenient malformed grammar lane");
            let subject_observation = GauntletEngine::observe(&subject, &cx, &case)
                .await
                .unwrap_or_else(|error| {
                    panic!(
                        "Quill parse_lenient must recover malformed typed grammar: provenance={} error={error}",
                        workload.provenance_for_ast(tree),
                    )
                });
            let oracle_observation = GauntletEngine::observe(&oracle, &cx, &case)
                .await
                .unwrap_or_else(|error| {
                    panic!(
                        "pinned Tantivy oracle must accept malformed typed grammar: provenance={} error={error}",
                        workload.provenance_for_ast(tree),
                    )
                });
            let asymmetry = subject
                .classify_typed_query_lenient_asymmetry(tree, &case.query, &oracle_observation)
                .expect("malformed grammar must classify the Quill recovery/oracle acceptance asymmetry");
            assert!(
                !asymmetry.quill_diagnostic_kinds.is_empty(),
                "the typed malformed lane must retain Quill recovery diagnostics"
            );
            assert_eq!(
                asymmetry.oracle_behavior,
                TypedQueryOracleBehavior::AcceptedWithoutAstDifferences,
                "the exact oracle behavior is successful observation without AST diagnostics"
            );
            assert!(
                subject_observation.ast_differences.is_empty(),
                "malformed syntax diagnostics must not be silently relabeled as a reviewed lowering"
            );
            DifferentialHarness::new(ComparisonMode::CrossEngine, comparator_config)
                .run(&cx, &subject, &oracle, &case)
                .await
                .map(|run| run.comparison)
        } else {
            run_supported_case(&cx, &indexable, &case, comparator_config).await
        };

        if tree.is_nonfinite_boost() {
            assert_nonfinite_oracle_refusal(
                result.expect_err("non-finite oracle scores must fail closed before comparison"),
                &workload.documents,
            );
            return;
        }
        let report = result.unwrap_or_else(|error| {
            panic!(
                "supported or lenient malformed AST failed before comparison: provenance={} query={:?} error={error}",
                workload.provenance_for_ast(tree),
                case.query,
            )
        });

        if tree.is_reviewed_oversized_lowering() {
            assert_reviewed_oversized_lowering(&report);
            return;
        }

        match report.status {
            ComparisonStatus::Exact => {}
            ComparisonStatus::Classified => assert!(
                !report.divergences.is_empty()
                    && report.divergences.iter().all(|divergence| {
                        matches!(
                            divergence.class,
                            DivergenceClass::ScoreEpsilon | DivergenceClass::TieOrder
                        )
                    }),
                "only reviewed automatic comparison classes may pass externally: provenance={} query={:?} divergences={:?}",
                workload.provenance_for_ast(tree),
                case.query,
                report.divergences,
            ),
            ComparisonStatus::Failed => {
                let original_fingerprint = TypedQueryFailureFingerprint::from_report(&report);
                let original_divergence_class = typed_query_failure_divergence_class(&report)
                    .expect("failed comparison must retain a non-automatic divergence class");
                let (minimized_tree, minimized_case, minimized_report, reduction_steps) =
                    shrink_preserving_fingerprint(
                        &cx,
                        &workload,
                        &indexable,
                        tree,
                        case,
                        report,
                        comparator_config,
                    )
                    .await;
                let minimized_fingerprint =
                    TypedQueryFailureFingerprint::from_report(&minimized_report);
                let minimized_divergence_class =
                    typed_query_failure_divergence_class(&minimized_report)
                        .expect("minimized failure must retain a non-automatic divergence class");
                assert_eq!(
                    minimized_fingerprint, original_fingerprint,
                    "minimization must retain the exact divergence fingerprint"
                );
                assert_eq!(
                    minimized_divergence_class, original_divergence_class,
                    "minimization must retain the exact divergence class"
                );
                let minimized_replay =
                    frankensearch_quill_gauntlet::TypedQueryFuzzReplay::from_failure(
                        &workload,
                        minimized_tree,
                        &minimized_report,
                    )
                    .expect("minimized replay must reconstruct its original corpus and query");
                assert_eq!(
                    minimized_replay.minimized_query, minimized_case.query,
                    "persisted minimized query must be the shrunk differential case query"
                );
                let replay_artifact = persist_typed_query_fuzz_replay(
                    std::path::Path::new("artifacts"),
                    &minimized_replay,
                )
                .expect("persist minimized replay under its corpus and fingerprint identity");
                let replayed_workload = replay_artifact
                    .replay_workload()
                    .expect("consume the descriptor-bound minimized replay");
                assert_eq!(
                    replayed_workload.case.query, minimized_case.query,
                    "descriptor-bound replay must reconstruct the shrunk query"
                );
                let replay_key = replay_artifact
                    .artifact_key()
                    .expect("describe the descriptor-bound minimized replay");
                panic!(
                    "unclassified Quill/Tantivy divergence: provenance={} original_fingerprint={original_fingerprint:?} original_divergence_class={original_divergence_class:?} minimized_replay_key={replay_key} minimized_replay={minimized_replay:?} shrink_steps={reduction_steps}",
                    workload.provenance_for_ast(minimized_tree),
                );
            }
        }
    });
});
