//! Query-policy regressions on real admitted vector owners and native graphs.
//! Providers and lexical scoring are deterministic fixtures, not recall evidence.

use super::*;
use frankensearch_fusion::RrfConfig;

fn same_page(left: &[NativeShardedResult], right: &[NativeShardedResult]) {
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right) {
        assert_eq!(
            serde_json::to_value(&left.result).unwrap(),
            serde_json::to_value(&right.result).unwrap()
        );
        assert_eq!(left.result.score.to_bits(), right.result.score.to_bits());
        assert_eq!(
            left.result.fast_score.map(f32::to_bits),
            right.result.fast_score.map(f32::to_bits)
        );
        assert_eq!(
            left.result.quality_score.map(f32::to_bits),
            right.result.quality_score.map(f32::to_bits)
        );
        assert_eq!(
            (left.fast_row, left.quality_row),
            (right.fast_row, right.quality_row)
        );
        assert!(right.result.index.is_none());
    }
}

#[test]
fn explicit_defaults_preserve_eager_rankings_and_both_row_maps() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for semantic in [false, true] {
            let fast = Provider::new("fast", 2, semantic);
            let quality = Provider::new("quality", 3, semantic);
            let a = fast_set(&cx, &fast);
            let b = quality_set(&cx, &quality);
            let lexical = Lexical::new(&["z-fast", "a-late", "lexical-only"]);
            for with_quality in [false, true] {
                for k in [1, 4] {
                    let mut query = a
                        .search_hybrid_progressive(
                            &cx,
                            &fast,
                            with_quality.then_some((&b, &quality as &dyn Embedder)),
                            &lexical,
                            "query",
                            k,
                        )
                        .unwrap()
                        .with_fusion(RrfConfig::default(), 0.7)
                        .unwrap()
                        .with_candidate_multiplier(3)
                        .unwrap()
                        .with_beam_widths(None, None)
                        .unwrap();
                    let NativeShardedSearchPhase::Initial { results, .. } =
                        query.next_phase().await.unwrap().unwrap()
                    else {
                        panic!("initial");
                    };
                    let expected = a
                        .search_hybrid_text(&cx, &fast, &lexical, "query", k)
                        .await
                        .unwrap();
                    same_page(&expected, &results);
                    if with_quality {
                        let NativeShardedSearchPhase::Refined { results, .. } =
                            query.next_phase().await.unwrap().unwrap()
                        else {
                            panic!("refined");
                        };
                        let expected = a
                            .search_hybrid_refined_text(
                                &cx,
                                &fast,
                                (&b, &quality),
                                &lexical,
                                "query",
                                k,
                            )
                            .await
                            .unwrap();
                        same_page(&expected, &results);
                    }
                    assert!(query.next_phase().await.unwrap().is_none());
                }
            }
        }
    });
}

#[test]
fn candidate_multiplier_limits_global_pools_instead_of_multiplying_by_shards() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2, true);
        let quality = Provider::new("quality", 3, true);
        let a = fast_set(&cx, &fast);
        let b = quality_set(&cx, &quality);
        let lexical = Lexical::new(&["z-fast", "mid-fast", "near-fast", "a-late", "q-other"]);
        for multiplier in [1_usize, 2, 5] {
            let mut query = a
                .search_hybrid_progressive(&cx, &fast, Some((&b, &quality)), &lexical, "query", 1)
                .unwrap()
                .with_candidate_multiplier(multiplier)
                .unwrap();
            let NativeShardedSearchPhase::Initial {
                results,
                candidates,
            } = query.next_phase().await.unwrap().unwrap()
            else {
                panic!("initial")
            };
            assert_eq!(results.len(), 1);
            assert_eq!(
                candidates,
                NativePhaseCandidates {
                    fast: multiplier.min(4),
                    quality: 0,
                    lexical: multiplier.min(5),
                }
            );
            let NativeShardedSearchPhase::Refined {
                results,
                candidates,
            } = query.next_phase().await.unwrap().unwrap()
            else {
                panic!("refined")
            };
            assert_eq!(results.len(), 1);
            assert_eq!(
                candidates,
                NativePhaseCandidates {
                    fast: multiplier.min(4),
                    quality: multiplier.min(3),
                    lexical: multiplier.min(5),
                }
            );
        }
        assert_eq!(fast.calls.load(Ordering::SeqCst), 3);
        assert_eq!(quality.calls.load(Ordering::SeqCst), 3);
        assert_eq!(lexical.calls.load(Ordering::SeqCst), 3);
    });
}

#[test]
fn rerank_window_and_multiplier_order_agree_and_keep_the_full_winner_envelope() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2, true);
        let quality = Provider::new("quality", 3, true);
        let a = fast_set(&cx, &fast);
        let b = quality_set(&cx, &quality);
        let lexical = Lexical::new(&["z-fast", "mid-fast", "near-fast", "a-late", "q-other"]);
        let model = CrossEncoder::new("a-late", RerankReply::Ready);
        let query = || {
            a.search_hybrid_progressive(&cx, &fast, Some((&b, &quality)), &lexical, "query", 1)
                .unwrap()
        };
        let mut left = query()
            .with_reranker(&model, &source_text, 4)
            .unwrap()
            .with_candidate_multiplier(1)
            .unwrap()
            .with_beam_widths(Some(1), Some(9))
            .unwrap();
        let mut right = query()
            .with_candidate_multiplier(1)
            .unwrap()
            .with_beam_widths(Some(1), Some(9))
            .unwrap()
            .with_reranker(&model, &source_text, 4)
            .unwrap();
        assert_eq!((left.budget, right.budget), (4, 4));
        assert_eq!(
            (left.policy.fast_ef, left.policy.quality_ef),
            (Some(1), Some(9))
        );
        assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
        for phase in 0..3 {
            let left = left.next_phase().await.unwrap().unwrap();
            let right = right.next_phase().await.unwrap().unwrap();
            match (left, right) {
                (
                    NativeShardedSearchPhase::Initial {
                        results: a,
                        candidates: ac,
                    },
                    NativeShardedSearchPhase::Initial {
                        results: b,
                        candidates: bc,
                    },
                )
                | (
                    NativeShardedSearchPhase::Refined {
                        results: a,
                        candidates: ac,
                    },
                    NativeShardedSearchPhase::Refined {
                        results: b,
                        candidates: bc,
                    },
                ) => {
                    same_page(&a, &b);
                    assert_eq!(ac, bc);
                    assert_eq!(ac.fast, 4);
                    assert_eq!(ac.lexical, 4);
                    assert_eq!(ac.quality, if phase == 0 { 0 } else { 3 });
                }
                (
                    NativeShardedSearchPhase::Reranked {
                        results: a,
                        evaluated: ae,
                        ..
                    },
                    NativeShardedSearchPhase::Reranked {
                        results: b,
                        evaluated: be,
                        ..
                    },
                ) => {
                    same_page(&a, &b);
                    assert_eq!((ae, be), (4, 4));
                    assert_eq!(a[0].result.doc_id, "a-late");
                    assert_eq!(
                        a[0].fast_row,
                        Some(NativeShardRow {
                            shard: 1,
                            physical_row: 1
                        })
                    );
                    assert_eq!(
                        a[0].quality_row,
                        Some(NativeShardRow {
                            shard: 1,
                            physical_row: 0
                        })
                    );
                    assert_eq!(a[0].result.metadata.as_deref().unwrap()["generation"], 7);
                }
                _ => panic!("same requested phase"),
            }
        }
        assert_eq!(model.calls.load(Ordering::SeqCst), 2);
        assert_eq!(fast.calls.load(Ordering::SeqCst), 2);
        assert_eq!(quality.calls.load(Ordering::SeqCst), 2);
        assert_eq!(lexical.calls.load(Ordering::SeqCst), 2);
    });
}

#[test]
fn global_fusion_weights_change_ranks_without_rewriting_raw_scores_or_rows() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for semantic in [false, true] {
            let fast = Provider::new("fast", 2, semantic);
            let quality = Provider::new("quality", 3, semantic);
            let a = fast_set(&cx, &fast);
            let b = quality_set(&cx, &quality);
            let lexical = Lexical::new(&[]);
            for (weight, winner) in [(0.0, "z-fast"), (1.0, "a-late")] {
                let mut query = a
                    .search_hybrid_progressive(
                        &cx,
                        &fast,
                        Some((&b, &quality)),
                        &lexical,
                        "query",
                        5,
                    )
                    .unwrap()
                    .with_fusion(
                        RrfConfig {
                            k: 0.0,
                            ..RrfConfig::default()
                        },
                        weight,
                    )
                    .unwrap()
                    .with_candidate_multiplier(1)
                    .unwrap();
                assert!(query.next_phase().await.unwrap().is_some());
                let NativeShardedSearchPhase::Refined { results, .. } =
                    query.next_phase().await.unwrap().unwrap()
                else {
                    panic!("refined")
                };
                assert_eq!(results[0].result.doc_id, winner);
                assert_eq!(results[0].result.score.to_bits(), 1.0_f32.to_bits());
                assert_eq!(
                    results.len(),
                    5,
                    "one-sided candidates survive endpoint weights"
                );
                for hit in &results {
                    for (set, row) in [(&a, hit.fast_row), (&b, hit.quality_row)] {
                        if let Some(row) = row {
                            assert_eq!(
                                set.shard(row.shard)
                                    .unwrap()
                                    .owner
                                    .doc_id_at(row.physical_row as usize)
                                    .unwrap(),
                                hit.result.doc_id
                            );
                        }
                    }
                    assert!(hit.result.index.is_none());
                    if !semantic {
                        assert_eq!(hit.result.source, ScoreSource::HashControl);
                        assert!(
                            hit.result.fast_score.is_none() && hit.result.quality_score.is_none()
                        );
                    }
                }
                if semantic {
                    let shared = results
                        .iter()
                        .find(|hit| hit.result.doc_id == "z-fast")
                        .unwrap();
                    assert_eq!(
                        (shared.result.fast_score, shared.result.quality_score),
                        (Some(1.0), Some(0.75))
                    );
                }
            }
            let lexical = Lexical::new(&["lexical-only"]);
            let mut query = a
                .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
                .unwrap()
                .with_fusion(
                    RrfConfig {
                        k: 0.0,
                        lexical_weight: 7.0,
                        semantic_weight: 1.0,
                        ..RrfConfig::default()
                    },
                    0.7,
                )
                .unwrap();
            let NativeShardedSearchPhase::Initial { results, .. } =
                query.next_phase().await.unwrap().unwrap()
            else {
                panic!("initial")
            };
            assert_eq!(results[0].result.doc_id, "lexical-only");
            assert_eq!(results[0].result.score.to_bits(), 7.0_f32.to_bits());
            assert!(results[0].fast_row.is_none() && results[0].quality_row.is_none());
        }
    });
}

#[test]
fn invalid_policies_and_overflow_refuse_before_any_query_work() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2, true);
        let a = fast_set(&cx, &fast);
        let lexical = Lexical::new(&[]);
        let model = CrossEncoder::new("a-late", RerankReply::Ready);
        let query = |k| {
            a.search_hybrid_progressive(&cx, &fast, None, &lexical, "query", k)
                .unwrap()
        };
        assert!(query(0).with_candidate_multiplier(0).is_err());
        assert!(query(2).with_candidate_multiplier(usize::MAX).is_err());
        for beams in [(Some(0), None), (None, Some(0))] {
            assert!(query(0).with_beam_widths(beams.0, beams.1).is_err());
        }
        for weight in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            assert!(query(0).with_fusion(RrfConfig::default(), weight).is_err());
        }
        for value in [f64::NAN, f64::INFINITY, -1.0] {
            assert!(
                query(0)
                    .with_fusion(
                        RrfConfig {
                            k: value,
                            ..RrfConfig::default()
                        },
                        0.7
                    )
                    .is_err()
            );
        }
        for value in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MAX] {
            assert!(
                query(0)
                    .with_fusion(
                        RrfConfig {
                            lexical_weight: value,
                            ..RrfConfig::default()
                        },
                        0.7
                    )
                    .is_err()
            );
            assert!(
                query(0)
                    .with_fusion(
                        RrfConfig {
                            semantic_weight: value,
                            ..RrfConfig::default()
                        },
                        0.7
                    )
                    .is_err()
            );
        }
        let window = usize::MAX / 3;
        assert!(
            query(1)
                .with_reranker(&model, &source_text, window)
                .unwrap()
                .with_candidate_multiplier(4)
                .is_err()
        );
        assert!(
            query(1)
                .with_candidate_multiplier(4)
                .unwrap()
                .with_reranker(&model, &source_text, window)
                .is_err()
        );
        assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
        assert_eq!(lexical.calls.load(Ordering::SeqCst), 0);
        assert_eq!(model.calls.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn configuration_is_frozen_after_initial_and_cancellation_precedes_policy_errors() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2, true);
        let quality = Provider::new("quality", 3, true);
        let a = fast_set(&cx, &fast);
        let b = quality_set(&cx, &quality);
        let lexical = Lexical::new(&["z-fast"]);
        for cancelled in [false, true] {
            for method in 0..3 {
                let mut query = a
                    .search_hybrid_progressive(
                        &cx,
                        &fast,
                        Some((&b, &quality)),
                        &lexical,
                        "query",
                        1,
                    )
                    .unwrap();
                if cancelled {
                    cx.set_cancel_requested(true);
                } else {
                    assert!(query.next_phase().await.unwrap().is_some());
                }
                let outcome = match method {
                    0 => query.with_candidate_multiplier(0),
                    1 => query.with_beam_widths(Some(0), None),
                    _ => query.with_fusion(RrfConfig::default(), f32::NAN),
                };
                if cancelled {
                    assert!(matches!(outcome, Err(SearchError::Cancelled { .. })));
                    cx.set_cancel_requested(false);
                } else {
                    assert!(
                        matches!(outcome, Err(SearchError::InvalidConfig { ref field, .. })
                        if field == "native_ann.shards.configuration")
                    );
                }
                assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
            }
        }
        assert_eq!(fast.calls.load(Ordering::SeqCst), 3);
        assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn zero_quality_weight_retains_required_failure_and_original_initial_page() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2, true);
        let mut quality = Provider::new("quality", 3, true);
        quality.reply = Reply::Failed;
        let a = fast_set(&cx, &fast);
        let b = quality_set(&cx, &quality);
        let lexical = Lexical::new(&["z-fast"]);
        let mut query = a
            .search_hybrid_progressive(&cx, &fast, Some((&b, &quality)), &lexical, "query", 1)
            .unwrap()
            .with_fusion(RrfConfig::default(), 0.0)
            .unwrap()
            .with_candidate_multiplier(1)
            .unwrap()
            .with_beam_widths(Some(1), Some(2))
            .unwrap();
        let NativeShardedSearchPhase::Initial { results, .. } =
            query.next_phase().await.unwrap().unwrap()
        else {
            panic!("initial")
        };
        let NativeShardedSearchPhase::RefinementFailed {
            initial_results, ..
        } = query.next_phase().await.unwrap().unwrap()
        else {
            panic!("required quality cannot be skipped")
        };
        same_page(&results, &initial_results);
        assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
        assert!(query.is_finished());
    });
}

#[test]
#[cfg(feature = "quill")]
fn tuned_scoped_retrieval_and_reranking_keep_only_eligible_rows() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Provider::new("fast", 2, true);
        let quality = Provider::new("quality", 3, true);
        let a = fast_set(&cx, &fast);
        let b = quality_set(&cx, &quality);
        let lexical = Lexical::new(&["a-late"]);
        let allowed = BTreeSet::from(["a-late".to_owned()]);
        let model = CrossEncoder::new("a-late", RerankReply::Ready);
        let mut query = a
            .search_hybrid_progressive(&cx, &fast, Some((&b, &quality)), &lexical, "query", 1)
            .unwrap()
            .with_allowed_documents(&allowed)
            .unwrap()
            .with_candidate_multiplier(1)
            .unwrap()
            .with_beam_widths(Some(1), Some(1))
            .unwrap()
            .with_fusion(
                RrfConfig {
                    k: 0.0,
                    ..RrfConfig::default()
                },
                1.0,
            )
            .unwrap()
            .with_reranker(&model, &source_text, 4)
            .unwrap();
        let mut phases = 0;
        while let Some(phase) = query.next_phase().await.unwrap() {
            let results = match phase {
                NativeShardedSearchPhase::Initial { results, .. }
                | NativeShardedSearchPhase::Refined { results, .. }
                | NativeShardedSearchPhase::Reranked { results, .. } => results,
                _ => panic!("eligible query succeeds"),
            };
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].result.doc_id, "a-late");
            assert_eq!(
                results[0].fast_row,
                Some(NativeShardRow {
                    shard: 1,
                    physical_row: 1
                })
            );
            if phases > 0 {
                assert_eq!(
                    results[0].quality_row,
                    Some(NativeShardRow {
                        shard: 1,
                        physical_row: 0
                    })
                );
            }
            phases += 1;
        }
        assert_eq!(phases, 3);
        assert_eq!(model.calls.load(Ordering::SeqCst), 1);
    });
}
