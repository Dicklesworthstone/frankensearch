//! Method-bound Quill cancellation contract, entry-phase matrix (bd-fjpu).
//!
//! Every public Quill search boundary invoked with a cancelled `Cx` must
//! return its typed `Cancelled { phase }` error, leave the published
//! snapshot untouched (`Arc` identity), and succeed identically on a fresh
//! retry once the cancellation clears — proving cancellation is a real
//! engine contract, not a spy-only replay. A noncancelled control on the
//! same corpus anchors the expected results.
//!
//! The feature-gated conformance cases additionally request cancellation from
//! deterministic checkpoints *inside* collection, hydration, and publication.
//! Those checkpoints flip the real request `Cx`; public methods still discover
//! and return their own typed cancellation outcome.

use std::sync::Arc;

use asupersync::Cx;
use frankensearch_core::{
    IndexableDocument, LexicalHydrationContext, LexicalRead, LexicalWrite, ScoredResult,
    SearchError,
};
use frankensearch_quill::QuillConfig;
#[cfg(feature = "conformance-internals")]
use frankensearch_quill::index::ConformanceCancellationStage;
use frankensearch_quill::index::{QuillIndex, QuillIndexError};
use frankensearch_quill::snippet::SnippetConfig;

const QUERY: &str = "alpha";
const LIMIT: usize = 10;

fn deterministic_config() -> QuillConfig {
    QuillConfig {
        deterministic_ingest: true,
        ..QuillConfig::default()
    }
}

/// Build a committed in-memory index with three documents matching `alpha`.
async fn fixture_index(cx: &Cx) -> QuillIndex {
    let index = QuillIndex::in_memory(deterministic_config()).expect("in-memory index");
    LexicalWrite::index_documents(
        &index,
        cx,
        &[
            IndexableDocument::new("doc-a", "alpha first document").with_metadata("ordinal", "a"),
            IndexableDocument::new("doc-b", "alpha second document")
                .with_metadata("lang", "rust")
                .with_metadata("ordinal", "b"),
            IndexableDocument::new("doc-c", "alpha third document").with_metadata("ordinal", "c"),
        ],
    )
    .await
    .expect("ingest fixture corpus");
    LexicalWrite::commit(&index, cx)
        .await
        .expect("commit fixture corpus");
    index
}

/// Assert a sync boundary rejects a cancelled `Cx` with the expected typed
/// phase while leaving the published snapshot byte-identical (Arc identity),
/// then succeeds after the cancellation clears.
fn assert_sync_boundary_cancels<T: std::fmt::Debug>(
    index: &QuillIndex,
    cx: &Cx,
    boundary: &str,
    expected_phase: &str,
    call: impl Fn(&QuillIndex, &Cx) -> Result<T, QuillIndexError>,
) -> T {
    let snapshot_before = index.search_snapshot();

    cx.set_cancel_requested(true);
    let error = call(index, cx).expect_err(&format!("{boundary}: cancelled Cx must be rejected"));
    let QuillIndexError::Cancelled { phase } = error else {
        panic!("{boundary}: expected typed cancellation, got {error:?}");
    };
    assert_eq!(
        phase, expected_phase,
        "{boundary}: cancellation phase is part of the contract"
    );
    assert!(
        Arc::ptr_eq(&snapshot_before, &index.search_snapshot()),
        "{boundary}: a cancelled query must not perturb the published snapshot"
    );

    cx.set_cancel_requested(false);
    let value = call(index, cx)
        .unwrap_or_else(|error| panic!("{boundary}: fresh retry must succeed, got {error:?}"));
    assert!(
        Arc::ptr_eq(&snapshot_before, &index.search_snapshot()),
        "{boundary}: a successful retry reads the same published snapshot"
    );
    value
}

#[cfg(feature = "conformance-internals")]
fn assert_results_identical(actual: &[ScoredResult], expected: &[ScoredResult], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: result count");
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.doc_id, expected.doc_id, "{context}: document order");
        assert_eq!(
            actual.score.to_bits(),
            expected.score.to_bits(),
            "{context}: score bits"
        );
        assert_eq!(
            actual.lexical_score.map(f32::to_bits),
            expected.lexical_score.map(f32::to_bits),
            "{context}: lexical score bits"
        );
        assert_eq!(actual.metadata, expected.metadata, "{context}: metadata");
    }
}

async fn candidate_parts(
    index: &QuillIndex,
    cx: &Cx,
) -> (Vec<ScoredResult>, Option<LexicalHydrationContext>) {
    LexicalRead::search_candidates(index, cx, QUERY, LIMIT)
        .await
        .expect("fusion candidates")
        .into_parts()
}

#[cfg(feature = "conformance-internals")]
async fn cancellation_search(
    index: &QuillIndex,
    cx: &Cx,
    candidate_boundary: bool,
) -> Result<Vec<ScoredResult>, SearchError> {
    if candidate_boundary {
        let batch = LexicalRead::search_candidates(index, cx, QUERY, LIMIT).await?;
        Ok(batch.into_parts().0)
    } else {
        LexicalRead::search(index, cx, QUERY, LIMIT).await
    }
}

#[test]
fn every_sync_search_boundary_rejects_a_cancelled_cx_and_retries_clean() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = fixture_index(&cx).await;

        // Noncancelled control first: the corpus genuinely matches.
        let control = index
            .search_results(&cx, QUERY, LIMIT)
            .expect("control search");
        assert_eq!(control.len(), 3, "control must match all three documents");

        let paginated =
            assert_sync_boundary_cancels(&index, &cx, "search_paginated", "search", |index, cx| {
                index.search_paginated(cx, QUERY, LIMIT, 0, true)
            });
        assert_eq!(paginated.hits.len(), 3);

        let results =
            assert_sync_boundary_cancels(&index, &cx, "search_results", "search", |index, cx| {
                index.search_results(cx, QUERY, LIMIT)
            });
        // The retry must reproduce the control exactly: ids, order, scores.
        assert_eq!(results.len(), control.len());
        for (retried, control) in results.iter().zip(control.iter()) {
            assert_eq!(retried.doc_id, control.doc_id);
            assert_eq!(retried.score.to_bits(), control.score.to_bits());
        }

        let doc_ids =
            assert_sync_boundary_cancels(&index, &cx, "search_doc_ids", "search", |index, cx| {
                index.search_doc_ids(cx, QUERY, LIMIT)
            });
        assert_eq!(doc_ids.len(), 3);

        let snippets = assert_sync_boundary_cancels(
            &index,
            &cx,
            "search_with_snippets",
            "search",
            |index, cx| index.search_with_snippets(cx, QUERY, LIMIT, &SnippetConfig::default()),
        );
        assert_eq!(snippets.len(), 3);

        let collected = assert_sync_boundary_cancels(
            &index,
            &cx,
            "collect_docids",
            "collect_docids",
            |index, cx| index.collect_docids(cx, QUERY),
        );
        assert_eq!(collected.len(), 3);
    });
}

#[test]
fn async_lexical_boundaries_reject_a_cancelled_cx_and_retry_clean() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = fixture_index(&cx).await;
        let snapshot_before = index.search_snapshot();

        // Control results for the trait-level search boundary.
        let control = LexicalRead::search(&index, &cx, QUERY, LIMIT)
            .await
            .expect("control trait search");
        assert_eq!(control.len(), 3);

        // LexicalRead::search with a cancelled Cx.
        cx.set_cancel_requested(true);
        let error = LexicalRead::search(&index, &cx, QUERY, LIMIT)
            .await
            .expect_err("trait search: cancelled Cx must be rejected");
        assert!(
            matches!(error, SearchError::Cancelled { .. }),
            "trait search: expected typed cancellation, got {error:?}"
        );

        // LexicalRead::search_candidates with a cancelled Cx.
        let error = LexicalRead::search_candidates(&index, &cx, QUERY, LIMIT)
            .await
            .expect_err("fusion candidates: cancelled Cx must be rejected");
        assert!(
            matches!(error, SearchError::Cancelled { .. }),
            "fusion candidates: expected typed cancellation, got {error:?}"
        );
        cx.set_cancel_requested(false);

        // Hydration boundary: produce real candidates, then cancel exactly at
        // the hydration call. The candidate payload must remain unhydrated
        // (no fabricated metadata) and the retry must hydrate it.
        let (mut candidates, context) = candidate_parts(&index, &cx).await;
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.metadata.is_none()),
            "deferred candidates carry no metadata before hydration"
        );

        cx.set_cancel_requested(true);
        let error = LexicalRead::hydrate_candidates(&index, &cx, context.as_ref(), &mut candidates)
            .await
            .expect_err("hydration: cancelled Cx must be rejected");
        let SearchError::Cancelled { phase, .. } = &error else {
            panic!("hydration: expected typed cancellation, got {error:?}");
        };
        assert_eq!(
            phase, "fusion metadata hydration",
            "hydration names its own phase, not a generic search phase"
        );
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.metadata.is_none()),
            "a cancelled hydration must not partially fabricate metadata"
        );

        cx.set_cancel_requested(false);
        LexicalRead::hydrate_candidates(&index, &cx, context.as_ref(), &mut candidates)
            .await
            .expect("hydration retry succeeds");
        let annotated = serde_json::json!({ "lang": "rust", "ordinal": "b" });
        let hydrated = candidates
            .iter()
            .find(|candidate| candidate.doc_id == "doc-b")
            .expect("annotated candidate present");
        assert_eq!(
            hydrated.metadata.as_deref(),
            Some(&annotated),
            "retry hydrates the same candidates the cancelled call refused"
        );

        assert!(
            Arc::ptr_eq(&snapshot_before, &index.search_snapshot()),
            "no async boundary, cancelled or retried, may swap the snapshot"
        );
    });
}

#[test]
fn cancelled_cx_still_rejects_on_an_empty_query_class_matrix() {
    // Cancellation dominates every other request property. A zero limit, an
    // empty query and an offset past the end would each short-circuit to an
    // empty result on their own, but the entry checkpoint runs first, so all
    // three report the typed cancellation instead. Asserting the exact
    // outcome matters: accepting "either a cancellation or an empty result"
    // would pass whichever way the precedence ran, and so would pin nothing.
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = fixture_index(&cx).await;
        let snapshot_before = index.search_snapshot();

        cx.set_cancel_requested(true);
        for (label, result) in [
            (
                "zero limit",
                index.search_results(&cx, QUERY, 0).map(|hits| hits.len()),
            ),
            (
                "empty query",
                index.search_results(&cx, "", LIMIT).map(|hits| hits.len()),
            ),
            (
                "offset past the end",
                index
                    .search_paginated(&cx, QUERY, LIMIT, 9_999, true)
                    .map(|page| page.hits.len()),
            ),
        ] {
            let error = result.expect_err(&format!(
                "{label}: cancellation must take precedence over the degenerate request"
            ));
            assert!(
                matches!(&error, QuillIndexError::Cancelled { phase } if *phase == "search"),
                "{label}: expected Cancelled at the search phase, got {error:?}"
            );
        }
        cx.set_cancel_requested(false);

        assert!(
            Arc::ptr_eq(&snapshot_before, &index.search_snapshot()),
            "degenerate cancelled requests must not perturb the snapshot"
        );
    });
}

#[cfg(feature = "conformance-internals")]
#[test]
fn real_public_search_methods_cancel_during_collection_and_retry_exactly() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = fixture_index(&cx).await;
        let controller = index.conformance_cancellation_controller();
        let snapshot_before = index.search_snapshot();
        let control = LexicalRead::search(&index, &cx, QUERY, LIMIT)
            .await
            .expect("full-search control");
        let control_candidates = candidate_parts(&index, &cx).await.0;

        for candidate_boundary in [false, true] {
            controller
                .arm(ConformanceCancellationStage::QueryCollection, u64::MAX)
                .expect("arm unreachable collection cancellation");
            let discovery = cancellation_search(&index, &cx, candidate_boundary)
                .await
                .expect("discover complete collection checkpoint count");
            let checkpoint_count = controller.observed_checkpoints();
            assert!(checkpoint_count > 0, "query must admit collection work");
            assert!(
                !controller.fired(),
                "the unreachable ordinal must not request cancellation"
            );
            controller.disarm();
            let expected = if candidate_boundary {
                &control_candidates
            } else {
                &control
            };
            assert_results_identical(&discovery, expected, "collection ordinal discovery");

            for trigger_ordinal in 1..=checkpoint_count {
                let state_before = index
                    .conformance_pending_writer_state()
                    .expect("capture exact pre-invocation writer topology");
                controller
                    .arm(
                        ConformanceCancellationStage::QueryCollection,
                        trigger_ordinal,
                    )
                    .expect("arm collection cancellation");
                let cancelled = cancellation_search(&index, &cx, candidate_boundary).await;
                let SearchError::Cancelled { phase, reason } =
                    cancelled.expect_err("public query must observe injected cancellation")
                else {
                    panic!("collection checkpoint returned a non-cancellation error");
                };
                assert_eq!(phase, "search");
                assert_eq!(reason, "Quill observed request cancellation");
                assert!(controller.fired());
                assert_eq!(controller.observed_checkpoints(), trigger_ordinal);
                assert!(
                    cx.is_cancel_requested(),
                    "the deterministic checkpoint must request cancellation on the real Cx"
                );
                assert!(Arc::ptr_eq(&snapshot_before, &index.search_snapshot()));
                assert_eq!(
                    index
                        .conformance_pending_writer_state()
                        .expect("capture cancelled writer topology"),
                    state_before,
                    "every collection failure ordinal must preserve exact writer topology"
                );

                controller.disarm();
                assert!(
                    cx.is_cancel_requested(),
                    "disarming the conformance seam must not clear request authority"
                );
                cx.set_cancel_requested(false);
                let retry = cancellation_search(&index, &cx, candidate_boundary)
                    .await
                    .expect("clean retry after collection cancellation");
                assert_results_identical(&retry, expected, "collection cancellation retry");
                assert_eq!(
                    index
                        .conformance_pending_writer_state()
                        .expect("capture retried writer topology"),
                    state_before,
                    "clean retry must retain the same writer topology"
                );
            }
        }
    });
}

#[cfg(feature = "conformance-internals")]
#[test]
fn real_public_hydration_cancels_with_exact_retained_prefix_and_replays() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = fixture_index(&cx).await;
        let controller = index.conformance_cancellation_controller();
        let snapshot_before = index.search_snapshot();

        let (mut control, control_context) = candidate_parts(&index, &cx).await;
        LexicalRead::hydrate_candidates(&index, &cx, control_context.as_ref(), &mut control)
            .await
            .expect("hydration control");

        let (mut discovery, discovery_context) = candidate_parts(&index, &cx).await;
        controller
            .arm(ConformanceCancellationStage::FusionHydration, u64::MAX)
            .expect("arm unreachable hydration cancellation");
        LexicalRead::hydrate_candidates(&index, &cx, discovery_context.as_ref(), &mut discovery)
            .await
            .expect("discover complete hydration checkpoint count");
        let checkpoint_count = controller.observed_checkpoints();
        assert_eq!(
            checkpoint_count,
            u64::try_from(control.len()).expect("fixture candidate count fits u64")
        );
        assert!(
            !controller.fired(),
            "the unreachable ordinal must not request cancellation"
        );
        controller.disarm();
        assert_results_identical(&discovery, &control, "hydration ordinal discovery");

        for trigger_ordinal in 1..=checkpoint_count {
            let state_before = index
                .conformance_pending_writer_state()
                .expect("capture exact pre-hydration writer topology");
            let (mut first, first_context) = candidate_parts(&index, &cx).await;
            controller
                .arm(
                    ConformanceCancellationStage::FusionHydration,
                    trigger_ordinal,
                )
                .expect("arm hydration cancellation");
            let SearchError::Cancelled { phase, reason } =
                LexicalRead::hydrate_candidates(&index, &cx, first_context.as_ref(), &mut first)
                    .await
                    .expect_err("hydration must observe injected cancellation")
            else {
                panic!("hydration checkpoint returned a non-cancellation error");
            };
            assert_eq!(phase, "fusion metadata hydration");
            assert_eq!(reason, "Quill observed request cancellation");
            assert!(controller.fired());
            assert_eq!(controller.observed_checkpoints(), trigger_ordinal);
            assert!(
                cx.is_cancel_requested(),
                "the hydration checkpoint must request cancellation on the real Cx"
            );
            let retained_prefix =
                usize::try_from(trigger_ordinal - 1).expect("fixture ordinal fits usize");
            for (candidate_index, candidate) in first.iter().enumerate() {
                if candidate_index < retained_prefix {
                    assert_eq!(
                        candidate.metadata, control[candidate_index].metadata,
                        "hydrated prefix must match the noncancelled control"
                    );
                } else {
                    assert!(
                        candidate.metadata.is_none(),
                        "unreached hydration suffix must remain untouched"
                    );
                }
            }
            let cancelled_state = first.clone();
            assert!(Arc::ptr_eq(&snapshot_before, &index.search_snapshot()));
            assert_eq!(
                index
                    .conformance_pending_writer_state()
                    .expect("capture cancelled hydration writer topology"),
                state_before,
                "every hydration failure ordinal must preserve exact writer topology"
            );

            controller.disarm();
            cx.set_cancel_requested(false);
            LexicalRead::hydrate_candidates(&index, &cx, first_context.as_ref(), &mut first)
                .await
                .expect("partially hydrated candidates retry cleanly");
            assert_results_identical(&first, &control, "hydration cancellation retry");

            let (mut replay, replay_context) = candidate_parts(&index, &cx).await;
            controller
                .arm(
                    ConformanceCancellationStage::FusionHydration,
                    trigger_ordinal,
                )
                .expect("rearm hydration cancellation");
            LexicalRead::hydrate_candidates(&index, &cx, replay_context.as_ref(), &mut replay)
                .await
                .expect_err("hydration replay must cancel");
            assert_eq!(controller.observed_checkpoints(), trigger_ordinal);
            assert_results_identical(&replay, &cancelled_state, "hydration cancellation replay");
            assert_eq!(
                index
                    .conformance_pending_writer_state()
                    .expect("capture replayed hydration writer topology"),
                state_before,
                "replay must preserve the same exact writer topology"
            );
            controller.disarm();
            cx.set_cancel_requested(false);
        }
    });
}

#[cfg(feature = "conformance-internals")]
#[test]
fn real_public_commit_cancels_before_publication_and_retains_pending_state() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = fixture_index(&cx).await;
        let pending = IndexableDocument::new("doc-pending", "alpha pending document")
            .with_metadata("ordinal", "pending");
        LexicalWrite::index_document(&index, &cx, &pending)
            .await
            .expect("stage pending document");
        assert!(index.has_uncommitted_changes());

        let control_index = fixture_index(&cx).await;
        LexicalWrite::index_document(&control_index, &cx, &pending)
            .await
            .expect("stage control document");
        LexicalWrite::commit(&control_index, &cx)
            .await
            .expect("commit noncancelled control");
        let control = LexicalRead::search(&control_index, &cx, QUERY, LIMIT)
            .await
            .expect("noncancelled commit control");

        let controller = index.conformance_cancellation_controller();
        let snapshot_before = index.search_snapshot();
        let snapshot_epoch_before = snapshot_before.snapshot_epoch();
        let keeper_generation_before = snapshot_before.keeper_generation();
        let doc_count_before = index.doc_count();

        controller
            .arm(ConformanceCancellationStage::CommitPublication, 1)
            .expect("arm publication cancellation");
        let SearchError::Cancelled { phase, reason } = LexicalWrite::commit(&index, &cx)
            .await
            .expect_err("commit must observe injected pre-publication cancellation")
        else {
            panic!("publication checkpoint returned a non-cancellation error");
        };
        assert_eq!(phase, "commit publish");
        assert_eq!(reason, "Quill observed request cancellation");
        assert!(controller.fired());
        assert_eq!(controller.observed_checkpoints(), 1);
        assert!(
            cx.is_cancel_requested(),
            "the publication checkpoint must request cancellation on the real Cx"
        );
        assert!(Arc::ptr_eq(&snapshot_before, &index.search_snapshot()));
        assert_eq!(index.doc_count(), doc_count_before);
        assert!(index.has_uncommitted_changes());
        let pending_writer_state = index
            .conformance_pending_writer_state()
            .expect("capture exact retained commit transaction");
        assert_eq!(pending_writer_state.dirty_shard_count(), 0);
        assert_eq!(pending_writer_state.pending_identity_count(), 0);
        assert_eq!(pending_writer_state.uncommitted_id_count(), 1);
        assert_eq!(pending_writer_state.pending_segment_count(), 1);
        assert_eq!(pending_writer_state.pending_owned_segment_count(), 1);
        assert!(
            pending_writer_state.pending_manifest_present(),
            "pre-publication cancellation must retain the prepared MANIFEST proposal"
        );

        controller.disarm();
        cx.set_cancel_requested(false);
        controller
            .arm(ConformanceCancellationStage::CommitPublication, 1)
            .expect("rearm publication cancellation");
        let SearchError::Cancelled { phase, reason } = LexicalWrite::commit(&index, &cx)
            .await
            .expect_err("commit replay must observe the same publication cancellation")
        else {
            panic!("publication replay returned a non-cancellation error");
        };
        assert_eq!(phase, "commit publish");
        assert_eq!(reason, "Quill observed request cancellation");
        assert_eq!(controller.observed_checkpoints(), 1);
        let replay_pending_writer_state = index
            .conformance_pending_writer_state()
            .expect("capture replayed retained commit transaction");
        assert_eq!(
            replay_pending_writer_state, pending_writer_state,
            "cancellation replay must retain the exact pending FSLX and MANIFEST transaction"
        );
        assert!(
            Arc::ptr_eq(&snapshot_before, &index.search_snapshot()),
            "replayed pre-publication cancellation must not install the prepared successor"
        );

        controller.disarm();
        cx.set_cancel_requested(false);
        LexicalWrite::commit(&index, &cx)
            .await
            .expect("commit retry publishes pending state");
        assert!(!index.has_uncommitted_changes());
        let snapshot_after = index.search_snapshot();
        assert_eq!(
            snapshot_after.snapshot_epoch(),
            snapshot_epoch_before + 1,
            "one successful retry must install exactly one composite epoch"
        );
        assert_eq!(
            snapshot_after.keeper_generation(),
            keeper_generation_before + 1,
            "one successful retry must publish exactly one MANIFEST generation"
        );
        let retry = LexicalRead::search(&index, &cx, QUERY, LIMIT)
            .await
            .expect("post-commit retry search");
        assert_results_identical(&retry, &control, "commit cancellation retry");
    });
}

#[cfg(feature = "conformance-internals")]
#[test]
fn pending_writer_fingerprint_changes_when_canonical_pending_state_changes() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = fixture_index(&cx).await;
        LexicalWrite::index_document(
            &index,
            &cx,
            &IndexableDocument::new("pending-a", "alpha first pending document")
                .with_metadata("ordinal", "first"),
        )
        .await
        .expect("stage first pending document");
        let first = index
            .conformance_pending_writer_state()
            .expect("fingerprint first pending transaction");
        assert_eq!(first.dirty_shard_count(), 1);
        assert_eq!(first.pending_identity_count(), 1);
        assert_eq!(first.uncommitted_id_count(), 1);

        LexicalWrite::index_document(
            &index,
            &cx,
            &IndexableDocument::new("pending-b", "beta second pending document")
                .with_metadata("ordinal", "second"),
        )
        .await
        .expect("stage second pending document");
        let changed = index
            .conformance_pending_writer_state()
            .expect("fingerprint changed pending transaction");
        assert_eq!(changed.dirty_shard_count(), 1);
        assert_eq!(changed.pending_identity_count(), 2);
        assert_eq!(changed.uncommitted_id_count(), 2);
        assert_ne!(
            changed.digest_sha256(),
            first.digest_sha256(),
            "canonical pending document and allocator mutations must change the writer receipt"
        );
    });
}
