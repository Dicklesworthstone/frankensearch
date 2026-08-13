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
#[cfg(feature = "pruning-conformance")]
use frankensearch_quill::index::ConformancePruningExecutionMode;
use frankensearch_quill::index::{QuillIndex, QuillIndexError};
use frankensearch_quill::snippet::SnippetConfig;
#[cfg(feature = "profile-internals")]
use frankensearch_quill::{
    QuillProfileCacheDisposition, QuillProfileExecutionMode, QuillProfileOutcome,
    QuillProfiledSearchOutcome, QuillSearchIndex,
};

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

#[cfg(feature = "profile-internals")]
#[test]
fn profiled_search_sidecar_executes_through_public_durable_reader() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("profile sidecar directory");
        let writer = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create durable profile fixture");
        LexicalWrite::index_document(
            &writer,
            &cx,
            &IndexableDocument::new("profiled-doc", "alpha profile sidecar document"),
        )
        .await
        .expect("ingest durable profile fixture");
        LexicalWrite::commit(&writer, &cx)
            .await
            .expect("publish durable profile fixture");

        let reader = QuillSearchIndex::open(&cx, directory.path(), deterministic_config())
            .await
            .expect("open public durable reader");
        let first = reader
            .search_paginated_with_profile(&cx, QUERY, LIMIT, 0, false)
            .expect("first profiled public search");
        let first_receipt = match first {
            QuillProfiledSearchOutcome::Completed { result, receipt } => {
                assert_eq!(
                    result.hits.len(),
                    1,
                    "ordinary search must return the fixture hit"
                );
                receipt
            }
            QuillProfiledSearchOutcome::Failed { error, .. } => {
                panic!("first profiled public search unexpectedly failed: {error}")
            }
        };
        assert_eq!(first_receipt.cache(), QuillProfileCacheDisposition::Miss);
        assert_eq!(
            first_receipt.execution(),
            Some(QuillProfileExecutionMode::Serial),
            "one sealed segment must use the shipping serial branch"
        );
        assert_eq!(
            first_receipt.counters().0,
            2,
            "the two default text fields each require a snapshot DF probe"
        );
        assert_eq!(
            first_receipt.counters().1,
            2,
            "the two default text fields each require a global DF probe"
        );
        assert_eq!(
            first_receipt.counters().2,
            4,
            "each default field reads the dictionary for DF and cursor lowering"
        );
        assert_eq!(first_receipt.counters().3, 1, "sealed lowering count");
        assert_eq!(first_receipt.outcome(), QuillProfileOutcome::Completed);

        let repeated = reader
            .search_paginated_with_profile(&cx, QUERY, LIMIT, 0, false)
            .expect("repeat profiled public search");
        let repeated_receipt = match repeated {
            QuillProfiledSearchOutcome::Completed { result, receipt } => {
                assert_eq!(
                    result.hits.len(),
                    1,
                    "cache hit must preserve ordinary result"
                );
                receipt
            }
            QuillProfiledSearchOutcome::Failed { error, .. } => {
                panic!("repeat profiled public search unexpectedly failed: {error}")
            }
        };
        assert_eq!(repeated_receipt.cache(), QuillProfileCacheDisposition::Hit);
        assert_eq!(repeated_receipt.execution(), None);
        assert_eq!(repeated_receipt.counters(), (0, 0, 0, 0, 0));
        assert_eq!(repeated_receipt.outcome(), QuillProfileOutcome::Completed);
    });
}

#[cfg(feature = "profile-internals")]
#[test]
fn profiled_search_public_reader_preserves_fuel_exhaustion_receipt() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("fuel profile sidecar directory");
        let config = QuillConfig {
            deterministic_ingest: true,
            query_fuel_budget: 1,
            ..QuillConfig::default()
        };
        let writer = QuillIndex::create(&cx, directory.path(), config.clone())
            .await
            .expect("create durable fuel fixture");
        LexicalWrite::index_document(
            &writer,
            &cx,
            &IndexableDocument::new("fuel-profiled-doc", "alpha fuel sidecar document"),
        )
        .await
        .expect("ingest durable fuel fixture");
        LexicalWrite::commit(&writer, &cx)
            .await
            .expect("publish durable fuel fixture");

        let reader = QuillSearchIndex::open(&cx, directory.path(), config)
            .await
            .expect("open public durable fuel reader");
        let outcome = reader
            .search_paginated_with_profile(&cx, QUERY, LIMIT, 0, false)
            .expect("profile admission must preserve ordinary fuel exhaustion");
        let (error, receipt) = match outcome {
            QuillProfiledSearchOutcome::Completed { .. } => {
                panic!("fuel-limited profiled search unexpectedly completed")
            }
            QuillProfiledSearchOutcome::Failed { error, receipt } => (error, receipt),
        };
        assert!(matches!(
            error,
            QuillIndexError::QueryFuelExhausted {
                budget: 1,
                consumed: 1,
                ..
            }
        ));
        assert_eq!(receipt.cache(), QuillProfileCacheDisposition::Miss);
        assert_eq!(receipt.fanout_eligible(), Some(false));
        assert_eq!(
            receipt.execution(),
            Some(QuillProfileExecutionMode::Serial),
            "one sealed segment reaches the serial branch before fuel refusal"
        );
        let Some((work_upper_bound, metering)) = receipt.work_plan() else {
            panic!("fuel-limited profiled search did not bind a work plan");
        };
        assert!(work_upper_bound > 1);
        assert!(metering);
        assert_eq!(receipt.work_units().requested(), [1, 1, 0, 0]);
        assert_eq!(receipt.work_units().admitted(), [1, 0, 0, 0]);
        assert_eq!(receipt.work_units().refused(), [0, 1, 0, 0]);
        assert_eq!(receipt.counters(), (1, 1, 1, 1, 1));
        assert_eq!(receipt.cancellation_observations(), 0);
        assert_eq!(receipt.outcome(), QuillProfileOutcome::FuelExhausted);
    });
}

#[cfg(feature = "profile-internals")]
#[test]
fn profiled_search_public_reader_preserves_precheck_cancellation_receipt() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("cancelled profile sidecar directory");
        let _writer = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create durable cancelled fixture");
        let reader = QuillSearchIndex::open(&cx, directory.path(), deterministic_config())
            .await
            .expect("open public durable cancelled reader");

        let cancelled = cx.clone();
        cancelled.set_cancel_requested(true);
        let outcome = reader
            .search_paginated_with_profile(&cancelled, QUERY, LIMIT, 0, false)
            .expect("profile admission must preserve ordinary cancellation");
        let (error, receipt) = match outcome {
            QuillProfiledSearchOutcome::Completed { .. } => {
                panic!("pre-cancelled profiled search unexpectedly completed")
            }
            QuillProfiledSearchOutcome::Failed { error, receipt } => (error, receipt),
        };
        assert!(matches!(error, QuillIndexError::Cancelled { phase } if phase == "search"));
        assert_eq!(receipt.cache(), QuillProfileCacheDisposition::NotChecked);
        assert_eq!(receipt.fanout_eligible(), None);
        assert_eq!(receipt.execution(), None);
        assert_eq!(receipt.work_plan(), None);
        assert_eq!(receipt.counters(), (0, 0, 0, 0, 0));
        assert_eq!(receipt.work_units().requested(), [0, 0, 0, 0]);
        assert_eq!(receipt.work_units().admitted(), [0, 0, 0, 0]);
        assert_eq!(receipt.work_units().refused(), [0, 0, 0, 0]);
        assert_eq!(receipt.cancellation_observations(), 1);
        assert_eq!(receipt.outcome(), QuillProfileOutcome::Cancelled);
    });
}

#[cfg(all(feature = "profile-internals", feature = "conformance-internals"))]
#[test]
fn profiled_search_public_reader_records_disabled_cache_without_skipping_work() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("disabled-cache profile sidecar directory");
        let writer = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create durable disabled-cache fixture");
        LexicalWrite::index_document(
            &writer,
            &cx,
            &IndexableDocument::new(
                "disabled-cache-profiled-doc",
                "alpha cache sidecar document",
            ),
        )
        .await
        .expect("ingest durable disabled-cache fixture");
        LexicalWrite::commit(&writer, &cx)
            .await
            .expect("publish durable disabled-cache fixture");

        let reader = QuillSearchIndex::open(&cx, directory.path(), deterministic_config())
            .await
            .expect("open public durable disabled-cache reader");
        let controller = reader.conformance_cancellation_controller();
        controller
            .arm(ConformanceCancellationStage::CommitPublication, 1)
            .expect("arm unrelated checkpoint to disable ranked cache");
        let outcome = reader
            .search_paginated_with_profile(&cx, QUERY, LIMIT, 0, false)
            .expect("disabled cache must not prevent ordinary profile search");
        controller.disarm();
        let (result, receipt) = match outcome {
            QuillProfiledSearchOutcome::Completed { result, receipt } => (result, receipt),
            QuillProfiledSearchOutcome::Failed { error, .. } => {
                panic!("disabled-cache profiled search unexpectedly failed: {error}")
            }
        };
        assert_eq!(result.hits.len(), 1);
        assert_eq!(receipt.cache(), QuillProfileCacheDisposition::Disabled);
        assert_eq!(receipt.fanout_eligible(), Some(false));
        assert_eq!(receipt.execution(), Some(QuillProfileExecutionMode::Serial));
        assert!(receipt.work_plan().is_some());
        assert_eq!(receipt.counters().0, 2);
        assert_eq!(receipt.counters().1, 2);
        assert_eq!(receipt.counters().2, 4);
        assert_eq!(receipt.counters().3, 1);
        assert!(receipt.counters().4 > 0);
        assert_eq!(receipt.outcome(), QuillProfileOutcome::Completed);
    });
}

#[cfg(all(feature = "profile-internals", feature = "conformance-internals"))]
#[test]
fn profiled_search_public_reader_records_checkpoint_cancellation_prefix() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("checkpoint-cancel profile sidecar directory");
        let writer = QuillIndex::create(&cx, directory.path(), deterministic_config())
            .await
            .expect("create durable checkpoint-cancel fixture");
        LexicalWrite::index_document(
            &writer,
            &cx,
            &IndexableDocument::new(
                "checkpoint-cancel-profiled-doc",
                "alpha checkpoint sidecar document",
            ),
        )
        .await
        .expect("ingest durable checkpoint-cancel fixture");
        LexicalWrite::commit(&writer, &cx)
            .await
            .expect("publish durable checkpoint-cancel fixture");

        let reader = QuillSearchIndex::open(&cx, directory.path(), deterministic_config())
            .await
            .expect("open public durable checkpoint-cancel reader");
        let controller = reader.conformance_cancellation_controller();
        controller
            .arm(ConformanceCancellationStage::QueryCollection, 2)
            .expect("arm cancellation at the second ordinary checkpoint");
        let outcome = reader
            .search_paginated_with_profile(&cx, QUERY, LIMIT, 0, false)
            .expect("checkpoint cancellation must retain the profile receipt");
        controller.disarm();
        let (error, receipt) = match outcome {
            QuillProfiledSearchOutcome::Completed { .. } => {
                panic!("checkpoint-cancelled profiled search unexpectedly completed")
            }
            QuillProfiledSearchOutcome::Failed { error, receipt } => (error, receipt),
        };
        assert!(matches!(error, QuillIndexError::Cancelled { phase } if phase == "search"));
        assert_eq!(receipt.cache(), QuillProfileCacheDisposition::Disabled);
        assert_eq!(receipt.fanout_eligible(), Some(false));
        assert_eq!(receipt.execution(), Some(QuillProfileExecutionMode::Serial));
        assert!(receipt.work_plan().is_some());
        assert_eq!(receipt.work_units().requested(), [1, 1, 0, 0]);
        assert_eq!(receipt.work_units().admitted(), [1, 0, 0, 0]);
        assert_eq!(receipt.work_units().refused(), [0, 0, 0, 0]);
        assert_eq!(receipt.counters(), (1, 1, 1, 1, 1));
        assert_eq!(receipt.cancellation_observations(), 1);
        assert_eq!(receipt.outcome(), QuillProfileOutcome::Cancelled);
    });
}

#[cfg(feature = "pruning-conformance")]
async fn two_segment_pruning_trace_fixture(cx: &Cx) -> QuillIndex {
    let index = QuillIndex::in_memory(deterministic_config()).expect("in-memory index");
    for (document_id, text) in [
        ("trace-segment-0", "alpha first sealed segment"),
        ("trace-segment-1", "alpha second sealed segment"),
    ] {
        LexicalWrite::index_document(&index, cx, &IndexableDocument::new(document_id, text))
            .await
            .expect("ingest one pruning-trace segment");
        LexicalWrite::commit(&index, cx)
            .await
            .expect("seal one pruning-trace segment");
    }
    assert_eq!(
        index
            .snapshot()
            .expect("fixture snapshot is authoritative")
            .segments()
            .len(),
        2,
        "fixture commit boundaries must remain observable"
    );
    assert_eq!(
        index
            .snapshot()
            .expect("fixture snapshot is authoritative")
            .segments()
            .iter()
            .map(|segment| segment.doc_count())
            .collect::<Vec<_>>(),
        vec![1, 1],
        "each committed boundary must contain one real scored document"
    );
    assert_eq!(
        index
            .snapshot()
            .expect("fixture snapshot is authoritative")
            .doc_count(),
        2,
        "Keeper must expose both committed documents"
    );
    assert_eq!(
        index.doc_count().expect("fixture count is authoritative"),
        2,
        "the public composite view must expose the same two live documents"
    );
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
    let snapshot_before = index
        .search_snapshot()
        .expect("published snapshot is authoritative");

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
        Arc::ptr_eq(
            &snapshot_before,
            &index
                .search_snapshot()
                .expect("published snapshot is authoritative"),
        ),
        "{boundary}: a cancelled query must not perturb the published snapshot"
    );

    cx.set_cancel_requested(false);
    let value = call(index, cx)
        .unwrap_or_else(|error| panic!("{boundary}: fresh retry must succeed, got {error:?}"));
    assert!(
        Arc::ptr_eq(
            &snapshot_before,
            &index
                .search_snapshot()
                .expect("published snapshot is authoritative"),
        ),
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
        let snapshot_before = index
            .search_snapshot()
            .expect("published snapshot is authoritative");

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
            Arc::ptr_eq(
                &snapshot_before,
                &index
                    .search_snapshot()
                    .expect("published snapshot is authoritative"),
            ),
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
        let snapshot_before = index
            .search_snapshot()
            .expect("published snapshot is authoritative");

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
                "empty snippet query",
                index
                    .search_with_snippets(&cx, " \t ", LIMIT, &SnippetConfig::default())
                    .map(|hits| hits.len()),
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
            Arc::ptr_eq(
                &snapshot_before,
                &index
                    .search_snapshot()
                    .expect("published snapshot is authoritative"),
            ),
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
        let snapshot_before = index
            .search_snapshot()
            .expect("published snapshot is authoritative");
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
                assert!(Arc::ptr_eq(
                    &snapshot_before,
                    &index
                        .search_snapshot()
                        .expect("published snapshot is authoritative"),
                ));
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

#[cfg(all(feature = "conformance-internals", feature = "pruning-conformance"))]
#[test]
fn pruning_conformance_traced_public_search_discards_partial_receipt_and_replays_exactly() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = two_segment_pruning_trace_fixture(&cx).await;
        let snapshot_before = index
            .search_snapshot()
            .expect("published snapshot is authoritative");
        let (control_page, control_receipt) = index
            .search_paginated_with_conformance_pruning_trace(&cx, QUERY, LIMIT, 0, false)
            .expect("noncancelled traced control");
        assert_eq!(
            control_receipt.execution_mode(),
            ConformancePruningExecutionMode::Serial
        );
        assert_eq!(
            control_receipt
                .segments()
                .iter()
                .map(|segment| segment.segment_ordinal())
                .collect::<Vec<_>>(),
            vec![0, 1],
            "control receipt must be complete and dense"
        );

        let controller = index.conformance_cancellation_controller();
        controller
            .arm(ConformanceCancellationStage::PruningTraceSegmentRecorded, 1)
            .expect("arm cancellation after the first recorded segment");
        let error = index
            .search_paginated_with_conformance_pruning_trace(&cx, QUERY, LIMIT, 0, false)
            .expect_err("a traced public search must not publish a partial receipt");
        assert!(
            matches!(error, QuillIndexError::Cancelled { phase: "search" }),
            "expected typed search cancellation, got {error:?}"
        );
        assert!(controller.fired());
        assert_eq!(
            controller.observed_checkpoints(),
            1,
            "the request must fail immediately after its first recorded receipt"
        );
        assert_eq!(
            controller.recorded_pruning_receipts_at_fire(),
            1,
            "the checkpoint must fire only after one receipt mutation succeeded"
        );
        assert_eq!(
            controller.discarded_pruning_trace_sessions(),
            1,
            "the cancelled invocation must fail and discard its partial trace session"
        );
        assert!(
            cx.is_cancel_requested(),
            "the typed segment boundary must cancel the real request Cx"
        );
        assert!(
            Arc::ptr_eq(
                &snapshot_before,
                &index
                    .search_snapshot()
                    .expect("published snapshot is authoritative"),
            ),
            "failed tracing must not perturb the published snapshot"
        );

        controller.disarm();
        cx.set_cancel_requested(false);
        let (retry_page, retry_receipt) = index
            .search_paginated_with_conformance_pruning_trace(&cx, QUERY, LIMIT, 0, false)
            .expect("clean traced replay");
        assert_eq!(retry_page, control_page);
        assert_eq!(retry_receipt, control_receipt);
    });
}

#[cfg(feature = "conformance-internals")]
#[test]
fn real_public_hydration_cancels_with_exact_retained_prefix_and_replays() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = fixture_index(&cx).await;
        let controller = index.conformance_cancellation_controller();
        let snapshot_before = index
            .search_snapshot()
            .expect("published snapshot is authoritative");

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
            assert!(Arc::ptr_eq(
                &snapshot_before,
                &index
                    .search_snapshot()
                    .expect("published snapshot is authoritative"),
            ));
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
        let snapshot_before = index
            .search_snapshot()
            .expect("published snapshot is authoritative");
        let snapshot_epoch_before = snapshot_before.snapshot_epoch();
        let keeper_generation_before = snapshot_before.keeper_generation();
        let doc_count_before = index.doc_count().expect("published count is authoritative");

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
        assert!(Arc::ptr_eq(
            &snapshot_before,
            &index
                .search_snapshot()
                .expect("published snapshot is authoritative"),
        ));
        assert_eq!(
            index.doc_count().expect("published count is authoritative"),
            doc_count_before
        );
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
            Arc::ptr_eq(
                &snapshot_before,
                &index
                    .search_snapshot()
                    .expect("published snapshot is authoritative"),
            ),
            "replayed pre-publication cancellation must not install the prepared successor"
        );

        controller.disarm();
        cx.set_cancel_requested(false);
        LexicalWrite::commit(&index, &cx)
            .await
            .expect("commit retry publishes pending state");
        assert!(!index.has_uncommitted_changes());
        let snapshot_after = index
            .search_snapshot()
            .expect("published snapshot is authoritative");
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
