//! Method-bound Quill cancellation contract, entry-phase matrix (bd-fjpu).
//!
//! Every public Quill search boundary invoked with a cancelled `Cx` must
//! return its typed `Cancelled { phase }` error, leave the published
//! snapshot untouched (`Arc` identity), and succeed identically on a fresh
//! retry once the cancellation clears — proving cancellation is a real
//! engine contract, not a spy-only replay. A noncancelled control on the
//! same corpus anchors the expected results.
//!
//! Scope: this file covers the *entry* checkpoint of each boundary. The
//! during-collection and during-hydration checkpoints are deterministic
//! fuel-meter seams (`QueryCheckpoint::admit`) and are exercised at the
//! unit level where the meter is constructible; see bd-fjpu for the
//! decomposition.

use std::sync::Arc;

use asupersync::Cx;
use frankensearch_core::{IndexableDocument, LexicalSearch, SearchError};
use frankensearch_quill::QuillConfig;
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
    index
        .index_documents(
            cx,
            &[
                IndexableDocument::new("doc-a", "alpha first document"),
                IndexableDocument::new("doc-b", "alpha second document")
                    .with_metadata("lang", "rust"),
                IndexableDocument::new("doc-c", "alpha third document"),
            ],
        )
        .await
        .expect("ingest fixture corpus");
    index.commit(cx).await.expect("commit fixture corpus");
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
        let control = LexicalSearch::search(&index, &cx, QUERY, LIMIT)
            .await
            .expect("control trait search");
        assert_eq!(control.len(), 3);

        // LexicalSearch::search with a cancelled Cx.
        cx.set_cancel_requested(true);
        let error = LexicalSearch::search(&index, &cx, QUERY, LIMIT)
            .await
            .expect_err("trait search: cancelled Cx must be rejected");
        assert!(
            matches!(error, SearchError::Cancelled { .. }),
            "trait search: expected typed cancellation, got {error:?}"
        );

        // LexicalSearch::search_fusion_candidates with a cancelled Cx.
        let error = LexicalSearch::search_fusion_candidates(&index, &cx, QUERY, LIMIT)
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
        let mut candidates = LexicalSearch::search_fusion_candidates(&index, &cx, QUERY, LIMIT)
            .await
            .expect("fusion candidates for hydration");
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.metadata.is_none()),
            "deferred candidates carry no metadata before hydration"
        );

        cx.set_cancel_requested(true);
        let error = LexicalSearch::hydrate_fusion_metadata(&index, &cx, &mut candidates)
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
        LexicalSearch::hydrate_fusion_metadata(&index, &cx, &mut candidates)
            .await
            .expect("hydration retry succeeds");
        let annotated = serde_json::json!({ "lang": "rust" });
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
    // Cancellation must dominate other request properties: an empty query,
    // a zero limit, and an offset past the end all still observe the
    // cancelled Cx before doing work (or return their benign empty results
    // without touching the snapshot — either way, no panic, no mutation).
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
        ] {
            match result {
                Err(QuillIndexError::Cancelled { .. }) => {}
                Ok(0) => {}
                other => panic!(
                    "{label}: expected typed cancellation or benign empty result, got {other:?}"
                ),
            }
        }
        cx.set_cancel_requested(false);

        assert!(
            Arc::ptr_eq(&snapshot_before, &index.search_snapshot()),
            "degenerate cancelled requests must not perturb the snapshot"
        );
    });
}
