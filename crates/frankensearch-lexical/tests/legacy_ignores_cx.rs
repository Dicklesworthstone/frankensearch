//! Live divergence-register control: the Tantivy adapter ignores `Cx`
//! cancellation (bd-fjpu).
//!
//! Quill observes `Cx` cancellation in query execution and hydration as a
//! required correctness contract (`frankensearch-quill/tests/
//! cancellation_contract.rs`). The legacy Tantivy adapter does not consult
//! `Cx` on its query paths at all. This file is the *documented live
//! control* the bd-fjpu contract requires: it proves — with running code,
//! not prose — that the legacy engine's behavior is `LegacyIgnoresCx`, so
//! the Quill/Tantivy difference is recorded as a divergence, never as
//! parity, and never as a waiver of Quill's cancellation obligations.
//!
//! If these assertions ever fail because the adapter became cancel-aware,
//! that is an improvement, not a regression: update the flip divergence
//! register (bd-fjpu / the E6 divergence-register bead) and rewrite this
//! control to document the new behavior. Do not "fix" the adapter back.

use asupersync::Cx;
use frankensearch_core::IndexableDocument;
use frankensearch_core::traits::{LexicalRead, LexicalWrite};
use frankensearch_lexical::TantivyIndex;

const QUERY: &str = "alpha";
const LIMIT: usize = 10;

async fn fixture_index(cx: &Cx) -> TantivyIndex {
    let index = TantivyIndex::in_memory().expect("in-memory tantivy index");
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

#[test]
fn tantivy_query_boundaries_ignore_a_cancelled_cx() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let index = fixture_index(&cx).await;

        // Noncancelled control.
        let control = LexicalRead::search(&index, &cx, QUERY, LIMIT)
            .await
            .expect("control search");
        assert_eq!(control.len(), 3, "control must match all three documents");
        let control_ids: Vec<_> = index
            .search_doc_ids(&cx, QUERY, LIMIT)
            .expect("control doc-id search")
            .into_iter()
            .map(|hit| hit.doc_id)
            .collect();

        // The same boundaries under a cancelled Cx: the legacy adapter
        // completes the work and returns identical results. This is the
        // recorded LegacyIgnoresCx divergence, not an endorsement.
        cx.set_cancel_requested(true);

        let cancelled_search = LexicalRead::search(&index, &cx, QUERY, LIMIT)
            .await
            .expect("legacy adapter ignores cancellation on trait search");
        assert_eq!(cancelled_search.len(), control.len());
        for (cancelled, control) in cancelled_search.iter().zip(control.iter()) {
            assert_eq!(cancelled.doc_id, control.doc_id);
            assert_eq!(cancelled.score.to_bits(), control.score.to_bits());
        }

        let batch = LexicalRead::search_candidates(&index, &cx, QUERY, LIMIT)
            .await
            .expect("legacy adapter ignores cancellation on fusion candidates");
        let (mut candidates, pin) = batch.into_parts();
        assert_eq!(candidates.len(), control.len());

        LexicalRead::hydrate_candidates(&index, &cx, pin.as_ref(), &mut candidates)
            .await
            .expect("legacy adapter ignores cancellation on hydration");
        let annotated = serde_json::json!({ "lang": "rust" });
        let hydrated = candidates
            .iter()
            .find(|candidate| candidate.doc_id == "doc-b")
            .expect("annotated candidate present");
        assert_eq!(
            hydrated.metadata.as_deref(),
            Some(&annotated),
            "hydration completes fully under a cancelled Cx"
        );

        let cancelled_ids: Vec<_> = index
            .search_doc_ids(&cx, QUERY, LIMIT)
            .expect("legacy adapter ignores cancellation on doc-id search")
            .into_iter()
            .map(|hit| hit.doc_id)
            .collect();
        assert_eq!(cancelled_ids, control_ids);

        cx.set_cancel_requested(false);
    });
}
