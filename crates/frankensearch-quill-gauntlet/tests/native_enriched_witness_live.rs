//! bd-8nqz.4.1: run the native enriched witness against the REAL engines.
//!
//! The unit tests beside the oracle exercise the adjudicator on synthetic
//! observations, which proves the oracle is strict but proves nothing about
//! whether its committed expectations are TRUE of the shipping engines. That
//! is the gap this suite closes: it builds the committed corpus in a real
//! `QuillIndex` (and, under `tantivy-oracle`, a real `TantivyIndex`), drives
//! each engine's own native paginated API, and adjudicates the result against
//! the hand-derived expectations.
//!
//! An expectation that is wrong about BM25 fails HERE, loudly, instead of
//! sitting in a green unit suite describing an engine nobody ran.

use frankensearch_core::IndexableDocument;
use frankensearch_quill::{QuillConfig, QuillIndex};
use frankensearch_quill_gauntlet::native_enriched_witness::{
    EnrichedExpectationV1, FIXTURE_CORPUS, FIXTURE_ENRICHMENT_EXPECTATIONS, FIXTURE_EXPECTATIONS,
    FIXTURE_METADATA, UTF8_DOC_ID, UTF8_INTACT_TOKEN, adjudicate, adjudicate_enrichment,
    adjudicate_truncation_determinism, adjudicate_utf8_window, observe_quill,
    observe_quill_enrichment, truncation_probe_queries,
};

/// Build the committed corpus in a real Quill index.
async fn build_quill(cx: &asupersync::Cx, dir: &std::path::Path) -> QuillIndex {
    let index = QuillIndex::create(
        cx,
        dir,
        QuillConfig {
            bulk_load_mode: true,
            deterministic_ingest: true,
            max_ingest_shards: 1,
            ..QuillConfig::default()
        },
    )
    .await
    .expect("create the witness Quill index");
    for document in fixture_documents() {
        index
            .index_document(cx, &document)
            .await
            .expect("index a witness fixture document");
    }
    index
        .finish_bulk_load(cx)
        .await
        .expect("finalize the witness Quill index");
    index
}

/// The committed corpus as indexable documents, with the metadata the
/// enrichment expectations describe.
///
/// Both engines are fed EXACTLY this list, so a metadata or snippet
/// divergence is a divergence in the engines, not in how they were loaded.
fn fixture_documents() -> Vec<IndexableDocument> {
    FIXTURE_CORPUS
        .iter()
        .map(|(doc_id, body)| {
            let mut document = IndexableDocument::new(*doc_id, *body);
            if let Some((_, pairs)) = FIXTURE_METADATA.iter().find(|(id, _)| id == doc_id) {
                for (key, value) in *pairs {
                    document
                        .metadata
                        .insert((*key).to_owned(), (*value).to_owned());
                }
            }
            document
        })
        .collect()
}

/// Every committed expectation must hold against the REAL native Quill
/// paginated API — exact count, offset pagination, ordering and all.
#[test]
fn the_committed_expectations_hold_against_real_quill() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().expect("witness tempdir");
        let index = build_quill(&cx, dir.path()).await;

        let mut failures = Vec::new();
        for expectation in FIXTURE_EXPECTATIONS {
            let observed =
                observe_quill(&cx, &index, expectation).expect("observe native Quill page");
            let verdict = adjudicate(expectation, &observed);
            if !verdict.passed() {
                failures.push(format!(
                    "query={:?} limit={} offset={} -> {:?} (observed page {:?}, total {})",
                    expectation.query,
                    expectation.limit,
                    expectation.offset,
                    verdict.oracle_failures,
                    observed.page_doc_ids,
                    observed.total,
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "the committed expectations do not describe the shipping Quill engine:\n{}",
            failures.join("\n")
        );
    });
}

/// Every committed ENRICHMENT expectation must hold against real Quill:
/// configured highlight tags, hand-derived query classification, and metadata
/// semantics.
#[test]
fn the_committed_enrichment_expectations_hold_against_real_quill() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().expect("witness tempdir");
        let index = build_quill(&cx, dir.path()).await;

        let mut failures = Vec::new();
        for expectation in FIXTURE_ENRICHMENT_EXPECTATIONS {
            let observed = observe_quill_enrichment(&cx, &index, expectation)
                .expect("observe native Quill enrichment");
            let verdict = adjudicate_enrichment(expectation, &observed);
            if !verdict.passed() {
                failures.push(format!(
                    "query={:?} tags={}{} -> {:?} (hits {:?})",
                    expectation.query,
                    expectation.highlight_prefix,
                    expectation.highlight_postfix,
                    verdict.oracle_failures,
                    observed.hits,
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "the committed enrichment expectations do not describe the shipping Quill engine:\n{}",
            failures.join("\n")
        );
    });
}

/// bd-8nqz.4.1: the UTF-8 window boundary and deterministic long-query
/// truncation dimensions, against real Quill.
#[test]
fn utf8_windows_and_long_query_truncation_hold_against_real_quill() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().expect("witness tempdir");
        let index = build_quill(&cx, dir.path()).await;

        // UTF-8: the snippet window must consist of whole scalar values.
        let utf8_row = FIXTURE_ENRICHMENT_EXPECTATIONS
            .iter()
            .find(|row| row.subject_doc == UTF8_DOC_ID)
            .expect("utf8 enrichment row");
        let observed = observe_quill_enrichment(&cx, &index, utf8_row).expect("observe utf8");
        let verdict = adjudicate_utf8_window(&observed, UTF8_DOC_ID, UTF8_INTACT_TOKEN);
        assert!(
            verdict.passed(),
            "UTF-8 window boundary violated: {:?} (hits {:?})",
            verdict.oracle_failures,
            observed.hits
        );

        // Truncation: an over-length query must behave exactly like its
        // first-MAX_QUERY_LENGTH-character prefix.
        let (long, prefix, excluded_doc) = truncation_probe_queries();
        let row = |query: &str| EnrichedExpectationV1 {
            query: Box::leak(query.to_owned().into_boxed_str()),
            limit: 10,
            offset: 0,
            matching_docs: &[],
            total: 0,
            unambiguous_top: None,
        };
        let long_row = row(&long);
        let prefix_row = row(&prefix);
        let long_observed = observe_quill(&cx, &index, &long_row).expect("observe long query");
        let prefix_observed = observe_quill(&cx, &index, &prefix_row).expect("observe prefix");
        let verdict = adjudicate_truncation_determinism(&long_observed, &prefix_observed);
        assert!(
            verdict.passed(),
            "long-query truncation is not deterministic: {:?}",
            verdict.oracle_failures
        );
        // Both directions of the boundary, so this is a CUT-POINT test and
        // not merely a determinism test.
        assert!(
            prefix_observed
                .page_doc_ids
                .iter()
                .any(|id| id == "doc-beta"),
            "the term just INSIDE the cap must survive truncation; got {:?}",
            prefix_observed.page_doc_ids
        );
        assert!(
            !long_observed
                .page_doc_ids
                .iter()
                .any(|id| id == excluded_doc),
            "the term just BEYOND the cap must be truncated away, but {excluded_doc} matched: {:?}",
            long_observed.page_doc_ids
        );
    });
}

/// The same committed expectations must hold against the REAL Tantivy
/// incumbent. Running both arms against ONE independent oracle is what makes
/// a common-mode defect visible: neither engine is the other's reference.
#[cfg(feature = "tantivy-oracle")]
#[test]
fn the_committed_expectations_hold_against_real_tantivy() {
    use frankensearch_core::traits::LexicalWrite;
    use frankensearch_lexical::TantivyIndex;
    use frankensearch_quill_gauntlet::native_enriched_witness::{
        observe_tantivy, observe_tantivy_enrichment,
    };

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().expect("witness tempdir");
        let index = TantivyIndex::create(dir.path()).expect("create the witness Tantivy index");
        for document in fixture_documents() {
            index
                .index_document(&cx, &document)
                .await
                .expect("index a witness fixture document");
        }
        index.commit(&cx).await.expect("commit the Tantivy index");

        // The ENRICHED arm runs against the same index and the same committed
        // expectations, so a divergence between the engines shows up as one
        // of them failing the SHARED oracle rather than as a diff.
        let mut enrichment_failures = Vec::new();
        for expectation in FIXTURE_ENRICHMENT_EXPECTATIONS {
            let observed = observe_tantivy_enrichment(&cx, &index, expectation)
                .expect("observe native Tantivy enrichment");
            let verdict = adjudicate_enrichment(expectation, &observed);
            if !verdict.passed() {
                enrichment_failures.push(format!(
                    "query={:?} tags={}{} -> {:?} (hits {:?})",
                    expectation.query,
                    expectation.highlight_prefix,
                    expectation.highlight_postfix,
                    verdict.oracle_failures,
                    observed.hits,
                ));
            }
        }
        assert!(
            enrichment_failures.is_empty(),
            "the committed enrichment expectations do not describe the shipping Tantivy \
             engine:\n{}",
            enrichment_failures.join("\n")
        );

        // UTF-8 window boundaries, same oracle as the Quill arm.
        let utf8_row = FIXTURE_ENRICHMENT_EXPECTATIONS
            .iter()
            .find(|row| row.subject_doc == UTF8_DOC_ID)
            .expect("utf8 enrichment row");
        let utf8_observed =
            observe_tantivy_enrichment(&cx, &index, utf8_row).expect("observe utf8");
        let utf8_verdict = adjudicate_utf8_window(&utf8_observed, UTF8_DOC_ID, UTF8_INTACT_TOKEN);
        assert!(
            utf8_verdict.passed(),
            "Tantivy UTF-8 window boundary violated: {:?} (hits {:?})",
            utf8_verdict.oracle_failures,
            utf8_observed.hits
        );

        // Deterministic long-query truncation, with the cut point observable
        // in both directions.
        let (long, prefix, excluded_doc) = truncation_probe_queries();
        let row = |query: &str| EnrichedExpectationV1 {
            query: Box::leak(query.to_owned().into_boxed_str()),
            limit: 10,
            offset: 0,
            matching_docs: &[],
            total: 0,
            unambiguous_top: None,
        };
        let long_observed = observe_tantivy(&cx, &index, &row(&long)).expect("observe long query");
        let prefix_observed = observe_tantivy(&cx, &index, &row(&prefix)).expect("observe prefix");
        let truncation_verdict =
            adjudicate_truncation_determinism(&long_observed, &prefix_observed);
        assert!(
            truncation_verdict.passed(),
            "Tantivy long-query truncation is not deterministic: {:?}",
            truncation_verdict.oracle_failures
        );
        assert!(
            prefix_observed
                .page_doc_ids
                .iter()
                .any(|id| id == "doc-beta"),
            "the term just INSIDE the cap must survive truncation; got {:?}",
            prefix_observed.page_doc_ids
        );
        assert!(
            !long_observed
                .page_doc_ids
                .iter()
                .any(|id| id == excluded_doc),
            "the term just BEYOND the cap must be truncated away, but {excluded_doc} matched: {:?}",
            long_observed.page_doc_ids
        );

        let mut failures = Vec::new();
        for expectation in FIXTURE_EXPECTATIONS {
            let observed =
                observe_tantivy(&cx, &index, expectation).expect("observe native Tantivy page");
            let verdict = adjudicate(expectation, &observed);
            if !verdict.passed() {
                failures.push(format!(
                    "query={:?} limit={} offset={} -> {:?} (observed page {:?}, total {})",
                    expectation.query,
                    expectation.limit,
                    expectation.offset,
                    verdict.oracle_failures,
                    observed.page_doc_ids,
                    observed.total,
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "the committed expectations do not describe the shipping Tantivy engine:\n{}",
            failures.join("\n")
        );
    });
}
