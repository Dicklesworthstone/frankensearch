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
    FIXTURE_CORPUS, FIXTURE_ENRICHMENT_EXPECTATIONS, FIXTURE_EXPECTATIONS, FIXTURE_METADATA,
    adjudicate, adjudicate_enrichment, observe_quill, observe_quill_enrichment,
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

/// The same committed expectations must hold against the REAL Tantivy
/// incumbent. Running both arms against ONE independent oracle is what makes
/// a common-mode defect visible: neither engine is the other's reference.
#[cfg(feature = "tantivy-oracle")]
#[test]
fn the_committed_expectations_hold_against_real_tantivy() {
    use frankensearch_core::traits::LexicalWrite;
    use frankensearch_lexical::TantivyIndex;
    use frankensearch_quill_gauntlet::native_enriched_witness::observe_tantivy;

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().expect("witness tempdir");
        let index = TantivyIndex::create(dir.path()).expect("create the witness Tantivy index");
        for (doc_id, body) in FIXTURE_CORPUS {
            index
                .index_document(&cx, &IndexableDocument::new(*doc_id, *body))
                .await
                .expect("index a witness fixture document");
        }
        index.commit(&cx).await.expect("commit the Tantivy index");

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
