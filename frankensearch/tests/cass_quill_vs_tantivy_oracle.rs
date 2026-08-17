//! Differential: the native Quill CASS path against the Tantivy CASS incumbent.
//!
//! The Quill CASS path has an end-to-end test proving it ingests and answers
//! queries. That proves the plumbing, not the semantics — it cannot tell us
//! whether a consumer migrating off the Tantivy backend gets the same answers.
//! This does: one corpus, both engines, the same query set, compared.
//!
//! What is compared, and why:
//!
//! * `total_count` — the exact match count, which is independent of top-k. A
//!   capped hit list only measures where each engine cut the list, so a
//!   membership claim built from two capped lists is not a membership claim at
//!   all. The count is the uncapped signal.
//! * the returned identifier SET at a limit far above any expected match count,
//!   so the comparison is genuinely about membership rather than truncation.
//!
//! Rank and score are deliberately NOT asserted. The two engines do not share a
//! BM25 accumulation order, and pinning a rank permutation here would encode
//! the incumbent's tie-breaking as a Quill requirement rather than testing what
//! a consumer actually depends on.
//!
//! ```text
//! cargo test -p frankensearch --features quill,cass-compat \
//!   --test cass_quill_vs_tantivy_oracle
//! ```
#![cfg(all(feature = "quill", feature = "cass-compat"))]

use std::collections::BTreeSet;

use frankensearch::lexical_tantivy as tantivy_cass;
use frankensearch::quill;

const CORPUS: usize = 400;
const UNCAPPED_LIMIT: usize = 10_000;
const TIE_EXPANSION: usize = 64;

/// Deterministic corpus shared by both engines.
///
/// Vocabulary is small relative to the document count so that queries land on
/// real posting lists with many matches, which is where two engines are most
/// likely to disagree about membership.
fn corpus() -> Vec<(String, String, String, String)> {
    let vocabulary: Vec<String> = (0..256).map(|index| format!("term{index:03}")).collect();
    let agents = ["claude", "codex", "gemini", "local"];
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..CORPUS)
        .map(|index| {
            let content = (0..40)
                .map(|_| {
                    vocabulary[usize::try_from(next() % 256).expect("vocabulary index")].as_str()
                })
                .collect::<Vec<_>>()
                .join(" ");
            let title = format!("session {} about term{:03}", index / 20, index % 256);
            (
                format!("source{:04}", index / 10),
                title,
                content,
                agents[index % agents.len()].to_owned(),
            )
        })
        .collect()
}

fn queries() -> Vec<String> {
    let mut queries = Vec::new();
    for index in 0..24 {
        queries.push(format!("term{:03}", index * 7 % 256));
    }
    for index in 0..12 {
        queries.push(format!("term{:03} term{:03}", index, (index * 13) % 256));
    }
    for index in 0..8 {
        queries.push(format!("term{:03} AND term{:03}", index, (index * 5) % 256));
    }
    for index in 0..8 {
        queries.push(format!("term{:03} OR term{:03}", index, (index * 11) % 256));
    }
    queries.push("session".to_owned());
    queries.push("nonexistentterm".to_owned());
    queries
}

#[test]
fn cass_quill_membership_matches_the_tantivy_incumbent() {
    let rows = corpus();

    // ---- incumbent -------------------------------------------------------
    let mut oracle = tantivy_cass::CassTantivyIndex::in_memory_single_threaded_oracle()
        .expect("build the Tantivy CASS oracle");
    let oracle_documents: Vec<tantivy_cass::CassDocument> = rows
        .iter()
        .enumerate()
        .map(
            |(index, (source_id, title, content, agent))| tantivy_cass::CassDocument {
                agent: agent.clone(),
                workspace: Some("frankensearch".to_owned()),
                workspace_original: Some("FrankenSearch".to_owned()),
                source_path: format!("/transcripts/{source_id}.jsonl"),
                msg_idx: u64::try_from(index).expect("ordinal"),
                created_at: Some(1_700_000_000 + i64::try_from(index).expect("ordinal")),
                title: Some(title.clone()),
                content: content.clone(),
                source_id: source_id.clone(),
                origin_kind: "local".to_owned(),
                origin_host: None,
                conversation_id: Some(i64::try_from(index / 20).expect("conversation")),
            },
        )
        .collect();
    oracle
        .add_cass_documents(&oracle_documents)
        .expect("ingest the oracle corpus");
    oracle.commit().expect("commit the oracle corpus");

    // ---- Quill -----------------------------------------------------------
    let quill_documents: Vec<quill::cass::CassDocument> = rows
        .iter()
        .enumerate()
        .map(
            |(index, (source_id, title, content, agent))| quill::cass::CassDocument {
                agent: agent.clone(),
                workspace: Some("frankensearch".to_owned()),
                workspace_original: Some("FrankenSearch".to_owned()),
                source_path: format!("/transcripts/{source_id}.jsonl"),
                msg_idx: u64::try_from(index).expect("ordinal"),
                created_at: Some(1_700_000_000 + i64::try_from(index).expect("ordinal")),
                title: Some(title.clone()),
                content: content.clone(),
                source_id: source_id.clone(),
                origin_kind: "local".to_owned(),
                origin_host: None,
                conversation_id: Some(i64::try_from(index / 20).expect("conversation")),
            },
        )
        .collect();

    asupersync::test_utils::run_test_with_cx(move |cx| async move {
        let directory = tempfile::tempdir().expect("quill cass directory");
        let index = quill::QuillIndex::create_with_schema(
            &cx,
            directory.path(),
            quill::schema::CASS_SEMANTIC_SCHEMA,
            quill::QuillConfig::default(),
        )
        .await
        .expect("create the Quill CASS index");

        let projected: Vec<quill::SchemaDocument> = quill_documents
            .iter()
            .map(quill::cass::CassDocument::to_schema_document)
            .collect();
        index
            .index_schema_documents(&cx, &projected)
            .await
            .expect("ingest the Quill corpus");
        index.commit(&cx).await.expect("commit the Quill corpus");

        let reader = quill::QuillSearchIndex::open_with_schema(
            &cx,
            directory.path(),
            quill::schema::CASS_SEMANTIC_SCHEMA,
            quill::QuillConfig::default(),
        )
        .await
        .expect("open the Quill CASS reader");
        let parser = quill::query::CassQueryParser::new(quill::schema::CASS_SEMANTIC_SCHEMA)
            .expect("build the Quill CASS parser");

        assert_eq!(
            reader.doc_count().expect("quill doc count"),
            u64::try_from(CORPUS).expect("corpus size"),
            "both engines must hold the whole corpus before anything is compared"
        );

        let mut divergences = Vec::new();
        let mut nonempty = 0_usize;

        for raw in queries() {
            let observed = oracle
                .cass_oracle_observe_query(
                    &raw,
                    &tantivy_cass::CassQueryFilters::default(),
                    UNCAPPED_LIMIT,
                    TIE_EXPANSION,
                )
                .expect("observe the incumbent");

            let parsed = parser.parse(&raw, &quill::query::CassQueryFilters::default());
            let result = reader
                .search_preparsed_paginated(&cx, &parsed.query, UNCAPPED_LIMIT, 0, true)
                .expect("query Quill");

            let incumbent_ids: BTreeSet<&str> = observed
                .hits
                .iter()
                .map(|hit| hit.doc_id.as_str())
                .collect();
            let quill_ids: BTreeSet<&str> = result
                .hits
                .iter()
                .map(|hit| hit.document_id.as_str())
                .collect();

            if !incumbent_ids.is_empty() {
                nonempty += 1;
            }

            let incumbent_total = observed.total_count;
            let quill_total =
                usize::try_from(result.total_count.unwrap_or_default()).expect("quill total count");

            if incumbent_total != quill_total || incumbent_ids != quill_ids {
                let only_incumbent: Vec<&str> = incumbent_ids
                    .difference(&quill_ids)
                    .copied()
                    .take(4)
                    .collect();
                let only_quill: Vec<&str> = quill_ids
                    .difference(&incumbent_ids)
                    .copied()
                    .take(4)
                    .collect();
                divergences.push(format!(
                    "  {raw:>28}  incumbent={incumbent_total:<5} quill={quill_total:<5} \
                     only_incumbent={only_incumbent:?} only_quill={only_quill:?}"
                ));
            }
        }

        // A corpus on which every query returns nothing would make the
        // comparison vacuous, so prove the queries actually exercise postings.
        assert!(
            nonempty >= 20,
            "the incumbent matched on only {nonempty} queries; the differential would be vacuous"
        );

        assert!(
            divergences.is_empty(),
            "Quill CASS diverged from the Tantivy incumbent on {} of {} queries:\n{}",
            divergences.len(),
            queries().len(),
            divergences.join("\n")
        );
    });
}
