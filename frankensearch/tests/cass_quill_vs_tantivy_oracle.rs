//! Differential: the native Quill CASS path against the Tantivy CASS incumbent.
//!
//! The Quill CASS path has an end-to-end test proving it ingests and answers
//! queries. That proves the plumbing, not the semantics — it cannot tell us
//! whether a consumer migrating off the Tantivy backend gets the same answers.
//! This does: one corpus, both engines, one probe set, compared.
//!
//! The comparison itself lives in [`frankensearch::cass_equivalence`] so the
//! identical gate can run against a consumer's *real* exported corpus rather
//! than only this synthetic one. This file is the CI-facing half: a corpus that
//! exercises the lowerings, and the assertions that make a green meaningful.
//!
//! ```text
//! cargo test -p frankensearch --features cass-equivalence \
//!   --test cass_quill_vs_tantivy_oracle
//! ```
#![cfg(feature = "cass-equivalence")]

use frankensearch::cass_equivalence::{
    CassEquivalenceReport, cass_engine_equivalence_report, default_probe_set,
};
use frankensearch::quill;

const CORPUS: usize = 400;

/// Vocabulary the synthetic corpus is drawn from.
///
/// Small relative to the document count so probes land on real posting lists
/// with many matches, which is where two engines are most likely to disagree.
fn vocabulary() -> Vec<String> {
    (0..256).map(|index| format!("term{index:03}")).collect()
}

/// Deterministic corpus.
///
/// Deliberately includes hyphenated and CJK text, because the CASS schema
/// carries a hyphen-normalizing analyzer and CJK bigrams — a corpus of plain
/// ASCII words would let those two analyzers diverge without any probe noticing.
fn corpus() -> Vec<quill::cass::CassDocument> {
    let vocabulary = vocabulary();
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
            let mut content = (0..40)
                .map(|_| {
                    vocabulary[usize::try_from(next() % 256).expect("vocabulary index")].as_str()
                })
                .collect::<Vec<_>>()
                .join(" ");
            if index % 5 == 0 {
                content.push_str(" well-known multi-part-token");
            }
            if index % 7 == 0 {
                content.push_str(" 日本語 検索 テキスト");
            }
            quill::cass::CassDocument {
                agent: agents[index % agents.len()].to_owned(),
                workspace: Some("frankensearch".to_owned()),
                workspace_original: Some("FrankenSearch".to_owned()),
                source_path: format!("/transcripts/source{:04}.jsonl", index / 10),
                msg_idx: u64::try_from(index).expect("ordinal"),
                created_at: Some(1_700_000_000 + i64::try_from(index).expect("ordinal")),
                title: Some(format!(
                    "session {} about term{:03}",
                    index / 20,
                    index % 256
                )),
                content,
                source_id: format!("source{:04}", index / 10),
                origin_kind: "local".to_owned(),
                origin_host: None,
                conversation_id: Some(i64::try_from(index / 20).expect("conversation")),
            }
        })
        .collect()
}

#[test]
fn cass_quill_membership_matches_the_tantivy_incumbent() {
    let documents = corpus();
    let probes = default_probe_set(&vocabulary());
    asupersync::test_utils::run_test_with_cx(move |cx| async move {
        let report = cass_engine_equivalence_report(&cx, &documents, &probes)
            .await
            .expect("run the CASS engine equivalence comparison");
        assert_report_is_equivalent(&report);
    });
}

fn assert_report_is_equivalent(report: &CassEquivalenceReport) {
    assert_eq!(
        report.incumbent_doc_count, report.quill_doc_count,
        "both engines must hold the whole corpus before membership means anything"
    );

    // A corpus on which every probe returned nothing would agree perfectly and
    // prove nothing. Require that most probes actually hit postings.
    assert!(
        report.discriminating_probes() >= 25,
        "only {} of {} probes matched anything in the incumbent; the differential is vacuous",
        report.discriminating_probes(),
        report.probes.len()
    );

    assert!(
        report.equivalent(),
        "Quill CASS diverged from the Tantivy incumbent on {} of {} probes:\n{}",
        report.divergences().len(),
        report.probes.len(),
        report.render_divergences()
    );
}
