//! Public-API tests for multi-publication maintenance and partial recovery.
#![cfg(unix)]
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use asupersync::Cx;
use asupersync::test_utils::run_test_with_cx;
use frankensearch_core::{IndexableDocument, LexicalWrite};
use frankensearch_quill::{
    QuillConfig, QuillIndex, QuillSearchIndex, QuillSearchResult, QuillSearchSnapshot,
    SessionMergeBudget, SessionMergeError, SessionMergePolicy, SessionMergeStop,
    load_manifest_pair,
};

const QUERIES: [&str; 3] = ["shared", "\"amber rust\"", "amber OR cobalt"];
type StoredRows = Vec<Vec<Option<Vec<u8>>>>;

fn config() -> QuillConfig {
    QuillConfig {
        deterministic_ingest: true,
        ..QuillConfig::default()
    }
}

fn policy() -> SessionMergePolicy {
    SessionMergePolicy {
        fanout: 2,
        max_input_docs: 4,
        max_input_bytes: u64::MAX,
    }
}

fn budget() -> SessionMergeBudget {
    SessionMergeBudget {
        max_merges: 8,
        max_input_docs: u64::MAX,
        max_input_bytes: u64::MAX,
    }
}

async fn fixture(cx: &Cx, directory: &Path) -> QuillIndex {
    let index = QuillIndex::create(cx, directory, config()).await.unwrap();
    for batch in 0..6 {
        // One ingest batch per source: a separate call that replaces a
        // published document first publishes the pending rows (the
        // engine-neutral LexicalWrite contract), which would split the
        // replacing source in two.
        let rows: Vec<_> = (0..2)
            .map(|row| {
                let id = if batch == 5 && row == 1 {
                    "maintenance-0-0".to_owned()
                } else {
                    format!("maintenance-{batch}-{row}")
                };
                let text = if row == 0 {
                    "shared amber rust"
                } else {
                    "shared cobalt python"
                };
                IndexableDocument::new(id.as_str(), text)
            })
            .collect();
        LexicalWrite::index_documents(&index, cx, &rows)
            .await
            .unwrap();
        LexicalWrite::commit(&index, cx).await.unwrap();
    }
    assert_eq!(index.snapshot().unwrap().segments().len(), 6);
    assert_eq!(index.doc_count().unwrap(), 11);
    assert_eq!(
        index.collect_docids(cx, "shared").unwrap(),
        (1..12).collect::<Vec<_>>()
    );
    // Reopen without a live ingest lease so a publication-failure fixture can
    // target the second manifest generation rather than a lease-retirement step.
    drop(index);
    QuillIndex::open(cx, directory, config()).await.unwrap()
}

fn queries(index: &QuillIndex, cx: &Cx) -> Vec<QuillSearchResult> {
    QUERIES
        .iter()
        .map(|query| index.search_paginated(cx, query, 20, 0, true).unwrap())
        .collect()
}

fn stored(snapshot: &QuillSearchSnapshot) -> StoredRows {
    (0..12)
        .map(|docid| {
            (0..5)
                .map(|field| snapshot.stored_field_value(field, docid).unwrap())
                .collect()
        })
        .collect()
}

async fn assert_reopened(
    cx: &Cx,
    directory: &Path,
    expected_queries: &[QuillSearchResult],
    expected_stored: &StoredRows,
) {
    let reader = QuillSearchIndex::open(cx, directory, config())
        .await
        .unwrap();
    for (query, expected) in QUERIES.iter().zip(expected_queries) {
        assert_eq!(
            reader.search_paginated(cx, query, 20, 0, true).unwrap(),
            *expected
        );
    }
    for (docid, fields) in expected_stored.iter().enumerate() {
        for (field, expected) in fields.iter().enumerate() {
            let value = reader
                .stored_field_value(u16::try_from(field).unwrap(), u32::try_from(docid).unwrap())
                .unwrap();
            assert_eq!(value.as_deref(), expected.as_deref());
        }
    }
}

fn directory_bytes(directory: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap())
        .filter(|entry| entry.file_type().unwrap().is_file())
        .map(|entry| (entry.path(), std::fs::read(entry.path()).unwrap()))
        .collect()
}

fn output_path(directory: &Path, id: u64) -> PathBuf {
    directory.join(format!("seg-{id:016x}.fslx"))
}

#[test]
fn repeated_merges_charge_rewritten_inputs_and_preserve_search_and_ingest() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let index = fixture(&cx, directory.path()).await;
        let initial = load_manifest_pair(directory.path()).unwrap().manifest;
        let expected_queries = queries(&index, &cx);
        let held = index.search_snapshot().unwrap();
        let expected_stored = stored(&held);
        assert!(expected_stored[0].iter().all(Option::is_none));
        let report = index
            .merge_sessions(
                &cx,
                SessionMergePolicy {
                    max_input_docs: 100,
                    ..policy()
                },
                SessionMergeBudget {
                    max_merges: 4,
                    ..budget()
                },
                0xc100,
                0,
            )
            .await
            .unwrap();
        assert_eq!(report.stop, SessionMergeStop::MergeLimit);
        assert_eq!(report.progress.merges.len(), 4);
        // Three pairs cost twelve physical rows, then a pair of their outputs
        // costs eight more. The tombstoned row is charged on both rewrites.
        assert_eq!(report.progress.input_docs, 20);
        let originals: u64 = initial.segments.iter().map(|source| source.file_len).sum();
        let rewritten: u64 = [0xc100, 0xc101]
            .iter()
            .map(|id| {
                std::fs::metadata(output_path(directory.path(), *id))
                    .unwrap()
                    .len()
            })
            .sum();
        assert_eq!(report.progress.input_bytes, originals + rewritten);
        for (receipt, sources) in report.progress.merges[..3]
            .iter()
            .zip(initial.segments.chunks(2))
        {
            assert_eq!(
                receipt.plan.source_segment_ids,
                sources
                    .iter()
                    .map(|source| source.segment_id)
                    .collect::<Vec<_>>(),
            );
            assert_eq!(receipt.plan.input_docs, 4);
        }
        assert_eq!(
            report.progress.merges[3].plan.source_segment_ids,
            [0xc100, 0xc101]
        );
        assert!(
            report
                .progress
                .merges
                .windows(2)
                .all(|pair| pair[0].generation < pair[1].generation)
        );
        let published = index.snapshot().unwrap();
        assert_eq!(published.segments().len(), 2);
        assert_eq!(published.doc_count(), 11);
        assert_eq!(
            published.loaded_manifest().manifest.field_stats,
            initial.field_stats
        );
        assert_eq!(queries(&index, &cx), expected_queries);
        assert_eq!(stored(&held), expected_stored);
        assert_eq!(stored(&index.search_snapshot().unwrap()), expected_stored);
        assert_reopened(&cx, directory.path(), &expected_queries, &expected_stored).await;
        LexicalWrite::index_document(
            &index,
            &cx,
            &IndexableDocument::new("after-maintenance", "shared additional document"),
        )
        .await
        .unwrap();
        LexicalWrite::commit(&index, &cx).await.unwrap();
        assert_eq!(index.doc_count().unwrap(), 12);
        assert_eq!(stored(&held), expected_stored);
        let cold = QuillSearchIndex::open(&cx, directory.path(), config())
            .await
            .unwrap();
        assert_eq!(
            cold.search_paginated(&cx, "shared", 20, 0, true).unwrap(),
            index.search_paginated(&cx, "shared", 20, 0, true).unwrap(),
        );
    });
}

#[test]
fn cumulative_row_byte_and_publication_caps_stop_before_another_attempt() {
    run_test_with_cx(|cx| async move {
        for case in 0..4 {
            let directory = tempfile::tempdir().unwrap();
            let index = fixture(&cx, directory.path()).await;
            let initial = load_manifest_pair(directory.path()).unwrap().manifest;
            let first_four_bytes: u64 = initial.segments[..4].iter().map(|s| s.file_len).sum();
            let (limits, count, stop) = match case {
                0 => (
                    SessionMergeBudget {
                        max_input_docs: 8,
                        ..budget()
                    },
                    2,
                    SessionMergeStop::InputBudget,
                ),
                1 => (
                    SessionMergeBudget {
                        max_input_bytes: first_four_bytes,
                        ..budget()
                    },
                    2,
                    SessionMergeStop::InputBudget,
                ),
                2 => (
                    SessionMergeBudget {
                        max_merges: 1,
                        ..budget()
                    },
                    1,
                    SessionMergeStop::MergeLimit,
                ),
                _ => (
                    SessionMergeBudget {
                        max_input_docs: 9,
                        ..budget()
                    },
                    2,
                    SessionMergeStop::NoEligibleRunWithinBudget,
                ),
            };
            let report = index
                .merge_sessions(&cx, policy(), limits, 0xc200, 0)
                .await
                .unwrap();
            assert_eq!(report.stop, stop);
            assert_eq!(report.progress.merges.len(), count);
            assert_eq!(report.progress.input_docs, 4 * count as u64);
            assert_eq!(
                report.progress.input_bytes,
                initial.segments[..2 * count]
                    .iter()
                    .map(|s| s.file_len)
                    .sum::<u64>(),
            );
            assert!(report.progress.input_docs <= limits.max_input_docs);
            assert!(report.progress.input_bytes <= limits.max_input_bytes);
            assert_eq!(index.snapshot().unwrap().segments().len(), 6 - count);
            assert!(!output_path(directory.path(), 0xc200 + count as u64).exists());
            assert_eq!(index.doc_count().unwrap(), 11);
        }
    });
}

#[test]
fn failed_second_publication_keeps_first_receipt_and_resumes_after_reconciliation() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let index = fixture(&cx, directory.path()).await;
        let initial = load_manifest_pair(directory.path()).unwrap().manifest;
        let expected_queries = queries(&index, &cx);
        let held = index.search_snapshot().unwrap();
        let expected_stored = stored(&held);
        let blocked = directory
            .path()
            .join(format!(".tmp-manifest-{}", initial.generation + 2));
        let conflicting_proposal = b"preserve the conflicting second proposal";
        std::fs::write(&blocked, conflicting_proposal).unwrap();
        let failure = index
            .merge_sessions(&cx, policy(), budget(), 0xc300, 0)
            .await
            .unwrap_err();
        assert_eq!(failure.progress.merges.len(), 1);
        assert_eq!(failure.progress.input_docs, 4);
        assert_eq!(
            failure.progress.input_bytes,
            initial.segments[..2]
                .iter()
                .map(|s| s.file_len)
                .sum::<u64>()
        );
        assert_eq!(failure.progress.merges[0].output_segment_id, 0xc300);
        let attempt = failure
            .attempted
            .as_ref()
            .expect("the second concat was attempted");
        assert_eq!(attempt.output_segment_id, 0xc301);
        assert_eq!(attempt.created_unix_s, 0);
        assert_eq!(
            attempt.plan.source_segment_ids,
            initial.segments[2..4]
                .iter()
                .map(|s| s.segment_id)
                .collect::<Vec<_>>()
        );
        assert!(matches!(
            failure.source.as_ref(),
            SessionMergeError::Index(_)
        ));
        assert!(failure.source.to_string().contains("temp"));
        let first_bytes = std::fs::read(output_path(directory.path(), 0xc300)).unwrap();
        let attempted_bytes = std::fs::read(output_path(directory.path(), 0xc301)).unwrap();
        let authority = load_manifest_pair(directory.path()).unwrap().manifest;
        assert_eq!(authority.generation, failure.progress.merges[0].generation);
        assert_eq!(authority.segments.len(), 5);
        assert!(authority.segments.iter().any(|s| s.segment_id == 0xc300));
        assert!(!authority.segments.iter().any(|s| s.segment_id == 0xc301));
        assert!(!output_path(directory.path(), 0xc302).exists());
        assert_eq!(std::fs::read(&blocked).unwrap(), conflicting_proposal);
        assert_eq!(stored(&held), expected_stored);
        assert_reopened(&cx, directory.path(), &expected_queries, &expected_stored).await;

        drop(index);
        // Preserve the blocker instead of deleting it, then use the real writer
        // reopen contract before resuming at the reconciled attempted identity.
        std::fs::rename(
            &blocked,
            directory.path().join("preserved-proposal.fixture"),
        )
        .unwrap();
        let recovered = QuillIndex::open(&cx, directory.path(), config())
            .await
            .unwrap();
        assert_eq!(
            recovered.snapshot().unwrap().loaded_manifest().manifest,
            authority
        );
        let resumed = recovered
            .merge_sessions(
                &cx,
                policy(),
                SessionMergeBudget {
                    max_merges: 2,
                    max_input_docs: 8,
                    ..budget()
                },
                attempt.output_segment_id,
                attempt.created_unix_s,
            )
            .await
            .unwrap();
        assert_eq!(resumed.progress.merges.len(), 2);
        assert_eq!(resumed.progress.input_docs, 8);
        assert_eq!(resumed.progress.merges[0].plan, attempt.plan);
        assert_eq!(recovered.snapshot().unwrap().segments().len(), 3);
        assert_eq!(recovered.doc_count().unwrap(), 11);
        assert_eq!(
            std::fs::read(output_path(directory.path(), 0xc300)).unwrap(),
            first_bytes
        );
        assert_eq!(
            std::fs::read(output_path(directory.path(), 0xc301)).unwrap(),
            attempted_bytes
        );
        assert_eq!(queries(&recovered, &cx), expected_queries);
        assert_eq!(stored(&held), expected_stored);
        assert_reopened(&cx, directory.path(), &expected_queries, &expected_stored).await;
    });
}

#[test]
fn no_op_invalid_sequence_cancellation_and_collision_preserve_authority() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let index = fixture(&cx, directory.path()).await;
        let initial = load_manifest_pair(directory.path()).unwrap().manifest;
        let before = directory_bytes(directory.path());
        let report = index
            .merge_sessions(&cx, SessionMergePolicy::default(), budget(), 0xc400, 0)
            .await
            .unwrap();
        assert!(report.progress.merges.is_empty());
        assert_eq!(report.stop, SessionMergeStop::NoEligibleRunWithinBudget);
        assert_eq!(directory_bytes(directory.path()), before);

        let invalid = index
            .merge_sessions(&cx, policy(), budget(), u64::MAX, 0)
            .await
            .unwrap_err();
        assert!(matches!(
            invalid.source.as_ref(),
            SessionMergeError::InvalidPolicy { .. }
        ));
        assert!(invalid.progress.merges.is_empty());
        assert!(invalid.attempted.is_none());
        assert_eq!(directory_bytes(directory.path()), before);

        cx.set_cancel_requested(true);
        let cancelled = index
            .merge_sessions(&cx, policy(), budget(), 0xc400, 0)
            .await
            .unwrap_err();
        cx.set_cancel_requested(false);
        assert!(matches!(
            cancelled.source.as_ref(),
            SessionMergeError::Cancelled
        ));
        assert!(cancelled.progress.merges.is_empty());
        assert!(cancelled.attempted.is_none());
        assert_eq!(directory_bytes(directory.path()), before);

        let collision = index
            .merge_sessions(&cx, policy(), budget(), initial.segments[0].segment_id, 0)
            .await
            .unwrap_err();
        assert!(collision.progress.merges.is_empty());
        assert_eq!(
            collision.attempted.unwrap().output_segment_id,
            initial.segments[0].segment_id
        );
        assert_eq!(directory_bytes(directory.path()), before);
        assert_eq!(
            load_manifest_pair(directory.path()).unwrap().manifest,
            initial
        );
    });
}
