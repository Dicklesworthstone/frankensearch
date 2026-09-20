//! Publication validation contracts for the delta-preflight change in gh#51.
//!
//! These public-API regressions also pass with the incumbent full preflight.
//! They deliberately do not claim to measure skipped validation work: the
//! private delta-selection tests belong beside Keeper's preflight validator.

#![cfg(unix)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use asupersync::Cx;
use asupersync::sync::LockError;
use frankensearch_core::{IndexableDocument, LexicalRead, LexicalWrite, ScoredResult};
use frankensearch_quill::{
    DEFAULT_SCHEMA, KeeperError, KeeperSnapshot, KeeperWriter, Manifest, QuillConfig, QuillIndex,
    QuillSearchIndex, SegmentReader, load_manifest_pair,
};

fn config() -> QuillConfig {
    QuillConfig {
        deterministic_ingest: true,
        ..QuillConfig::default()
    }
}

fn retained_manifest(writer: &KeeperWriter) -> Manifest {
    writer
        .snapshot()
        .expect("authoritative writer snapshot")
        .loaded_manifest()
        .manifest
        .clone()
}

fn successor(writer: &KeeperWriter) -> Manifest {
    let mut manifest = retained_manifest(writer);
    manifest.generation = manifest
        .generation
        .checked_add(1)
        .expect("fixture generation must not overflow");
    manifest
}

async fn append_document(
    index: &QuillIndex,
    cx: &Cx,
    directory: &Path,
    id: &str,
    text: &str,
) -> Manifest {
    LexicalWrite::index_document(index, cx, &IndexableDocument::new(id, text))
        .await
        .expect("stage fixture document");
    LexicalWrite::commit(index, cx)
        .await
        .expect("publish fixture document");
    load_manifest_pair(directory)
        .expect("read fixture MANIFEST")
        .manifest
}

/// Stage real FSLX files without manufacturing headers, hashes, or docid ranges.
/// The donor and recipient have separate writer admissions. Never overwrite a
/// retained recipient file while constructing a successor proposal.
fn stage_segments(
    donor: &Path,
    recipient: &Path,
    manifest: &Manifest,
) -> BTreeMap<u64, PathBuf> {
    let mut staged = BTreeMap::new();
    for entry in std::fs::read_dir(donor).expect("enumerate donor files") {
        let entry = entry.expect("read donor directory entry");
        if !entry.file_type().expect("inspect donor entry").is_file() {
            continue;
        }
        let source = entry.path();
        let Ok(reader) = SegmentReader::open_published(&source, DEFAULT_SCHEMA) else {
            continue;
        };
        let segment_id = reader.header().segment_id;
        if !manifest
            .segments
            .iter()
            .any(|segment| segment.segment_id == segment_id)
        {
            continue;
        }
        let destination = recipient.join(entry.file_name());
        if destination.exists() {
            assert_eq!(
                std::fs::read(&destination).expect("read retained recipient segment"),
                std::fs::read(&source).expect("read retained donor segment"),
                "fixture staging must leave retained segment bytes unchanged"
            );
        } else {
            std::fs::copy(&source, &destination).expect("stage new recipient segment");
        }
        assert!(staged.insert(segment_id, destination).is_none());
    }
    assert_eq!(
        staged.len(),
        manifest.segments.len(),
        "stage every proposed binding"
    );
    staged
}

fn directory_bytes(directory: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    std::fs::read_dir(directory)
        .expect("enumerate authority files")
        .filter_map(|entry| {
            let entry = entry.expect("read authority entry");
            if !entry.file_type().expect("inspect authority entry").is_file() {
                return None;
            }
            let path = entry.path();
            let bytes = std::fs::read(&path).expect("read authority file");
            Some((path, bytes))
        })
        .collect()
}

fn corrupt_same_length(path: &Path) -> Vec<u8> {
    let original = std::fs::read(path).expect("read intact segment");
    assert!(original.len() > 64, "fixture must contain a real FSLX body");
    let mut damaged = original.clone();
    let offset = damaged.len() / 2;
    damaged[offset] ^= 0x80;
    // Do not truncate a file that an existing snapshot may still have mapped.
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .expect("open segment for deterministic corruption");
    std::io::Write::write_all(&mut file, &damaged).expect("write same-length corruption");
    file.sync_all().expect("sync corrupted fixture");
    assert_eq!(
        std::fs::metadata(path).expect("inspect damaged file").len(),
        u64::try_from(original.len()).expect("fixture length fits u64")
    );
    original
}

fn restore_segment(path: &Path, original: &[u8]) {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .expect("open segment for fixture restoration");
    std::io::Write::write_all(&mut file, original).expect("restore original segment bytes");
    file.sync_all().expect("sync restored fixture");
}

async fn assert_fresh_search_matches(cx: &Cx, directory: &Path, expected: &[ScoredResult]) {
    let reopened = QuillSearchIndex::open(cx, directory, config())
        .await
        .expect("fresh public reader verifies the published files");
    let actual = LexicalRead::search(&reopened, cx, "shared", 10)
        .await
        .expect("search through fresh public reader");
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.doc_id, expected.doc_id, "fresh-open result order and ids");
        assert_eq!(
            actual.score.to_bits(),
            expected.score.to_bits(),
            "fresh-open scores"
        );
    }
}

#[test]
fn new_corrupt_segment_is_rejected_with_retained_segments_and_retry_reopens_cleanly() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let donor_dir = tempfile::tempdir().expect("donor directory");
        let recipient_dir = tempfile::tempdir().expect("recipient directory");
        let donor = QuillIndex::create(&cx, donor_dir.path(), config())
            .await
            .expect("create donor");
        let mut writer = KeeperWriter::create(&cx, recipient_dir.path(), DEFAULT_SCHEMA)
            .await
            .expect("create recipient writer");
        let mut first = append_document(
            &donor,
            &cx,
            donor_dir.path(),
            "retained",
            "shared alpha retained document",
        )
        .await;
        first.generation = successor(&writer).generation;
        stage_segments(donor_dir.path(), recipient_dir.path(), &first);
        writer
            .publish(&cx, &first)
            .await
            .expect("publish retained segment");
        let retained = retained_manifest(&writer);

        let mut proposed = append_document(
            &donor,
            &cx,
            donor_dir.path(),
            "added",
            "shared beta added document",
        )
        .await;
        proposed.generation = retained.generation + 1;
        for segment in &retained.segments {
            assert!(
                proposed.segments.contains(segment),
                "fixture must carry exact retained bindings"
            );
        }
        let staged = stage_segments(donor_dir.path(), recipient_dir.path(), &proposed);
        let added: Vec<_> = staged
            .iter()
            .filter(|(id, _)| {
                !retained
                    .segments
                    .iter()
                    .any(|segment| segment.segment_id == **id)
            })
            .collect();
        assert_eq!(
            added.len(),
            1,
            "exercise a genuine retained-plus-new publication"
        );
        let damaged_path = added[0].1;
        let original = corrupt_same_length(damaged_path);
        let before = directory_bytes(recipient_dir.path());
        let error = writer
            .publish(&cx, &proposed)
            .await
            .err()
            .expect("corrupt delta must not publish");
        assert!(
            matches!(error, KeeperError::SegmentOpen { .. }),
            "unexpected failure: {error}"
        );
        assert_eq!(
            directory_bytes(recipient_dir.path()),
            before,
            "preflight refusal changes no authority or staged files"
        );
        assert!(!writer.publication_awaits_reconciliation());
        assert_eq!(retained_manifest(&writer), retained);
        assert_eq!(
            KeeperSnapshot::open(recipient_dir.path(), DEFAULT_SCHEMA)
                .expect("old generation still opens")
                .loaded_manifest()
                .manifest,
            retained
        );

        restore_segment(damaged_path, &original);
        writer
            .publish(&cx, &proposed)
            .await
            .expect("publish repaired caller-owned delta");
        let expected = LexicalRead::search(&donor, &cx, "shared", 10)
            .await
            .expect("donor search");
        assert_eq!(expected.len(), 2);
        drop(writer);
        assert_fresh_search_matches(&cx, recipient_dir.path(), &expected).await;
    });
}

#[test]
fn changed_witness_for_a_retained_id_cannot_be_published_as_unchanged() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("index directory");
        let index = QuillIndex::create(&cx, directory.path(), config())
            .await
            .expect("create index");
        append_document(
            &index,
            &cx,
            directory.path(),
            "retained",
            "shared retained document",
        )
        .await;
        drop(index);
        let mut writer = KeeperWriter::open(&cx, directory.path(), DEFAULT_SCHEMA)
            .await
            .expect("open writer");
        let retained = retained_manifest(&writer);
        assert_eq!(retained.segments.len(), 1);
        for change_length in [false, true] {
            let mut proposed = successor(&writer);
            if change_length {
                proposed.segments[0].file_len += 1;
            } else {
                proposed.segments[0].file_xxh3 ^= 1;
            }
            assert_eq!(proposed.segments[0].segment_id, retained.segments[0].segment_id);
            let before = directory_bytes(directory.path());
            let error = writer
                .publish(&cx, &proposed)
                .await
                .err()
                .expect("changed binding must be refused");
            assert!(
                matches!(
                    error,
                    KeeperError::SegmentMetadataMismatch { .. }
                        | KeeperError::InvalidTransition { .. }
                ),
                "unexpected failure: {error}"
            );
            assert_eq!(directory_bytes(directory.path()), before);
            assert!(!writer.publication_awaits_reconciliation());
            assert_eq!(retained_manifest(&writer), retained);
        }
        let valid = successor(&writer);
        writer
            .publish(&cx, &valid)
            .await
            .expect("rejections must not poison a valid retry");
    });
}

#[test]
fn cancelled_retained_only_publication_keeps_authority_and_retries_once() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("index directory");
        let index = QuillIndex::create(&cx, directory.path(), config())
            .await
            .expect("create index");
        append_document(
            &index,
            &cx,
            directory.path(),
            "retained",
            "shared retained document",
        )
        .await;
        let expected = LexicalRead::search(&index, &cx, "shared", 10)
            .await
            .expect("control search");
        assert_eq!(expected.len(), 1);
        drop(index);
        let mut writer = KeeperWriter::open(&cx, directory.path(), DEFAULT_SCHEMA)
            .await
            .expect("open writer");
        let retained = retained_manifest(&writer);
        let proposed = successor(&writer);
        assert_eq!(proposed.segments, retained.segments);
        let before = directory_bytes(directory.path());
        for _ in 0..2 {
            cx.set_cancel_requested(true);
            let error = writer
                .publish(&cx, &proposed)
                .await
                .err()
                .expect("cancelled publication must stop");
            cx.set_cancel_requested(false);
            assert!(matches!(
                error,
                KeeperError::PublishLock {
                    source: LockError::Cancelled
                }
            ));
            assert_eq!(directory_bytes(directory.path()), before);
            assert!(!writer.publication_awaits_reconciliation());
            assert_eq!(retained_manifest(&writer), retained);
        }
        writer
            .publish(&cx, &proposed)
            .await
            .expect("uncancelled retry publishes exactly once");
        assert_eq!(retained_manifest(&writer).generation, retained.generation + 1);
        assert!(!writer.publication_awaits_reconciliation());
        drop(writer);
        assert_fresh_search_matches(&cx, directory.path(), &expected).await;
    });
}

#[test]
fn fresh_open_rechecks_retained_bytes_after_repeated_successful_publications() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("index directory");
        let index = QuillIndex::create(&cx, directory.path(), config())
            .await
            .expect("create index");
        let initial = append_document(
            &index,
            &cx,
            directory.path(),
            "retained",
            "shared retained document",
        )
        .await;
        let expected = LexicalRead::search(&index, &cx, "shared", 10)
            .await
            .expect("control search");
        assert_eq!(expected.len(), 1);
        drop(index);
        let mut writer = KeeperWriter::open(&cx, directory.path(), DEFAULT_SCHEMA)
            .await
            .expect("open writer");
        for _ in 0..3 {
            let proposed = successor(&writer);
            assert_eq!(proposed.segments, initial.segments);
            writer
                .publish(&cx, &proposed)
                .await
                .expect("publish retained-only successor");
            assert_fresh_search_matches(&cx, directory.path(), &expected).await;
        }
        let paths = stage_segments(directory.path(), directory.path(), &initial);
        assert_eq!(paths.len(), 1);
        let path = paths.values().next().expect("retained segment path");
        let original = corrupt_same_length(path);
        let before = directory_bytes(directory.path());
        let error = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)
            .err()
            .expect("fresh open must hash retained bytes again");
        assert!(
            matches!(error, KeeperError::SegmentOpen { .. }),
            "unexpected failure: {error}"
        );
        assert!(
            QuillSearchIndex::open(&cx, directory.path(), config())
                .await
                .is_err()
        );
        assert_eq!(
            directory_bytes(directory.path()),
            before,
            "read-only opens must not repair corruption or change authority"
        );
        restore_segment(path, &original);
        drop(writer);
        assert_fresh_search_matches(&cx, directory.path(), &expected).await;
    });
}
