//! Publication validation contracts for the delta-preflight change in gh#51.
//!
//! These public-API regressions protect both the incumbent and delta preflight.
//! They deliberately do not claim to measure skipped validation work: the
//! private delta-selection tests belong beside Keeper's preflight validator.

#![cfg(unix)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
#[cfg(feature = "conformance-internals")]
use std::sync::Arc;

use asupersync::Cx;
use asupersync::sync::LockError;
#[cfg(feature = "conformance-internals")]
use frankensearch_core::SearchError;
use frankensearch_core::{IndexableDocument, LexicalRead, LexicalWrite, ScoredResult};
#[cfg(feature = "conformance-internals")]
use frankensearch_quill::index::ConformanceCancellationStage;
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
fn stage_segments(donor: &Path, recipient: &Path, manifest: &Manifest) -> BTreeMap<u64, PathBuf> {
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
            if !entry
                .file_type()
                .expect("inspect authority entry")
                .is_file()
            {
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
        assert_eq!(
            actual.doc_id, expected.doc_id,
            "fresh-open result order and ids"
        );
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
            assert_eq!(
                proposed.segments[0].segment_id,
                retained.segments[0].segment_id
            );
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
        assert_eq!(
            retained_manifest(&writer).generation,
            retained.generation + 1
        );
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

#[cfg(feature = "conformance-internals")]
#[test]
fn durable_commit_cancellation_retains_exact_delta_until_one_successful_retry() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("durable cancellation directory");
        let index = QuillIndex::create(&cx, directory.path(), config())
            .await
            .expect("create durable index");
        let retained = append_document(
            &index,
            &cx,
            directory.path(),
            "retained",
            "shared retained document",
        )
        .await;
        let expected_before = LexicalRead::search(&index, &cx, "shared", 10)
            .await
            .expect("retained-generation control search");
        assert_eq!(expected_before.len(), 1);
        let manifest_before = std::fs::read(directory.path().join("MANIFEST"))
            .expect("read retained authority bytes");
        let snapshot_before = index.search_snapshot().expect("retained public snapshot");

        LexicalWrite::index_document(
            &index,
            &cx,
            &IndexableDocument::new("added", "shared added document"),
        )
        .await
        .expect("stage one new document beside the retained segment");
        let controller = index.conformance_cancellation_controller();
        let mut pending_before = None;
        let mut staged_before = None;
        for _ in 0..2 {
            controller
                .arm(ConformanceCancellationStage::CommitPublication, 1)
                .expect("arm real pre-publication checkpoint");
            let error = LexicalWrite::commit(&index, &cx)
                .await
                .expect_err("checkpoint must cancel the actual durable commit");
            assert!(matches!(
                error,
                SearchError::Cancelled { phase, reason }
                    if phase == "commit publish"
                        && reason == "Quill observed request cancellation"
            ));
            assert!(controller.fired());
            assert_eq!(controller.observed_checkpoints(), 1);
            assert!(cx.is_cancel_requested());
            assert!(index.has_uncommitted_changes());
            assert!(Arc::ptr_eq(
                &snapshot_before,
                &index
                    .search_snapshot()
                    .expect("retained snapshot stays readable"),
            ));
            assert_eq!(
                std::fs::read(directory.path().join("MANIFEST"))
                    .expect("read authority after cancellation"),
                manifest_before
            );

            let pending = index
                .conformance_pending_writer_state()
                .expect("capture exact pending transaction");
            assert_eq!(pending.dirty_shard_count(), 0);
            assert_eq!(pending.pending_identity_count(), 0);
            assert_eq!(pending.uncommitted_id_count(), 1);
            assert_eq!(pending.pending_segment_count(), 1);
            assert_eq!(pending.pending_owned_segment_count(), 1);
            assert!(pending.pending_manifest_present());
            let staged = directory_bytes(directory.path());
            if let Some(before) = &pending_before {
                assert_eq!(
                    &pending, before,
                    "retry retains the exact prepared transaction"
                );
            }
            if let Some(before) = &staged_before {
                assert_eq!(
                    &staged, before,
                    "retry changes no staged or authoritative bytes"
                );
            }
            pending_before = Some(pending);
            staged_before = Some(staged);

            controller.disarm();
            cx.set_cancel_requested(false);
            assert_fresh_search_matches(&cx, directory.path(), &expected_before).await;
        }

        LexicalWrite::commit(&index, &cx)
            .await
            .expect("uncancelled retry publishes the staged delta");
        assert!(!index.has_uncommitted_changes());
        let installed = load_manifest_pair(directory.path())
            .expect("read successor authority")
            .manifest;
        assert_eq!(installed.generation, retained.generation + 1);
        for segment in &retained.segments {
            assert!(installed.segments.contains(segment));
        }
        let snapshot_after = index.search_snapshot().expect("successor public snapshot");
        assert_eq!(
            snapshot_after.snapshot_epoch(),
            snapshot_before.snapshot_epoch() + 1
        );
        assert_eq!(
            snapshot_after.keeper_generation(),
            snapshot_before.keeper_generation() + 1
        );
        let expected_after = LexicalRead::search(&index, &cx, "shared", 10)
            .await
            .expect("successful-retry control search");
        assert_eq!(expected_after.len(), 2);
        assert!(expected_after.iter().any(|hit| hit.doc_id == "retained"));
        assert!(expected_after.iter().any(|hit| hit.doc_id == "added"));
        drop(index);
        assert_fresh_search_matches(&cx, directory.path(), &expected_after).await;
    });
}

/// Produce one real segment with two live rows, so a tombstone-only proposal
/// keeps the immutable id, range, length, and file witness exactly unchanged.
async fn two_row_manifest(cx: &Cx, directory: &Path) -> Manifest {
    let index = QuillIndex::create(
        cx,
        directory,
        QuillConfig {
            max_ingest_shards: 1,
            max_visibility_lag_ms: u64::MAX,
            ..config()
        },
    )
    .await
    .expect("create two-row fixture");
    LexicalWrite::index_documents(
        &index,
        cx,
        &[
            IndexableDocument::new("first", "shared first document"),
            IndexableDocument::new("second", "shared second document"),
        ],
    )
    .await
    .expect("stage two rows in one shard");
    LexicalWrite::commit(&index, cx)
        .await
        .expect("publish one two-row segment");
    let manifest = load_manifest_pair(directory)
        .expect("read two-row fixture manifest")
        .manifest;
    assert_eq!(manifest.segments.len(), 1);
    assert_eq!(manifest.segments[0].docid_lo, 0);
    assert_eq!(manifest.segments[0].doc_count, 2);
    manifest
}

#[test]
fn tombstone_only_change_revalidates_retained_bytes_before_publication() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("tombstone fixture directory");
        let initial = two_row_manifest(&cx, directory.path()).await;
        let paths = stage_segments(directory.path(), directory.path(), &initial);
        let path = paths.values().next().expect("retained segment path");
        let mut writer = KeeperWriter::open(&cx, directory.path(), DEFAULT_SCHEMA)
            .await
            .expect("admit intact retained segment");
        let retained = retained_manifest(&writer);
        let mut proposed = successor(&writer);
        proposed.segments[0]
            .tombstones
            .insert(0)
            .expect("tombstone a real row");
        let mut expected_entry = retained.segments[0].clone();
        expected_entry.tombstones = proposed.segments[0].tombstones.clone();
        assert_eq!(proposed.segments[0], expected_entry);
        assert_ne!(proposed.segments[0], retained.segments[0]);

        let original = corrupt_same_length(path);
        let before = directory_bytes(directory.path());
        let error = writer
            .publish(&cx, &proposed)
            .await
            .err()
            .expect("changed tombstones must not inherit preflight admission");
        assert!(
            matches!(error, KeeperError::SegmentOpen { .. }),
            "unexpected failure: {error}"
        );
        assert_eq!(directory_bytes(directory.path()), before);
        assert!(!writer.publication_awaits_reconciliation());
        assert_eq!(retained_manifest(&writer), retained);

        restore_segment(path, &original);
        writer
            .publish(&cx, &proposed)
            .await
            .expect("the same tombstone proposal succeeds after restoring its file");
        let installed = retained_manifest(&writer);
        assert_eq!(installed.generation, retained.generation + 1);
        assert_eq!(installed.segments, proposed.segments);
        assert_eq!(writer.snapshot().expect("installed authority").doc_count(), 1);
        assert_eq!(std::fs::read(path).expect("retained bytes"), original);
        drop(writer);
        let reopened = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)
            .expect("fresh open authenticates the file and binds its new tombstones");
        assert_eq!(reopened.loaded_manifest().manifest, installed);
        assert_eq!(reopened.doc_count(), 1);
        let reader = QuillSearchIndex::open(&cx, directory.path(), config())
            .await
            .expect("open fresh search reader");
        let hits = LexicalRead::search(&reader, &cx, "shared", 10)
            .await
            .expect("search after tombstone-only publication");
        assert_eq!(hits.len(), 1, "the deleted physical row must stay invisible");
    });
}

#[cfg(feature = "durability")]
#[test]
fn tombstone_only_change_requires_retained_segment_sidecar_and_retries_cleanly() {
    use std::sync::Arc;

    use frankensearch_durability::{DefaultSymbolCodec, DurabilityConfig, FileProtector};

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let donor_dir = tempfile::tempdir().expect("durable donor directory");
        let recipient_dir = tempfile::tempdir().expect("durable recipient directory");
        let mut first = two_row_manifest(&cx, donor_dir.path()).await;
        let protector = FileProtector::new(Arc::new(DefaultSymbolCodec), DurabilityConfig::default())
            .expect("real durability protector");
        let mut writer = KeeperWriter::create_durable(
            &cx,
            recipient_dir.path(),
            DEFAULT_SCHEMA,
            protector.clone(),
        )
        .await
        .expect("create protected recipient");
        let staged = stage_segments(donor_dir.path(), recipient_dir.path(), &first);
        let path = staged.values().next().expect("staged two-row segment");
        protector.protect_file(path).expect("protect the real FSLX file");
        first.generation = successor(&writer).generation;
        writer
            .publish(&cx, &first)
            .await
            .expect("admit the healthy segment and its sidecar");
        let retained = retained_manifest(&writer);
        let mut proposed = successor(&writer);
        proposed.segments[0]
            .tombstones
            .insert(0)
            .expect("change only retained tombstones");
        let mut expected_entry = retained.segments[0].clone();
        expected_entry.tombstones = proposed.segments[0].tombstones.clone();
        assert_eq!(proposed.segments[0], expected_entry);
        assert_ne!(proposed.segments[0], retained.segments[0]);
        let original = std::fs::read(path).expect("intact retained FSLX bytes");
        let sidecar = FileProtector::sidecar_path(path);
        assert!(
            protector
                .verify_file(path, &sidecar)
                .expect("verify healthy control sidecar")
                .healthy
        );
        let saved_sidecar = recipient_dir.path().join("saved-sidecar.fixture");
        std::fs::rename(&sidecar, &saved_sidecar).expect("hide, do not delete, the sidecar");
        let before = directory_bytes(recipient_dir.path());
        let error = writer
            .publish(&cx, &proposed)
            .await
            .err()
            .expect("a tombstone change still requires sidecar preflight");
        assert!(
            matches!(
                error,
                KeeperError::Durability {
                    operation: "preflight durable segment sidecar",
                    ..
                }
            ),
            "unexpected failure: {error}"
        );
        assert_eq!(directory_bytes(recipient_dir.path()), before);
        assert!(!writer.publication_awaits_reconciliation());
        assert_eq!(retained_manifest(&writer), retained);
        assert_eq!(writer.snapshot().expect("retained authority").doc_count(), 2);

        std::fs::rename(&saved_sidecar, &sidecar).expect("restore the exact healthy sidecar");
        writer
            .publish(&cx, &proposed)
            .await
            .expect("retry the same proposal with its restored durability evidence");
        let installed = retained_manifest(&writer);
        assert_eq!(installed.generation, retained.generation + 1);
        assert_eq!(installed.segments, proposed.segments);
        assert_eq!(std::fs::read(path).expect("unchanged FSLX bytes"), original);
        assert!(
            protector
                .verify_file(path, &sidecar)
                .expect("sidecar remains healthy after publication")
                .healthy
        );
        drop(writer);
        let reopened = KeeperSnapshot::open(recipient_dir.path(), DEFAULT_SCHEMA)
            .expect("fresh reopen after durable tombstone retry");
        assert_eq!(reopened.loaded_manifest().manifest, installed);
        assert_eq!(reopened.doc_count(), 1);
    });
}

#[test]
fn upsert_preserves_unchanged_segments_and_fresh_reopen_query_results() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("incremental upsert directory");
        let index = QuillIndex::create(
            &cx,
            directory.path(),
            QuillConfig {
                max_ingest_shards: 1,
                tier_fanout: 32,
                compaction_tombstone_density: 1.0,
                max_visibility_lag_ms: u64::MAX,
                ..config()
            },
        )
        .await
        .expect("create deterministic retained-segment fixture");
        // Keep a live anchor in the replaced segment, so this tests a changed
        // entry rather than depending on fully deleted segment retention.
        LexicalWrite::index_documents(
            &index,
            &cx,
            &[
                IndexableDocument::new("one", "shared antique"),
                IndexableDocument::new("anchor", "shared stable anchor"),
            ],
        )
        .await
        .expect("stage original and anchor in one segment");
        LexicalWrite::commit(&index, &cx)
            .await
            .expect("publish original and anchor");
        for (id, text) in [
            ("two", "shared blue"),
            ("three", "shared green"),
        ] {
            append_document(&index, &cx, directory.path(), id, text).await;
        }
        let before = load_manifest_pair(directory.path())
            .expect("read retained generation")
            .manifest;
        assert_eq!(before.segments.len(), 3);
        let after = append_document(
            &index,
            &cx,
            directory.path(),
            "one",
            "shared replacement",
        )
        .await;
        assert_eq!(after.generation, before.generation + 1);
        assert_eq!(after.segments.len(), 4);
        let mut unchanged = 0;
        let mut tombstone_changed = 0;
        for old in &before.segments {
            let carried = after
                .segments
                .iter()
                .find(|entry| entry.segment_id == old.segment_id)
                .expect("every old segment remains in the successor");
            if carried == old {
                unchanged += 1;
            } else {
                let mut expected_entry = old.clone();
                expected_entry.tombstones = carried.tombstones.clone();
                assert_eq!(carried, &expected_entry, "only tombstones may change");
                assert_ne!(carried.tombstones, old.tombstones);
                tombstone_changed += 1;
            }
        }
        assert_eq!(unchanged, 2);
        assert_eq!(tombstone_changed, 1);
        assert_eq!(
            after
                .segments
                .iter()
                .filter(|entry| {
                    !before
                        .segments
                        .iter()
                        .any(|old| old.segment_id == entry.segment_id)
                })
                .count(),
            1,
            "the upsert adds one segment beside the changed tombstone entry"
        );
        let expected = LexicalRead::search(&index, &cx, "shared", 10)
            .await
            .expect("search live upserted index");
        let mut ids: Vec<_> = expected.iter().map(|hit| hit.doc_id.as_str()).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec!["anchor", "one", "three", "two"]);
        assert!(
            LexicalRead::search(&index, &cx, "antique", 10)
                .await
                .expect("query replaced text before reopen")
                .is_empty()
        );
        drop(index);
        assert_fresh_search_matches(&cx, directory.path(), &expected).await;
        let reopened = QuillSearchIndex::open(&cx, directory.path(), config())
            .await
            .expect("open fresh upsert reader");
        assert!(
            LexicalRead::search(&reopened, &cx, "antique", 10)
                .await
                .expect("query replaced text after reopen")
                .is_empty()
        );
        let replacement = LexicalRead::search(&reopened, &cx, "replacement", 10)
            .await
            .expect("query replacement text after reopen");
        assert_eq!(replacement.len(), 1);
        assert_eq!(replacement[0].doc_id, "one");
    });
}

#[test]
fn retained_damage_never_returns_success_and_reconciliation_requires_fresh_bytes() {
    for missing in [false, true] {
        asupersync::test_utils::run_test_with_cx(move |cx| async move {
            let directory = tempfile::tempdir().expect("retained damage directory");
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
                .expect("intact control search");
            assert_eq!(expected.len(), 1);
            drop(index);
            let mut writer = KeeperWriter::open(&cx, directory.path(), DEFAULT_SCHEMA)
                .await
                .expect("open writer");
            let retained = retained_manifest(&writer);
            let proposed = successor(&writer);
            assert_eq!(proposed.segments, retained.segments);
            let paths = stage_segments(directory.path(), directory.path(), &retained);
            assert_eq!(paths.len(), 1);
            let path = paths.values().next().expect("retained segment path");
            let hidden = directory.path().join("hidden-retained-segment");
            let original = if missing {
                std::fs::rename(path, &hidden).expect("hide unchanged retained segment");
                None
            } else {
                Some(corrupt_same_length(path))
            };

            let error = writer
                .publish(&cx, &proposed)
                .await
                .err()
                .expect("retained damage must never produce a successful publication");
            assert!(
                matches!(&error, KeeperError::SegmentOpen { path: failed, .. } if failed == path),
                "unexpected failure: {error}"
            );
            let installed = load_manifest_pair(directory.path())
                .expect("read authority after refused publication")
                .manifest;
            if installed.generation == retained.generation {
                // Incumbent full preflight detects the fault before promotion.
                assert_eq!(installed, retained);
                assert_eq!(retained_manifest(&writer), retained);
                assert!(!writer.publication_awaits_reconciliation());
            } else {
                // Delta preflight may promote unchanged bindings, but a failed
                // mandatory fresh open must not expose the old snapshot as new.
                assert_eq!(installed.generation, proposed.generation);
                assert_eq!(installed.segments, proposed.segments);
                assert!(writer.publication_awaits_reconciliation());
                assert!(matches!(
                    writer.snapshot(),
                    Err(KeeperError::PublicationReconciliationRequired {
                        retained_generation,
                        proposed_generation,
                    }) if retained_generation == retained.generation
                        && proposed_generation == proposed.generation
                ));
                assert!(
                    writer.reconcile_publication(&cx).await.is_err(),
                    "reconciliation must not adopt still-damaged backing files"
                );
                assert!(writer.publication_awaits_reconciliation());
                assert!(writer.snapshot().is_err());
            }

            let before_open = directory_bytes(directory.path());
            assert!(matches!(
                KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA),
                Err(KeeperError::SegmentOpen { .. })
            ));
            assert!(
                QuillSearchIndex::open(&cx, directory.path(), config())
                    .await
                    .is_err()
            );
            assert_eq!(directory_bytes(directory.path()), before_open);
            if let Some(original) = original {
                restore_segment(path, &original);
            } else {
                std::fs::rename(&hidden, path).expect("restore unchanged retained segment");
            }
            let reconciled = writer
                .reconcile_publication(&cx)
                .await
                .expect("reconcile exact restored authority");
            assert_eq!(reconciled.loaded_manifest().manifest, installed);
            assert!(!writer.publication_awaits_reconciliation());
            drop(writer);
            assert_fresh_search_matches(&cx, directory.path(), &expected).await;
        });
    }
}
