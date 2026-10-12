//! Publication validation contracts for gh#51 and size-bounded merges in gh#66.
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
use frankensearch_quill::keeper::MANIFEST_FLAG_BULK_MODE_IN_PROGRESS;
use frankensearch_quill::quiver::{StatsSection, aggregate_field_stats};
use frankensearch_quill::scribe::{
    ColumnarAccumulator, DOC_ORDS_PER_LEASE, FlushDocumentInput, FlushSegmentInput,
    IndexedFieldValue, StoredFieldValue, flush_accumulator,
};
use frankensearch_quill::{
    CURRENT_ENGINE_VERSION, CompactionPolicy, DEFAULT_SCHEMA, KeeperError, KeeperSnapshot,
    KeeperWriter, Manifest, ManifestFieldStats, ManifestSegment, QuillConfig, QuillIndex,
    QuillSearchIndex, QuillSearchResult, QuillSearchSnapshot, SectionKind, SegmentReader,
    SessionMergePolicy, TombstoneSet, load_manifest_pair,
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
            // Durable flushes retain the sealed FSLX file on disk. Only the
            // memory backend retains an owned encoded buffer for publication.
            assert_eq!(pending.pending_owned_segment_count(), 0);
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
        assert_eq!(
            writer.snapshot().expect("installed authority").doc_count(),
            1
        );
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
        assert_eq!(
            hits.len(),
            1,
            "the deleted physical row must stay invisible"
        );
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
        let protector =
            FileProtector::new(Arc::new(DefaultSymbolCodec), DurabilityConfig::default())
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
        protector
            .protect_file(path)
            .expect("protect the real FSLX file");
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
        assert_eq!(
            writer.snapshot().expect("retained authority").doc_count(),
            2
        );

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
        for (id, text) in [("two", "shared blue"), ("three", "shared green")] {
            append_document(&index, &cx, directory.path(), id, text).await;
        }
        let before = load_manifest_pair(directory.path())
            .expect("read retained generation")
            .manifest;
        assert_eq!(before.segments.len(), 3);
        let after =
            append_document(&index, &cx, directory.path(), "one", "shared replacement").await;
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

// The dense metadata layout makes the gaps expensive even though every real
// source file below has only two rows. A hull across 2^29 docids needs more
// than u32 can address in STOREDMETA; 2^31 also overflows DOCLEN. These fixtures
// exercise the shipping format limit without constructing a gigabyte buffer.
const METADATA_GAP: u64 = 1 << 29;
const GAP_QUERIES: [&str; 5] = [
    "shared",
    "\"amber rust\"",
    "amber OR cobalt",
    "title:boundary",
    "removed",
];

struct GapDocument {
    global_docid: u32,
    stored: [Option<Vec<u8>>; 5],
}

async fn controlled_gap_fixture(
    cx: &Cx,
    directory: &Path,
    starts: &[u64],
    bulk: bool,
) -> (Manifest, Vec<GapDocument>) {
    let mut writer = KeeperWriter::create(cx, directory, DEFAULT_SCHEMA)
        .await
        .expect("create controlled-gap writer");
    let mut manifest = writer
        .snapshot()
        .expect("fixture genesis")
        .next_manifest()
        .expect("fixture successor");
    let mut stats = Vec::new();
    let mut documents = Vec::new();
    for (source, &start) in starts.iter().enumerate() {
        let source_id = 0x6600 + u64::try_from(source).expect("small fixture source index");
        let lease_base = start / u64::from(DOC_ORDS_PER_LEASE) * u64::from(DOC_ORDS_PER_LEASE);
        let local_start = u32::try_from(start - lease_base).expect("local ordinal fits u32");
        let source_documents: Vec<_> = (0..2)
            .map(|row| {
                let global_docid = u32::try_from(start + row).expect("global docid fits u32");
                let body = if global_docid == 1 {
                    "shared removed"
                } else if row == 0 {
                    "shared amber rust amber"
                } else {
                    "shared cobalt python"
                };
                GapDocument {
                    global_docid,
                    stored: [
                        Some(format!("gap-{global_docid}").into_bytes()),
                        Some(body.as_bytes().to_vec()),
                        (row == 0).then(|| {
                            if source % 2 == 0 {
                                format!("boundary {global_docid}").into_bytes()
                            } else {
                                Vec::new()
                            }
                        }),
                        (row == 0).then(|| {
                            serde_json::to_vec(&serde_json::json!({
                                "source": source,
                                "tag": "boundary",
                            }))
                            .expect("encode fixture metadata")
                        }),
                        Some(u64::from(global_docid).to_le_bytes().to_vec()),
                    ],
                }
            })
            .collect();
        let mut accumulator =
            ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("default fixture accumulator");
        let mut identities = Vec::new();
        for (row, document) in source_documents.iter().enumerate() {
            let indexed: Vec<_> = document.stored[..3]
                .iter()
                .enumerate()
                .filter_map(|(field, value)| {
                    value.as_deref().map(|bytes| {
                        IndexedFieldValue::new(
                            u16::try_from(field).expect("fixture field ordinal"),
                            std::str::from_utf8(bytes).expect("fixture text is UTF-8"),
                        )
                    })
                })
                .collect();
            let stored: Vec<_> = document.stored[3..]
                .iter()
                .enumerate()
                .filter_map(|(field, value)| {
                    value.as_deref().map(|bytes| {
                        StoredFieldValue::new(
                            u16::try_from(field + 3).expect("fixture stored ordinal"),
                            bytes,
                        )
                    })
                })
                .collect();
            let ordinal = local_start + u32::try_from(row).expect("fixture row ordinal");
            accumulator
                .add_document_with_stored(ordinal, &indexed, &stored)
                .expect("accumulate tiny source document");
            identities.push(FlushDocumentInput::from_canonical_content(
                ordinal,
                std::str::from_utf8(document.stored[0].as_deref().expect("stored identity"))
                    .expect("identity is UTF-8"),
                document.stored[1].as_deref().expect("stored content"),
            ));
        }
        let encoded = flush_accumulator(
            &accumulator,
            FlushSegmentInput {
                segment_id: source_id,
                lease_docid_base: lease_base,
                created_unix_s: 0,
                engine_version: CURRENT_ENGINE_VERSION,
                documents: &identities,
            },
        )
        .expect("seal two-row source in the existing FSLX format");
        assert!(encoded.file_len() < 16 * 1024, "source must stay tiny");
        let reader = SegmentReader::from_bytes(encoded.as_bytes(), DEFAULT_SCHEMA)
            .expect("read source format");
        stats.push(
            StatsSection::parse(
                reader
                    .section(SectionKind::STATS)
                    .expect("source STATS checksum")
                    .expect("source STATS present"),
                &[0, 1, 2],
                2,
            )
            .expect("source STATS decode"),
        );
        let mut tombstones = TombstoneSet::new();
        if start == 0 {
            tombstones.insert(1).expect("tombstone one physical row");
        }
        manifest.segments.push(ManifestSegment {
            segment_id: source_id,
            seal_seq: u64::try_from(source).expect("fixture seal sequence") + 1,
            file_len: encoded.file_len(),
            file_xxh3: encoded.file_xxh3(),
            docid_lo: encoded.header().docid_lo,
            docid_hi: encoded.header().docid_hi,
            doc_count: encoded.header().doc_count,
            tombstones,
        });
        writer
            .publish_segment(
                cx,
                encoded
                    .write_temp(directory)
                    .expect("sync tiny source segment"),
            )
            .await
            .expect("install tiny source segment");
        documents.extend(source_documents);
    }
    manifest.docid_high_watermark = starts.last().expect("nonempty fixture") + 2;
    manifest.field_stats = aggregate_field_stats(stats.iter())
        .expect("aggregate real source statistics")
        .into_iter()
        .map(|field| ManifestFieldStats {
            field_ord: field.field_ord,
            total_tokens: field.total_tokens,
            doc_count: u32::try_from(field.doc_count).expect("small physical count"),
        })
        .collect();
    if bulk {
        manifest.flags |= MANIFEST_FLAG_BULK_MODE_IN_PROGRESS;
    }
    writer
        .publish(cx, &manifest)
        .await
        .expect("publish real controlled-gap fixture");
    (retained_manifest(&writer), documents)
}

fn gap_queries(index: &QuillIndex, cx: &Cx) -> Vec<QuillSearchResult> {
    GAP_QUERIES
        .iter()
        .map(|query| {
            index
                .search_paginated(cx, query, 20, 0, true)
                .expect("query controlled-gap index")
        })
        .collect()
}

fn assert_gap_stored(snapshot: &QuillSearchSnapshot, documents: &[GapDocument]) {
    assert_gap_stored_values(documents, |field, docid| {
        snapshot
            .stored_field_value(field, docid)
            .expect("read original stored column bytes")
    });
}

fn assert_gap_stored_values(
    documents: &[GapDocument],
    mut read: impl FnMut(u16, u32) -> Option<Vec<u8>>,
) {
    for document in documents {
        for (field, expected) in document.stored.iter().enumerate() {
            let actual = read(
                u16::try_from(field).expect("fixture field ordinal"),
                document.global_docid,
            );
            assert_eq!(
                actual.as_deref(),
                if document.global_docid == 1 {
                    None
                } else {
                    expected.as_deref()
                },
                "stored field {field} at global docid {}",
                document.global_docid,
            );
        }
    }
}

async fn assert_gap_reopened(
    cx: &Cx,
    directory: &Path,
    expected: &[QuillSearchResult],
    documents: &[GapDocument],
) {
    let reader = QuillSearchIndex::open(cx, directory, config())
        .await
        .expect("reopen unchanged FSLX format after bounded maintenance");
    for (query, expected) in GAP_QUERIES.iter().zip(expected) {
        assert_eq!(
            reader
                .search_paginated(cx, query, 20, 0, true)
                .expect("query fresh bounded-merge reader"),
            *expected,
            "reopened query {query} preserves scores, order, identities, and counts",
        );
    }
    assert_gap_stored_values(documents, |field, docid| {
        reader
            .stored_field_value(field, docid)
            .expect("cold reader preserves every stored value")
    });
}

#[test]
fn unrepresentable_concat_is_rejected_before_allocation_or_publication() {
    for (gap, section) in [(METADATA_GAP, "STOREDMETA"), (1 << 31, "DOCLEN")] {
        asupersync::test_utils::run_test_with_cx(move |cx| async move {
            let directory = tempfile::tempdir().expect("overflow refusal directory");
            let (manifest, documents) =
                controlled_gap_fixture(&cx, directory.path(), &[0, gap], false).await;
            let index = QuillIndex::open(&cx, directory.path(), config())
                .await
                .expect("open tiny distant source files");
            let before = directory_bytes(directory.path());
            let expected = gap_queries(&index, &cx);
            let sources: Vec<_> = manifest.segments.iter().map(|s| s.segment_id).collect();
            let error = index
                .concat_merge(&cx, &sources, 0x66ff, 0)
                .await
                .err()
                .expect("unrepresentable dense hull must be refused");
            assert!(
                error.to_string().contains(section),
                "expected {section} size diagnosis, got {error}",
            );
            assert_eq!(directory_bytes(directory.path()), before);
            assert_eq!(gap_queries(&index, &cx), expected);
            assert_eq!(
                index
                    .snapshot()
                    .expect("retained authority")
                    .loaded_manifest()
                    .manifest,
                manifest,
            );
            assert_gap_stored(&index.search_snapshot().expect("retained view"), &documents);
            assert_gap_reopened(&cx, directory.path(), &expected, &documents).await;
        });
    }
}

#[test]
fn bounded_session_merges_skip_unrepresentable_hulls_and_preserve_all_live_rows() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("bounded session directory");
        let starts = [0, 2, METADATA_GAP, METADATA_GAP + 2, 2 * METADATA_GAP];
        let (initial, documents) =
            controlled_gap_fixture(&cx, directory.path(), &starts, false).await;
        let index = QuillIndex::open(&cx, directory.path(), config())
            .await
            .expect("open session maintenance fixture");
        let expected = gap_queries(&index, &cx);
        assert_eq!(expected[0].total_count, Some(9));
        assert!(
            expected[4].hits.is_empty(),
            "tombstoned row stays invisible"
        );
        let held = index.search_snapshot().expect("pin source search snapshot");
        let policy = SessionMergePolicy {
            fanout: 2,
            max_input_docs: 100,
            max_input_bytes: u64::MAX,
        };
        for output in [0x66f0, 0x66f1] {
            let merged = index
                .merge_sessions_once(&cx, policy, output, 0)
                .await
                .expect("skip impossible hull and merge a compatible adjacent pair")
                .expect("an eligible pair remains");
            assert_eq!(merged.doc_count(), 9);
            assert_eq!(
                merged.loaded_manifest().manifest.field_stats,
                initial.field_stats
            );
            assert_eq!(gap_queries(&index, &cx), expected);
            assert_gap_stored(&held, &documents);
            assert_gap_stored(&index.search_snapshot().expect("merged view"), &documents);
            assert_gap_reopened(&cx, directory.path(), &expected, &documents).await;
        }
        let finished = index.snapshot().expect("bounded final snapshot");
        assert_eq!(finished.segments().len(), 3);
        assert_eq!(
            finished
                .segments()
                .iter()
                .map(|segment| (segment.manifest().docid_lo, segment.manifest().docid_hi))
                .collect::<Vec<_>>(),
            [
                (0, 4),
                (METADATA_GAP, METADATA_GAP + 4),
                (2 * METADATA_GAP, 2 * METADATA_GAP + 2)
            ],
        );
        let before_noop = directory_bytes(directory.path());
        assert!(
            index
                .merge_sessions_once(&cx, policy, 0x66f2, 0)
                .await
                .expect("capacity-limited maintenance completes normally")
                .is_none()
        );
        assert_eq!(directory_bytes(directory.path()), before_noop);

        let report = index
            .compact(&cx, CompactionPolicy::default())
            .await
            .expect("fold the tombstone in its bounded merged segment");
        assert_eq!(report.compacted_segments, 1);
        assert_eq!(report.dropped_documents, 1);
        assert_eq!(index.doc_count().expect("compacted live count"), 9);
        for (query, previous) in GAP_QUERIES.iter().zip(&expected) {
            let mut previous_docids: Vec<_> =
                previous.hits.iter().map(|hit| hit.global_docid).collect();
            previous_docids.sort_unstable();
            assert_eq!(
                index
                    .collect_docids(&cx, query)
                    .expect("compacted document set"),
                previous_docids,
                "compaction keeps matching documents for {query}",
            );
        }
        assert_gap_stored(&held, &documents);
        assert_gap_stored(
            &index.search_snapshot().expect("compacted view"),
            &documents,
        );
        assert_gap_reopened(&cx, directory.path(), &gap_queries(&index, &cx), &documents).await;
    });
}

#[test]
fn automatic_tiers_and_resumed_bulk_finish_leave_multiple_representable_segments() {
    for bulk in [false, true] {
        asupersync::test_utils::run_test_with_cx(move |cx| async move {
            let directory = tempfile::tempdir().expect("bounded automatic merge directory");
            let starts = [0, 2, METADATA_GAP, METADATA_GAP + 2, 2 * METADATA_GAP];
            let (initial, documents) =
                controlled_gap_fixture(&cx, directory.path(), &starts, bulk).await;
            let index = QuillIndex::open(
                &cx,
                directory.path(),
                QuillConfig {
                    tier_fanout: 2,
                    merge_max_hole_ratio: 1.0,
                    bulk_load_mode: bulk,
                    ..config()
                },
            )
            .await
            .expect("open interrupted bulk or ordinary tier fixture");
            let expected = gap_queries(&index, &cx);
            let held = index.search_snapshot().expect("pin source view");
            if bulk {
                index
                    .finish_bulk_load(&cx)
                    .await
                    .expect("finish size-bounded bulk tail");
            } else {
                index
                    .commit(&cx)
                    .await
                    .expect("apply size-bounded automatic tiers");
            }
            let snapshot = index.snapshot().expect("finished bounded maintenance");
            let manifest = &snapshot.loaded_manifest().manifest;
            assert_eq!(manifest.segments.len(), 3, "bulk={bulk}");
            assert_eq!(manifest.field_stats, initial.field_stats);
            assert!(!manifest.bulk_mode_in_progress());
            assert_eq!(gap_queries(&index, &cx), expected);
            assert_gap_stored(&held, &documents);
            assert_gap_stored(&index.search_snapshot().expect("new view"), &documents);
            assert_gap_reopened(&cx, directory.path(), &expected, &documents).await;
            let before_noop = directory_bytes(directory.path());
            index
                .commit(&cx)
                .await
                .expect("no remaining representable tier run");
            assert_eq!(directory_bytes(directory.path()), before_noop);
        });
    }
}

#[test]
fn bounded_merge_output_stays_unreachable_until_manifest_publication_and_retries() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("bounded merge publication directory");
        // The first candidate crosses the overflow gap; the second is legal.
        let (initial, documents) = controlled_gap_fixture(
            &cx,
            directory.path(),
            &[0, METADATA_GAP, METADATA_GAP + 2],
            false,
        )
        .await;
        let index = QuillIndex::open(&cx, directory.path(), config())
            .await
            .expect("open publication fixture");
        let expected = gap_queries(&index, &cx);
        let held = index.search_snapshot().expect("pin pre-publication view");
        let blocked_temp = directory
            .path()
            .join(format!(".tmp-manifest-{}", initial.generation + 1));
        let blocker = b"foreign proposal must not be overwritten";
        std::fs::write(&blocked_temp, blocker).expect("stage deterministic publication failure");
        let policy = SessionMergePolicy {
            fanout: 2,
            max_input_docs: 100,
            max_input_bytes: u64::MAX,
        };
        let error = index
            .merge_sessions_once(&cx, policy, 0x66ee, 0)
            .await
            .err()
            .expect("fail after output installation, before MANIFEST promotion");
        assert!(
            error.to_string().contains("temp"),
            "unexpected publication error: {error}"
        );
        let output_path = directory.path().join("seg-00000000000066ee.fslx");
        let staged_output = std::fs::read(&output_path).expect("synced unreferenced merge output");
        assert_eq!(
            std::fs::read(&blocked_temp).expect("retained foreign temp"),
            blocker
        );
        assert_eq!(
            load_manifest_pair(directory.path())
                .expect("old authority")
                .manifest,
            initial
        );
        assert_gap_stored(&held, &documents);
        assert_gap_reopened(&cx, directory.path(), &expected, &documents).await;
        for segment in &initial.segments {
            assert!(
                directory
                    .path()
                    .join(format!("seg-{:016x}.fslx", segment.segment_id))
                    .exists()
            );
        }

        std::fs::rename(
            &blocked_temp,
            directory
                .path()
                .join("preserved-conflicting-proposal.fixture"),
        )
        .expect("preserve the conflicting temp outside the publication namespace");
        index
            .reconcile_publication(&cx)
            .await
            .expect("reconcile authoritative old generation");
        index
            .merge_sessions_once(&cx, policy, 0x66ee, 0)
            .await
            .expect("retry adopts the exact already-installed output")
            .expect("compatible pair still merges");
        assert_eq!(
            std::fs::read(&output_path).expect("reused output"),
            staged_output
        );
        assert_eq!(gap_queries(&index, &cx), expected);
        assert_gap_stored(&held, &documents);
        assert_gap_stored(
            &index.search_snapshot().expect("published merge view"),
            &documents,
        );
        assert_gap_reopened(&cx, directory.path(), &expected, &documents).await;
        assert_eq!(
            Manifest::from_bytes(
                &std::fs::read(directory.path().join("MANIFEST.prev")).expect("previous slot")
            )
            .expect("decode retained predecessor"),
            initial,
        );
    });
}

#[test]
fn interrupted_bounded_bulk_finish_resumes_after_a_group_has_been_published() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("partially finished bulk directory");
        let starts = [0, 2, METADATA_GAP, METADATA_GAP + 2, 2 * METADATA_GAP];
        let (initial, documents) =
            controlled_gap_fixture(&cx, directory.path(), &starts, true).await;
        let bulk_config = QuillConfig {
            tier_fanout: 2,
            merge_max_hole_ratio: 1.0,
            bulk_load_mode: true,
            ..config()
        };
        let index = QuillIndex::open(&cx, directory.path(), bulk_config.clone())
            .await
            .expect("open interrupted bulk source generation");
        let expected = gap_queries(&index, &cx);
        let held = index
            .search_snapshot()
            .expect("pin original bulk generation");
        // The first output publishes normally. Refuse the second group's
        // manifest so recovery has to retain that completed first group.
        let blocked_temp = directory
            .path()
            .join(format!(".tmp-manifest-{}", initial.generation + 2));
        std::fs::write(&blocked_temp, b"conflicting second bulk publication")
            .expect("stage second-group publication fault");
        let error = index
            .finish_bulk_load(&cx)
            .await
            .err()
            .expect("interrupt after the first bounded output became authoritative");
        assert!(
            error.to_string().contains("temp"),
            "unexpected partial-bulk failure: {error}",
        );
        let partial = load_manifest_pair(directory.path())
            .expect("read partially finished bulk authority")
            .manifest;
        assert_eq!(partial.generation, initial.generation + 1);
        assert_eq!(partial.segments.len(), 4);
        assert!(partial.bulk_mode_in_progress());
        assert_eq!(partial.field_stats, initial.field_stats);
        let completed_group = partial.segments[0].clone();
        assert_eq!((completed_group.docid_lo, completed_group.docid_hi), (0, 4));
        assert_gap_stored(&held, &documents);
        assert_gap_reopened(&cx, directory.path(), &expected, &documents).await;

        std::fs::rename(
            &blocked_temp,
            directory
                .path()
                .join("preserved-bulk-publication-conflict.fixture"),
        )
        .expect("preserve the fault fixture before reopening");
        drop(index);
        let resumed = QuillIndex::open(&cx, directory.path(), bulk_config)
            .await
            .expect("recover authoritative partial progress under a new writer");
        let finished = resumed
            .finish_bulk_load(&cx)
            .await
            .expect("resume the remaining compatible bulk groups");
        let manifest = &finished.loaded_manifest().manifest;
        assert_eq!(manifest.segments.len(), 3);
        assert!(manifest.segments.contains(&completed_group));
        assert!(!manifest.bulk_mode_in_progress());
        assert_eq!(finished.doc_count(), 9);
        assert_eq!(manifest.field_stats, initial.field_stats);
        assert_eq!(gap_queries(&resumed, &cx), expected);
        assert_gap_stored(&held, &documents);
        assert_gap_stored(
            &resumed.search_snapshot().expect("resumed view"),
            &documents,
        );
        assert_gap_reopened(&cx, directory.path(), &expected, &documents).await;
    });
}
