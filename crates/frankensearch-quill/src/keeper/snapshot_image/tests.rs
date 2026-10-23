use super::super::{MANIFEST_FLAG_BULK_MODE_IN_PROGRESS, QuarantinedSegment};
use super::*;
use crate::{DEFAULT_SCHEMA, FSFS_CHUNK_SCHEMA, QuillConfig, QuillIndex, QuillSnapshotIndex};
use frankensearch_core::generation::{CanonicalDocsetV1, CommitRange};
use frankensearch_core::{IndexableDocument, LexicalRead, SearchError};

const IMAGE_BUDGET: usize = 16 * 1024 * 1024;

fn source_checkpoint() -> SourceCheckpointV1 {
    SourceCheckpointV1::derive(&CommitRange { low: 1, high: 3 })
}

fn config() -> QuillConfig {
    QuillConfig {
        deterministic_ingest: true,
        ..QuillConfig::default()
    }
}

async fn populate(cx: &Cx, index: &QuillIndex) {
    index
        .index_documents(
            cx,
            &[
                IndexableDocument::new("one", "alpha stable first document")
                    .with_metadata("ordinal", "one"),
                IndexableDocument::new("two", "alpha removed document")
                    .with_metadata("ordinal", "two"),
            ],
        )
        .await
        .expect("first document batch");
    index.commit(cx).await.expect("first segment");
    index
        .index_document(
            cx,
            &IndexableDocument::new("three", "alpha third document")
                .with_metadata("ordinal", "three"),
        )
        .await
        .expect("second document batch");
    index.commit(cx).await.expect("second segment");
    assert!(
        index
            .delete_document(cx, "two")
            .await
            .expect("tombstone publication")
    );
}

async fn fixture(cx: &Cx) -> QuillIndex {
    let index = QuillIndex::in_memory(config()).expect("in-memory writer");
    populate(cx, &index).await;
    index
}

fn admit(cx: &Cx, image: &ExactLexicalSnapshotImage) -> KeeperSnapshot {
    KeeperSnapshot::from_exact_image(
        cx,
        image.bytes(),
        DEFAULT_SCHEMA,
        source_checkpoint(),
        image.receipt(),
        IMAGE_BUDGET,
    )
    .expect("admit exact image")
}

fn rebind_raw(bytes: &[u8], receipt: &ExactComponentReceiptV1) -> ExactComponentReceiptV1 {
    let mut receipt = receipt.clone();
    receipt.bytes.byte_len = u64::try_from(bytes.len()).expect("test byte length");
    receipt.bytes.sha256 = Sha256::digest(bytes).into();
    receipt
}

fn manifest_range(bytes: &[u8]) -> Range<usize> {
    Decoder {
        bytes,
        offset: EXACT_LEXICAL_SNAPSHOT_IMAGE_DOMAIN.len(),
    }
    .frame(b"manifest", IMAGE_BUDGET)
    .expect("fixture MANIFEST frame")
}

fn first_segment_range(bytes: &[u8]) -> Range<usize> {
    let mut decoder = Decoder {
        bytes,
        offset: manifest_range(bytes).end,
    };
    decoder.u64().expect("fixture segment ID");
    decoder
        .frame(b"segment", IMAGE_BUDGET)
        .expect("fixture segment frame")
}

#[test]
fn retained_image_roundtrip_preserves_real_queries_tombstones_and_deferred_metadata() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let writer = fixture(&cx).await;
        let snapshot = writer.snapshot().expect("published snapshot");
        assert_eq!(snapshot.segments().len(), 2);
        let expected = writer
            .search_paginated(&cx, "alpha", 10, 0, true)
            .expect("source query");
        let image = snapshot
            .export_exact_image(&cx, source_checkpoint(), IMAGE_BUDGET)
            .expect("image export");
        let raw = image.bytes();
        assert_eq!(
            image.receipt().bytes.byte_len,
            u64::try_from(raw.len()).expect("image length")
        );
        let raw_hash: [u8; 32] = Sha256::digest(&raw).into();
        assert_eq!(
            image.receipt().bytes.sha256,
            raw_hash,
            "the one physical object satisfies its receipt"
        );
        assert_eq!(image.receipt().live_document_count, 2);
        assert_eq!(
            image.receipt().docset_digest,
            CanonicalDocsetV1::from_ordered_live_documents(["one", "three"])
                .expect("live ordered IDs")
                .digest()
        );

        let admitted = admit(&cx, &image);
        let retained_range = raw.as_ptr_range();
        for segment in admitted.segments() {
            let segment_range = segment.reader.source_bytes().as_ptr_range();
            assert!(
                segment_range.start >= retained_range.start
                    && segment_range.end <= retained_range.end,
                "segment payload must share the exact supplied Arc allocation"
            );
        }
        let recaptured = admitted
            .export_exact_image(&cx, source_checkpoint(), IMAGE_BUDGET)
            .expect("image re-export");
        assert_eq!(recaptured.bytes().as_ref(), raw.as_ref());
        assert_eq!(recaptured.receipt(), image.receipt());
        let reader =
            QuillSnapshotIndex::new(&cx, admitted, config()).expect("production read facade");
        let page = reader
            .search_paginated(&cx, "alpha", 10, 0, true)
            .expect("retained query");
        assert_eq!(page.hits, expected.hits);
        assert_eq!(page.total_count, Some(2));
        assert_eq!(reader.doc_count().expect("retained count"), 2);
        assert_eq!(LexicalRead::doc_count(&reader).expect("trait count"), 2);
        assert!(
            reader
                .search_results(&cx, "removed", 10)
                .expect("tombstoned query")
                .is_empty()
        );

        let (mut candidates, context) = LexicalRead::search_candidates(&reader, &cx, "alpha", 10)
            .await
            .expect("real fusion candidates")
            .into_parts();
        assert!(context.is_some());
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.metadata.is_none())
        );
        LexicalRead::hydrate_candidates(&reader, &cx, context.as_ref(), &mut candidates)
            .await
            .expect("retained image metadata hydration");
        let one = candidates
            .iter()
            .find(|candidate| candidate.doc_id == "one")
            .expect("first live result");
        assert_eq!(
            one.metadata.as_deref(),
            Some(&serde_json::json!({"ordinal": "one"}))
        );
        assert!(candidates.iter().all(|candidate| candidate.doc_id != "two"));
    });
}

#[test]
fn retained_reader_survives_new_publication_and_source_directory_relocation() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().expect("fixture root");
        let source = directory.path().join("source");
        let writer = QuillIndex::create(&cx, &source, config())
            .await
            .expect("durable writer");
        populate(&cx, &writer).await;
        let expected = writer
            .search_paginated(&cx, "alpha", 10, 0, true)
            .expect("old source query");
        let image = writer
            .snapshot()
            .expect("source snapshot")
            .export_exact_image(&cx, source_checkpoint(), IMAGE_BUDGET)
            .expect("durable image export");
        writer
            .index_document(&cx, &IndexableDocument::new("new", "alpha new generation"))
            .await
            .expect("new generation document");
        writer
            .commit(&cx)
            .await
            .expect("new generation publication");
        assert_eq!(
            writer
                .search_paginated(&cx, "alpha", 10, 0, true)
                .expect("new source query")
                .total_count,
            Some(3)
        );
        drop(writer);
        std::fs::rename(&source, directory.path().join("relocated-source"))
            .expect("relocate source");
        assert!(!source.exists());

        let raw = image.bytes();
        let weak = Arc::downgrade(&raw);
        let reader =
            QuillSnapshotIndex::new(&cx, admit(&cx, &image), config()).expect("pathless reader");
        let retained_receipt = image.receipt().clone();
        drop(raw);
        drop(image);
        assert!(
            weak.upgrade().is_some(),
            "the reader must retain its admitted allocation"
        );
        let observed = reader
            .search_paginated(&cx, "alpha", 10, 0, true)
            .expect("query after relocation");
        assert_eq!(observed.hits, expected.hits);
        assert_eq!(observed.total_count, Some(2));
        assert_eq!(
            reader
                .snapshot()
                .expect("retained snapshot")
                .exact_lexical_component_receipt(source_checkpoint())
                .expect("receipt without source paths"),
            retained_receipt
        );
        assert!(
            LexicalRead::search(&reader, &cx, "generation", 10)
                .await
                .expect("old generation search")
                .is_empty()
        );
    });
}

#[test]
fn retained_image_rejects_independent_identity_drift_and_enforces_byte_budgets() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let writer = fixture(&cx).await;
        let snapshot = writer.snapshot().expect("fixture snapshot");
        let image = snapshot
            .export_exact_image(&cx, source_checkpoint(), IMAGE_BUDGET)
            .expect("fixture image");
        let parse = |receipt: &ExactComponentReceiptV1,
                     checkpoint: SourceCheckpointV1,
                     schema: SchemaDescriptor,
                     budget: usize| {
            KeeperSnapshot::from_exact_image(
                &cx,
                image.bytes(),
                schema,
                checkpoint,
                receipt,
                budget,
            )
        };
        assert!(
            parse(
                image.receipt(),
                source_checkpoint(),
                DEFAULT_SCHEMA,
                IMAGE_BUDGET
            )
            .is_ok()
        );
        let other_checkpoint = SourceCheckpointV1::derive(&CommitRange { low: 1, high: 4 });
        assert!(
            parse(
                image.receipt(),
                other_checkpoint,
                DEFAULT_SCHEMA,
                IMAGE_BUDGET
            )
            .is_err()
        );
        assert!(
            parse(
                image.receipt(),
                source_checkpoint(),
                FSFS_CHUNK_SCHEMA,
                IMAGE_BUDGET
            )
            .is_err()
        );
        assert!(
            parse(
                image.receipt(),
                source_checkpoint(),
                DEFAULT_SCHEMA,
                image.bytes().len() - 1
            )
            .is_err()
        );
        assert!(
            snapshot
                .export_exact_image(&cx, source_checkpoint(), image.bytes().len() - 1)
                .is_err()
        );

        let mut wrong_order = image.receipt().clone();
        wrong_order.docset_digest =
            CanonicalDocsetV1::from_ordered_live_documents(["three", "one"])
                .expect("same-count reordered docset")
                .digest();
        assert!(
            parse(
                &wrong_order,
                source_checkpoint(),
                DEFAULT_SCHEMA,
                IMAGE_BUDGET
            )
            .is_err()
        );
        let mut wrong_count = image.receipt().clone();
        wrong_count.live_document_count += 1;
        assert!(
            parse(
                &wrong_count,
                source_checkpoint(),
                DEFAULT_SCHEMA,
                IMAGE_BUDGET
            )
            .is_err()
        );
        let mut wrong_role = image.receipt().clone();
        wrong_role.role = GenerationComponentRole::Metadata;
        assert!(
            parse(
                &wrong_role,
                source_checkpoint(),
                DEFAULT_SCHEMA,
                IMAGE_BUDGET
            )
            .is_err()
        );
        let mut wrong_hash = image.receipt().clone();
        wrong_hash.bytes.sha256[0] ^= 1;
        assert!(
            parse(
                &wrong_hash,
                source_checkpoint(),
                DEFAULT_SCHEMA,
                IMAGE_BUDGET
            )
            .is_err()
        );
    });
}

#[test]
fn retained_image_rejects_inner_corruption_after_raw_receipt_is_rebound() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let writer = fixture(&cx).await;
        let image = writer
            .snapshot()
            .expect("fixture snapshot")
            .export_exact_image(&cx, source_checkpoint(), IMAGE_BUDGET)
            .expect("fixture image");
        let pristine = image.bytes();
        let manifest = manifest_range(&pristine);
        let segment = first_segment_range(&pristine);
        let mut mutations = Vec::new();

        let mut bytes = pristine.to_vec();
        bytes[EXACT_LEXICAL_SNAPSHOT_IMAGE_DOMAIN.len() - 1] = b'1';
        mutations.push(("unpersisted v1 domain", bytes));
        let mut bytes = pristine.to_vec();
        bytes[EXACT_LEXICAL_SNAPSHOT_IMAGE_DOMAIN.len() + 2] = b'X';
        mutations.push(("wrong manifest frame role", bytes));
        let mut bytes = pristine.to_vec();
        bytes[manifest.start - 8..manifest.start].copy_from_slice(&u64::MAX.to_le_bytes());
        mutations.push(("overflowing manifest frame length", bytes));
        let mut bytes = pristine.to_vec();
        bytes[manifest.end] ^= 1;
        mutations.push(("segment order differs from MANIFEST", bytes));
        let mut bytes = pristine.to_vec();
        bytes[segment.start - 8..segment.start].copy_from_slice(&u64::MAX.to_le_bytes());
        mutations.push(("overflowing segment frame length", bytes));
        let mut bytes = pristine.to_vec();
        bytes[segment.end - 13] ^= 1;
        mutations.push(("inner FSLX payload corruption", bytes));
        let mut bytes = pristine.to_vec();
        bytes[manifest.start] ^= 1;
        mutations.push(("inner MANIFEST corruption", bytes));
        let mut bytes = pristine.to_vec();
        bytes.pop();
        mutations.push(("truncated final frame", bytes));
        let mut bytes = pristine.to_vec();
        bytes.push(0);
        mutations.push(("trailing bytes", bytes));

        for (case, bytes) in mutations {
            // Rebinding raw identity is deliberate: these cases must reach the
            // owner parser instead of failing only the outer SHA comparison.
            let receipt = rebind_raw(&bytes, image.receipt());
            assert!(
                KeeperSnapshot::from_exact_image(
                    &cx,
                    Arc::from(bytes),
                    DEFAULT_SCHEMA,
                    source_checkpoint(),
                    &receipt,
                    IMAGE_BUDGET
                )
                .is_err(),
                "accepted {case}"
            );
        }

        let mut bulk_manifest =
            Manifest::from_bytes(&pristine[manifest.clone()]).expect("fixture manifest");
        bulk_manifest.flags |= MANIFEST_FLAG_BULK_MODE_IN_PROGRESS;
        let replacement = bulk_manifest
            .to_bytes()
            .expect("valid unfinished bulk manifest");
        assert_eq!(replacement.len(), manifest.len());
        let mut bytes = pristine.to_vec();
        bytes[manifest].copy_from_slice(&replacement);
        let receipt = rebind_raw(&bytes, image.receipt());
        assert!(
            KeeperSnapshot::from_exact_image(
                &cx,
                Arc::from(bytes),
                DEFAULT_SCHEMA,
                source_checkpoint(),
                &receipt,
                IMAGE_BUDGET
            )
            .is_err(),
            "bulk state must never become ready through an image"
        );
        assert!(
            KeeperSnapshot::from_exact_image(
                &cx,
                pristine,
                DEFAULT_SCHEMA,
                source_checkpoint(),
                image.receipt(),
                IMAGE_BUDGET
            )
            .is_ok(),
            "paired healthy control"
        );
    });
}

#[test]
fn image_export_refuses_incomplete_recovered_or_quarantined_sources() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let writer = fixture(&cx).await;
        let original = writer.snapshot().expect("source snapshot");
        for source in [
            ManifestSource::PreviousAfterMissingCurrent,
            ManifestSource::PreviousAfterCorruptCurrent,
        ] {
            let mut snapshot = (*original).clone();
            snapshot.loaded.source = source;
            assert!(
                snapshot
                    .export_exact_image(&cx, source_checkpoint(), IMAGE_BUDGET)
                    .is_err()
            );
        }
        let mut snapshot = (*original).clone();
        snapshot.loaded.manifest.flags |= MANIFEST_FLAG_BULK_MODE_IN_PROGRESS;
        assert!(
            snapshot
                .export_exact_image(&cx, source_checkpoint(), IMAGE_BUDGET)
                .is_err()
        );
        let mut snapshot = (*original).clone();
        snapshot.quarantined_segments.push(QuarantinedSegment {
            segment_id: 99,
            path: PathBuf::from("seg-0000000000000063.fslx.quarantine"),
            estimated_missing_docs: Some(1),
        });
        assert!(
            snapshot
                .export_exact_image(&cx, source_checkpoint(), IMAGE_BUDGET)
                .is_err()
        );
        assert!(
            QuillSnapshotIndex::new(&cx, (*original).clone(), config()).is_err(),
            "unadmitted source snapshots cannot bypass the retained-image boundary"
        );
        assert!(
            original
                .export_exact_image(&cx, source_checkpoint(), IMAGE_BUDGET)
                .is_ok()
        );
    });
}

#[test]
fn empty_retained_image_is_searchable_and_cancellation_is_typed() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let snapshot = KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("empty source");
        let image = snapshot
            .export_exact_image(&cx, source_checkpoint(), IMAGE_BUDGET)
            .expect("empty image");
        let reader =
            QuillSnapshotIndex::new(&cx, admit(&cx, &image), config()).expect("empty reader");
        assert!(
            reader
                .search_results(&cx, "alpha", 10)
                .expect("empty query")
                .is_empty()
        );
        assert_eq!(reader.doc_count().expect("empty count"), 0);
        cx.set_cancel_requested(true);
        assert!(matches!(
            snapshot.export_exact_image(&cx, source_checkpoint(), IMAGE_BUDGET),
            Err(KeeperError::SnapshotImageCancelled)
        ));
        assert!(matches!(
            KeeperSnapshot::from_exact_image(
                &cx,
                image.bytes(),
                DEFAULT_SCHEMA,
                source_checkpoint(),
                image.receipt(),
                IMAGE_BUDGET
            ),
            Err(KeeperError::SnapshotImageCancelled)
        ));
        assert!(matches!(
            LexicalRead::search(&reader, &cx, "alpha", 10).await,
            Err(SearchError::Cancelled { .. })
        ));
    });
}
