//! Attempt-local reuse of fully validated immutable publication images.
//!
//! Preflight is not publication authority. A successor still opens the selected
//! MANIFEST and every selected FSLX, including the ordinary same-descriptor
//! streamed witness checks. Only an exact byte comparison with that fresh reader
//! permits reusing an OWNED backing and its pointer-bound metadata. A mapped
//! reader never becomes a reusable proof merely because its checks are warm.
//!
//! The budget limits retained raw images, not metadata, mapped pages or process
//! RSS. Successful readers retain these images for their lifetime, and several
//! pinned generations can retain several budgets. Ordinary publication opts out;
//! callers must qualify the CPU/memory tradeoff before enabling this path.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::{
    AuthenticatedFileWitness, KeeperError, KeeperSnapshot, ManifestSegment,
    ReadOnlyMappedFile, RecoveredSegment, RecoveredSegmentBacking, SchemaDescriptor,
    SegmentReader, fully_verified_file_witness, recovery_retryable, spawn_blocking,
    validate_segment_witnesses,
};

#[derive(Default)]
pub(super) struct PublicationValidationCache {
    remaining_bytes: usize,
    // Exact paths are only lookup keys. Neither a matching path nor a matching
    // MANIFEST witness is sufficient to reuse the stored proof.
    segments: BTreeMap<PathBuf, RecoveredSegment>,
}

impl PublicationValidationCache {
    pub(super) fn new(byte_budget: usize) -> Self {
        Self {
            remaining_bytes: byte_budget,
            segments: BTreeMap::new(),
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Finish ordinary preflight binding, retaining an immutable image only
    /// when its complete byte length fits the remaining attempt-local budget.
    pub(super) fn admit(
        &mut self,
        path: PathBuf,
        manifest: ManifestSegment,
        reader: SegmentReader<ReadOnlyMappedFile>,
        schema: SchemaDescriptor,
        authenticated_file_witness: AuthenticatedFileWitness,
    ) -> Result<(), KeeperError> {
        let byte_len = reader.source_bytes().len();
        let mut bytes = Vec::new();
        if byte_len > self.remaining_bytes || bytes.try_reserve_exact(byte_len).is_err() {
            // Cache admission is optional, but full segment admission is not.
            let _validated = RecoveredSegment::bind(
                path,
                manifest,
                reader,
                schema,
                authenticated_file_witness,
            )?;
            return Ok(());
        }
        bytes.extend_from_slice(reader.source_bytes());
        let owned = SegmentReader::from_owned(bytes, schema).map_err(|source| {
            KeeperError::SegmentOpen {
                path: path.clone(),
                source,
            }
        })?;
        // A copy made from a mapped file is NOT EncodedSegment and has no
        // producer-minted witness. Authenticate this exact copy afresh: the
        // underlying file might have changed during the copy, after the mapped
        // preflight's checksum gates were populated.
        let file_xxh3 = fully_verified_file_witness(&path, &owned)?;
        validate_segment_witnesses(&path, &manifest, &owned, || Ok(file_xxh3))?;
        #[cfg(test)]
        let authenticated_copy = AuthenticatedFileWitness::mint_after_full_prefix_validation(
            file_xxh3,
            Arc::new(super::AtomicU64::new(1)),
        );
        #[cfg(not(test))]
        let authenticated_copy =
            AuthenticatedFileWitness::mint_after_full_prefix_validation(file_xxh3);
        let recovered = RecoveredSegment::bind_backing(
            path.clone(),
            manifest,
            RecoveredSegmentBacking::ValidatedCopy(owned),
            schema,
            authenticated_copy,
        )?;
        // Charge only after the complete copied image and all binding metadata
        // have passed admission. Failure cannot leave a reusable partial entry.
        self.remaining_bytes -= byte_len;
        self.segments.insert(path, recovered);
        Ok(())
    }

    /// The caller MUST have freshly authenticated `reader` against `manifest`.
    /// A cache miss takes the ordinary full-binding path; no copied image can
    /// stand in for missing, truncated, corrupt or substituted durable bytes.
    pub(super) fn rebind_if_identical(
        &self,
        path: &Path,
        manifest: &ManifestSegment,
        reader: &SegmentReader<ReadOnlyMappedFile>,
        schema: SchemaDescriptor,
        authenticated_file_witness: &AuthenticatedFileWitness,
    ) -> Result<Option<RecoveredSegment>, KeeperError> {
        let Some(previous) = self.segments.get(path) else {
            return Ok(None);
        };
        if previous.reader.is_mapped()
            || previous.term_dictionary_metadata.schema() != schema
            || previous.reader.source_bytes() != reader.source_bytes()
        {
            return Ok(None);
        }
        // Crucially, use the original immutable allocation, not the newly
        // opened mapping with someone else's pointer-bound metadata. The fresh
        // witness comes from the actual post-publication descriptor. Existing
        // binding checks still validate the new ranges, tombstones, IDMAP/IDHASH,
        // whole-file witness, schema and TERMDICT address/length association.
        RecoveredSegment::bind_shared(
            path.to_path_buf(),
            manifest.clone(),
            Arc::clone(&previous.reader),
            Arc::clone(&previous.rank_pruning_cache),
            schema,
            authenticated_file_witness.clone(),
            Some(previous),
        )
        .map(Some)
    }
}

pub(super) async fn open_snapshot_after_preflight(
    directory: PathBuf,
    schema: SchemaDescriptor,
    cache: PublicationValidationCache,
) -> Result<KeeperSnapshot, KeeperError> {
    spawn_blocking(move || {
        // Retain the original read-only opener's one retry and classification.
        // Each attempt starts with a NEW authority and same-descriptor file read.
        match KeeperSnapshot::open_once_reusing_validation(&directory, schema, Some(&cache)) {
            Err(error) if recovery_retryable(&error) => {
                KeeperSnapshot::open_once_reusing_validation(&directory, schema, Some(&cache))
            }
            result => result,
        }
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    use frankensearch_core::{IndexableDocument, LexicalWrite};

    use crate::{DEFAULT_SCHEMA, QuillConfig, QuillIndex};
    use super::super::{
        AtomicOrdering, KeeperWriter, PublishIntent, WriterProtection,
        validate_proposed_manifest_segments_with_cache,
    };

    async fn fixture(cx: &asupersync::Cx, path: &Path) -> KeeperSnapshot {
        let index = QuillIndex::create(
            cx,
            path,
            QuillConfig {
                bulk_load_mode: true,
                deterministic_ingest: true,
                max_ingest_shards: 1,
                ..QuillConfig::default()
            },
        )
        .await
        .unwrap();
        for document in [
            IndexableDocument::new("one", "alpha needle").with_metadata("version", "old"),
            IndexableDocument::new("two", "beta needle").with_title("retained title"),
        ] {
            LexicalWrite::index_document(&index, cx, &document).await.unwrap();
        }
        index.finish_bulk_load(cx).await.unwrap();
        let snapshot = KeeperSnapshot::open(path, DEFAULT_SCHEMA).unwrap();
        assert!(!snapshot.segments.is_empty());
        drop(index);
        snapshot
    }

    fn preflight(
        path: &Path,
        snapshot: &KeeperSnapshot,
        budget: usize,
    ) -> PublicationValidationCache {
        validate_proposed_manifest_segments_with_cache(
            path,
            &snapshot.loaded.manifest,
            DEFAULT_SCHEMA,
            &WriterProtection::Disabled,
            budget,
        )
        .unwrap()
    }

    fn assert_same_documents(left: &KeeperSnapshot, right: &KeeperSnapshot) {
        assert_eq!(left.loaded, right.loaded);
        assert_eq!(left.segments.len(), right.segments.len());
        for (a, b) in left.segments.iter().zip(&right.segments) {
            assert_eq!(a.manifest, b.manifest);
            assert_eq!(a.reader.source_bytes(), b.reader.source_bytes());
            for id in a.manifest.docid_lo..a.manifest.docid_hi {
                let id = u32::try_from(id).unwrap();
                assert_eq!(a.materialize_document_id(id), b.materialize_document_id(id));
            }
        }
    }

    #[test]
    fn fresh_authority_reuses_only_the_original_immutable_allocation() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let baseline = fixture(&cx, dir.path()).await;
            let cache = preflight(dir.path(), &baseline, usize::MAX);
            let observed = KeeperSnapshot::open_once_reusing_validation(
                dir.path(), DEFAULT_SCHEMA, Some(&cache),
            )
            .unwrap();
            let reference = KeeperSnapshot::open(dir.path(), DEFAULT_SCHEMA).unwrap();
            assert_same_documents(&reference, &observed);
            for segment in &observed.segments {
                let cached = cache.segments.get(&segment.path).unwrap();
                assert!(Arc::ptr_eq(&segment.reader, &cached.reader));
                assert!(!segment.reader.is_mapped());
                assert_eq!(segment.authenticated_file_witness.full_prefix_hash_count(), 1);
                assert!(!Arc::ptr_eq(
                    &segment.authenticated_file_witness.full_prefix_hash_count,
                    &cached.authenticated_file_witness.full_prefix_hash_count,
                ));
                assert_eq!(
                    segment.term_dictionary_cache_counters.full_validations.load(AtomicOrdering::Relaxed),
                    1,
                );
                assert_eq!(
                    segment.term_dictionary_cache_counters.metadata_reuses.load(AtomicOrdering::Relaxed),
                    1,
                );
            }
            // The default independent opener still performs fresh mapped binding.
            assert!(reference.segments.iter().all(|segment| segment.reader.is_mapped()));
        });
    }

    #[test]
    fn raw_image_budget_is_respected_and_zero_keeps_the_existing_path() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let baseline = fixture(&cx, dir.path()).await;
            let smallest = baseline.segments.iter()
                .map(|segment| segment.reader.source_bytes().len()).min().unwrap();
            for budget in [0, smallest - 1, smallest, usize::MAX] {
                let cache = preflight(dir.path(), &baseline, budget);
                let retained: usize = cache.segments.values()
                    .map(|segment| segment.reader.source_bytes().len()).sum();
                assert!(retained <= budget);
                assert_eq!(cache.remaining_bytes, budget - retained);
                if budget < smallest { assert!(cache.is_empty()); }
                let reopened = KeeperSnapshot::open_once_reusing_validation(
                    dir.path(), DEFAULT_SCHEMA, Some(&cache),
                ).unwrap();
                assert_same_documents(&baseline, &reopened);
                for segment in &reopened.segments {
                    assert_eq!(segment.reader.is_mapped(), !cache.segments.contains_key(&segment.path));
                }
            }
        });
    }

    #[test]
    fn cached_images_never_mask_changed_prefix_trailer_or_length() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for mutation in 0..3 {
                let dir = tempfile::tempdir().unwrap();
                let baseline = fixture(&cx, dir.path()).await;
                let cache = preflight(dir.path(), &baseline, usize::MAX);
                let (path, cached) = cache.segments.first_key_value().unwrap();
                let original = cached.reader.source_bytes().to_vec();
                let mut changed = original.clone();
                match mutation {
                    0 => { let middle = changed.len() / 2; changed[middle] ^= 1; }
                    1 => { let end = changed.len() - 1; changed[end] ^= 1; }
                    _ => changed.push(0),
                }
                drop(baseline); // No mutable-file mapping survives the mutation.
                fs::write(path, &changed).unwrap();
                assert!(KeeperSnapshot::open_once_reusing_validation(
                    dir.path(), DEFAULT_SCHEMA, Some(&cache),
                ).is_err());
                // The proof is still valid for its owned bytes, but cannot serve
                // instead of the file that the current authority actually names.
                assert_eq!(cached.reader.source_bytes(), original);
                fs::write(path, &original).unwrap();
                assert!(KeeperSnapshot::open_once_reusing_validation(
                    dir.path(), DEFAULT_SCHEMA, Some(&cache),
                ).is_ok());
            }
        });
    }

    #[test]
    fn missing_selected_file_is_not_recovered_from_preflight_memory() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let baseline = fixture(&cx, dir.path()).await;
            let cache = preflight(dir.path(), &baseline, usize::MAX);
            let path = cache.segments.first_key_value().unwrap().0.clone();
            drop(baseline);
            let moved = path.with_extension("retained-not-selected");
            fs::rename(&path, &moved).unwrap();
            assert!(KeeperSnapshot::open_once_reusing_validation(
                dir.path(), DEFAULT_SCHEMA, Some(&cache),
            ).is_err());
            fs::rename(&moved, &path).unwrap();
            assert!(KeeperSnapshot::open_once_reusing_validation(
                dir.path(), DEFAULT_SCHEMA, Some(&cache),
            ).is_ok());
        });
    }

    #[test]
    fn a_new_authority_is_loaded_instead_of_the_preflight_manifest() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let baseline = fixture(&cx, dir.path()).await;
            let cache = preflight(dir.path(), &baseline, usize::MAX);
            let before = baseline.loaded.manifest.generation;
            drop(baseline);
            let writer = QuillIndex::open(&cx, dir.path(), QuillConfig::default())
                .await
                .unwrap();
            let replacement = IndexableDocument::new("one", "replacement needle")
                .with_metadata("version", "new");
            LexicalWrite::index_document(&writer, &cx, &replacement)
                .await
                .unwrap();
            writer.commit(&cx).await.unwrap();
            drop(writer);
            let observed = KeeperSnapshot::open_once_reusing_validation(
                dir.path(), DEFAULT_SCHEMA, Some(&cache),
            )
            .unwrap();
            let reference = KeeperSnapshot::open(dir.path(), DEFAULT_SCHEMA).unwrap();
            assert!(observed.loaded.manifest.generation > before);
            assert_same_documents(&reference, &observed);
            // New segments have no preflight entry and must bind freshly.
            // Retained segments can reuse bytes, but must adopt the ACTUAL
            // successor's tombstones and identity/range checks.
            assert!(observed.segments.iter().any(|segment| {
                !cache.segments.contains_key(&segment.path) && segment.reader.is_mapped()
            }));
        });
    }

    #[test]
    fn ordinary_and_opted_in_publication_adopt_the_same_complete_proposal() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let baseline = fixture(&cx, dir.path()).await;
            drop(baseline);
            let mut writer = KeeperWriter::open(&cx, dir.path(), DEFAULT_SCHEMA).await.unwrap();
            for budget in [0, usize::MAX] {
                let mut proposal = writer.snapshot.loaded.manifest.clone();
                proposal.generation += 1;
                let observed = writer.publish_with_intent_reusing_validation(
                    &cx, &proposal, PublishIntent::PreserveLiveDocuments, budget,
                ).await.unwrap();
                assert_eq!(observed.loaded.manifest, proposal);
                let reference = KeeperSnapshot::open(dir.path(), DEFAULT_SCHEMA).unwrap();
                assert_same_documents(&reference, observed);
                for segment in &observed.segments {
                    assert_eq!(segment.reader.is_mapped(), budget == 0);
                    assert_eq!(segment.authenticated_file_witness.full_prefix_hash_count(), 1);
                    assert_eq!(
                        segment.term_dictionary_cache_counters.metadata_reuses.load(AtomicOrdering::Relaxed),
                        u64::from(budget != 0),
                    );
                }
                assert!(writer.pending_publication.is_none());
            }
            let before = fs::read(dir.path().join("MANIFEST")).unwrap();
            let mut proposal = writer.snapshot.loaded.manifest.clone();
            proposal.generation += 1;
            cx.set_cancel_requested(true);
            assert!(writer.publish_with_intent_reusing_validation(
                &cx, &proposal, PublishIntent::PreserveLiveDocuments, usize::MAX,
            ).await.is_err());
            cx.set_cancel_requested(false);
            assert_eq!(fs::read(dir.path().join("MANIFEST")).unwrap(), before);
            assert!(writer.pending_publication.is_none());
        });
    }
}
