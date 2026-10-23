//! One immutable lexical object for composite generation publication.
//!
//! The v1 receipt hashed a framed stream but counted only its payload bytes,
//! so no physical object could satisfy its length and hash simultaneously.
//! V2 persists that stream explicitly and binds its complete raw byte image.
//! It is an artifact codec, not a selection authority: the caller supplies
//! the exact expected receipt and independently derived source checkpoint.

use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::AtomicU64;

use asupersync::Cx;
use frankensearch_core::generation::{
    ExactComponentReceiptV1, GenerationComponentRole, SourceCheckpointV1,
};
use sha2::{Digest, Sha256};

use super::{
    AuthenticatedFileWitness, KeeperError, KeeperSnapshot, LoadedManifest, MAX_MANIFEST_BYTES,
    Manifest, ManifestSource, RecoveredSegment, RecoveredSegmentBacking, canonical_segment_name,
    exact_snapshot_manifest_bytes, fully_verified_file_witness, validate_segment_witnesses,
};
use crate::schema::SchemaDescriptor;
use crate::segment::SegmentReader;

/// Exact snapshot-image domain and version. The remaining bytes are a
/// `manifest` frame followed by each MANIFEST segment ID and `segment` frame.
/// Frame tags have a little-endian u16 length and payloads a little-endian
/// u64 length. Segment IDs are little-endian u64. There are no trailing bytes.
pub const EXACT_LEXICAL_SNAPSHOT_IMAGE_DOMAIN: &[u8] =
    b"frankensearch.quill.exact-lexical-snapshot.v2";

/// Complete, bounded lexical bytes paired with their independently derived
/// document-set and checkpoint receipt.
#[derive(Clone, Debug)]
pub struct ExactLexicalSnapshotImage {
    bytes: Arc<[u8]>,
    receipt: ExactComponentReceiptV1,
}

impl ExactLexicalSnapshotImage {
    /// Exact immutable bytes to stage under the composite authority.
    #[must_use]
    pub fn bytes(&self) -> Arc<[u8]> {
        Arc::clone(&self.bytes)
    }

    /// Receipt whose length and SHA-256 cover every staged byte.
    #[must_use]
    pub const fn receipt(&self) -> &ExactComponentReceiptV1 {
        &self.receipt
    }
}

fn rejected(detail: &str) -> KeeperError {
    KeeperError::ExactSnapshotReceipt {
        detail: detail.to_owned(),
    }
}

fn checkpoint(cx: &Cx) -> Result<(), KeeperError> {
    if cx.checkpoint().is_err() || cx.is_cancel_requested() {
        return Err(KeeperError::SnapshotImageCancelled);
    }
    Ok(())
}

fn digest(cx: &Cx, bytes: &[u8]) -> Result<[u8; 32], KeeperError> {
    let mut hasher = Sha256::new();
    for chunk in bytes.chunks(1 << 20) {
        checkpoint(cx)?;
        hasher.update(chunk);
    }
    checkpoint(cx)?;
    Ok(hasher.finalize().into())
}

fn append(output: &mut Vec<u8>, bytes: &[u8], limit: usize) -> Result<(), KeeperError> {
    let length = output
        .len()
        .checked_add(bytes.len())
        .ok_or_else(|| rejected("image output length overflow"))?;
    if length > limit {
        return Err(rejected(
            "lexical source grew beyond the image allocation budget",
        ));
    }
    output.extend_from_slice(bytes);
    Ok(())
}

fn append_frame(
    output: &mut Vec<u8>,
    role: &[u8],
    bytes: &[u8],
    limit: usize,
) -> Result<(), KeeperError> {
    let role_len = u16::try_from(role.len()).map_err(|_| rejected("image role is too long"))?;
    let byte_len = u64::try_from(bytes.len()).map_err(|_| rejected("image frame is too long"))?;
    append(output, &role_len.to_le_bytes(), limit)?;
    append(output, role, limit)?;
    append(output, &byte_len.to_le_bytes(), limit)?;
    append(output, bytes, limit)?;
    Ok(())
}

struct Decoder<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl Decoder<'_> {
    fn take(&mut self, len: usize) -> Result<Range<usize>, KeeperError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| rejected("image range overflow"))?;
        let range = self.offset..end;
        if self.bytes.get(range.clone()).is_none() {
            return Err(rejected("truncated lexical snapshot image"));
        }
        self.offset = end;
        Ok(range)
    }

    fn u64(&mut self) -> Result<u64, KeeperError> {
        let range = self.take(8)?;
        let word = self.bytes[range]
            .try_into()
            .map_err(|_| rejected("invalid image u64"))?;
        Ok(u64::from_le_bytes(word))
    }

    fn frame(&mut self, role: &[u8], max_bytes: usize) -> Result<Range<usize>, KeeperError> {
        let range = self.take(2)?;
        let word = self.bytes[range]
            .try_into()
            .map_err(|_| rejected("invalid image u16"))?;
        let len = usize::from(u16::from_le_bytes(word));
        let tag = self.take(len)?;
        if &self.bytes[tag] != role {
            return Err(rejected("unexpected lexical snapshot frame role"));
        }
        let len =
            usize::try_from(self.u64()?).map_err(|_| rejected("image frame length overflow"))?;
        if len > max_bytes {
            return Err(rejected("lexical snapshot frame exceeds its byte budget"));
        }
        self.take(len)
    }
}

impl KeeperSnapshot {
    pub(crate) fn is_retained_image(&self) -> bool {
        self.retained_manifest.is_some()
    }

    pub(super) fn validate_exact_image_source(&self) -> Result<(), KeeperError> {
        if self.loaded.manifest.bulk_mode_in_progress() {
            return Err(rejected(
                "an unfinished bulk publication cannot become a sealed lexical image",
            ));
        }
        if !self.quarantined_segments.is_empty() {
            return Err(rejected(
                "a quarantined lexical snapshot cannot claim complete generation coverage",
            ));
        }
        if matches!(
            self.loaded.source,
            ManifestSource::PreviousAfterMissingCurrent
                | ManifestSource::PreviousAfterCorruptCurrent
        ) {
            return Err(rejected(
                "a recovered predecessor requires explicit repair before image publication",
            ));
        }
        Ok(())
    }

    /// Freeze one complete lexical snapshot into an independently searchable
    /// immutable artifact. The byte budget is checked before image allocation.
    ///
    /// A durable source must still name the exact selected MANIFEST; its mapped
    /// segments are copied into the image, then validated from that owned image.
    /// An in-memory or previously admitted image needs no filesystem access.
    /// No file, selection pointer, or source snapshot is modified.
    ///
    /// # Errors
    /// Rejects cancellation, incomplete/recovered/quarantined source state,
    /// a changed source, resource limits, or invalid MANIFEST/FSLX contents.
    pub fn export_exact_image(
        &self,
        cx: &Cx,
        source_checkpoint: SourceCheckpointV1,
        max_bytes: usize,
    ) -> Result<ExactLexicalSnapshotImage, KeeperError> {
        checkpoint(cx)?;
        self.validate_exact_image_source()?;
        let manifest = exact_snapshot_manifest_bytes(self)?;
        let mut expected_len = EXACT_LEXICAL_SNAPSHOT_IMAGE_DOMAIN
            .len()
            .checked_add(2 + 8 + 8)
            .and_then(|length| length.checked_add(manifest.len()))
            .ok_or_else(|| rejected("lexical image length overflow"))?;
        for segment in &self.segments {
            checkpoint(cx)?;
            expected_len = expected_len
                .checked_add(8 + 2 + 7 + 8)
                .and_then(|length| length.checked_add(segment.reader.source_bytes().len()))
                .ok_or_else(|| rejected("lexical image length overflow"))?;
        }
        if expected_len > max_bytes {
            return Err(rejected("lexical snapshot image exceeds its byte budget"));
        }
        // Enforce the physical budget before hashing or materializing live IDs.
        let receipt = self
            .exact_lexical_component_receipt_with_checkpoint(source_checkpoint, || {
                checkpoint(cx)
            })?;
        if usize::try_from(receipt.bytes.byte_len).ok() != Some(expected_len) {
            return Err(rejected(
                "lexical source changed before its image was captured",
            ));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(expected_len)
            .map_err(|_| rejected("cannot allocate lexical snapshot image"))?;
        append(
            &mut bytes,
            EXACT_LEXICAL_SNAPSHOT_IMAGE_DOMAIN,
            expected_len,
        )?;
        append_frame(&mut bytes, b"manifest", &manifest, expected_len)?;
        for segment in &self.segments {
            checkpoint(cx)?;
            append(
                &mut bytes,
                &segment.manifest.segment_id.to_le_bytes(),
                expected_len,
            )?;
            append_frame(
                &mut bytes,
                b"segment",
                segment.reader.source_bytes(),
                expected_len,
            )?;
        }
        if bytes.len() != expected_len || digest(cx, &bytes)? != receipt.bytes.sha256 {
            return Err(rejected(
                "lexical source changed while its immutable image was captured",
            ));
        }
        let bytes: Arc<[u8]> = Arc::from(bytes);
        // Copying a mapped source does not authenticate its current contents.
        // Admit the copied bytes with every normal FSLX and MANIFEST check.
        Self::from_exact_image(
            cx,
            Arc::clone(&bytes),
            self.schema,
            source_checkpoint,
            &receipt,
            max_bytes,
        )?;
        Ok(ExactLexicalSnapshotImage { bytes, receipt })
    }

    /// Admit an exact lexical image using only caller-retained immutable bytes.
    ///
    /// Every segment retains a range of the supplied `Arc`; no segment payload
    /// is copied or reopened by path. The independently obtained receipt binds
    /// byte identity, ordered live IDs, count and source checkpoint. MANIFEST
    /// references, schema, full-prefix and section checksums, live IDMAP and
    /// tombstone membership are validated before a reader is returned.
    ///
    /// The caller must acquire the image under its own generation authority.
    /// This method does not discover a generation, mutate an index, or infer
    /// provenance from its bytes. Cancellation brackets each bounded parser.
    ///
    /// # Errors
    /// Rejects cancellation, budget excess, wrong receipt/checkpoint/schema,
    /// malformed or reordered frames, incomplete bulk state, corrupt segments,
    /// document-set drift and trailing bytes. Failure retains no reader state.
    pub fn from_exact_image(
        cx: &Cx,
        bytes: Arc<[u8]>,
        schema: SchemaDescriptor,
        source_checkpoint: SourceCheckpointV1,
        expected: &ExactComponentReceiptV1,
        max_bytes: usize,
    ) -> Result<Self, KeeperError> {
        checkpoint(cx)?;
        expected
            .validate()
            .map_err(|_| rejected("invalid expected lexical component receipt"))?;
        if expected.role != GenerationComponentRole::Lexical
            || expected.source_checkpoint != source_checkpoint.to_bytes()
        {
            return Err(rejected(
                "lexical role or independently derived checkpoint mismatch",
            ));
        }
        if bytes.len() > max_bytes
            || u64::try_from(bytes.len()).ok() != Some(expected.bytes.byte_len)
        {
            return Err(rejected("lexical image length or byte budget mismatch"));
        }
        if digest(cx, &bytes)? != expected.bytes.sha256 {
            return Err(rejected(
                "lexical image raw SHA-256 differs from its expected receipt",
            ));
        }
        let mut decoder = Decoder {
            bytes: &bytes,
            offset: 0,
        };
        let domain = decoder.take(EXACT_LEXICAL_SNAPSHOT_IMAGE_DOMAIN.len())?;
        if &bytes[domain] != EXACT_LEXICAL_SNAPSHOT_IMAGE_DOMAIN {
            return Err(rejected(
                "unsupported lexical snapshot image domain or version",
            ));
        }
        let manifest_range = decoder.frame(b"manifest", MAX_MANIFEST_BYTES)?;
        let manifest = Manifest::from_bytes(&bytes[manifest_range.clone()])
            .map_err(|source| KeeperError::InvalidManifest { source })?;
        if manifest.bulk_mode_in_progress() {
            return Err(rejected("unfinished bulk publication in lexical image"));
        }
        let schema_id = schema
            .schema_id()
            .map_err(|source| KeeperError::InvalidSchema { source })?;
        if manifest.schema_id != schema_id {
            return Err(rejected(
                "lexical image does not match the independently selected schema",
            ));
        }
        let mut segments = Vec::new();
        segments
            .try_reserve_exact(manifest.segments.len())
            .map_err(|_| rejected("cannot allocate lexical segment inventory"))?;
        for entry in &manifest.segments {
            checkpoint(cx)?;
            if decoder.u64()? != entry.segment_id {
                return Err(rejected(
                    "lexical image segment order differs from its MANIFEST",
                ));
            }
            let range = decoder.frame(b"segment", max_bytes)?;
            let path = PathBuf::from("<retained-lexical-image>")
                .join(canonical_segment_name(entry.segment_id));
            let reader = SegmentReader::from_retained_range(Arc::clone(&bytes), range, schema)
                .map_err(|source| KeeperError::SegmentOpen {
                    path: path.clone(),
                    source,
                })?;
            let mut authenticated = None;
            validate_segment_witnesses(&path, entry, &reader, || {
                let witness = fully_verified_file_witness(&path, &reader)?;
                authenticated = Some(witness);
                Ok(witness)
            })?;
            let witness = authenticated
                .ok_or_else(|| rejected("segment admission omitted full-prefix validation"))?;
            #[cfg(test)]
            let authenticated = AuthenticatedFileWitness::mint_after_full_prefix_validation(
                witness,
                Arc::new(AtomicU64::new(1)),
            );
            #[cfg(not(test))]
            let authenticated =
                AuthenticatedFileWitness::mint_after_full_prefix_validation(witness);
            segments.push(RecoveredSegment::bind_backing(
                path,
                entry.clone(),
                RecoveredSegmentBacking::Retained(reader),
                schema,
                authenticated,
            )?);
        }
        if decoder.offset != bytes.len() {
            return Err(rejected(
                "trailing bytes after the complete lexical snapshot",
            ));
        }
        let mut snapshot = Self::from_parts(
            None,
            schema,
            LoadedManifest {
                manifest,
                source: ManifestSource::InMemory,
            },
            segments,
        )?;
        // Retain the complete original allocation even for an empty image,
        // whose MANIFEST is its only component and has no segment-held Arc.
        snapshot.retained_manifest = Some((Arc::clone(&bytes), manifest_range));
        if snapshot
            .exact_lexical_component_receipt_with_checkpoint(source_checkpoint, || checkpoint(cx))?
            != *expected
        {
            return Err(rejected(
                "lexical image document set, live count or checkpoint differs from its receipt",
            ));
        }
        checkpoint(cx)?;
        Ok(snapshot)
    }
}

#[cfg(test)]
mod tests;
