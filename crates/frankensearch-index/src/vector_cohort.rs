//! Complete fast/quality FSVI images retained as one authority Vector component.
//!
//! This container preserves each tier's existing FSVI v2 format and independent
//! identity admission. The caller supplies both trusted bindings and the source
//! checkpoint; neither identity is inferred from the image or the other tier.
//! Partial coverage and sharded layouts are not represented by this format.
//!
//! # FSVC version 1 wire contract
//!
//! Integers are little-endian. The fixed header is exactly 192 bytes:
//!
//! | Offset | Length | Field |
//! |---:|---:|---|
//! | 0 | 4 | `FSVC` magic |
//! | 4 | 2 | Version, exactly 1 |
//! | 6 | 2 | Header length, exactly 192 |
//! | 8 | 32 | Source checkpoint derived by the caller |
//! | 40 | 32 | Canonical ordered live-document-set digest |
//! | 72 | 8 | Common live-document count |
//! | 80 | 56 | Fast entry |
//! | 136 | 56 | Quality entry |
//!
//! Each entry is a one-byte role (fast=1, quality=2), seven zero bytes,
//! an eight-byte offset, an eight-byte length, and a 32-byte SHA-256. Fast starts
//! at 192; quality starts immediately after fast. Padding, additional entries,
//! unknown versions, reserved bits and trailing bytes are rejected. Checked
//! offsets and lengths precede payload allocation.
//!
//! Both independently admitted owners must agree on the full artifact generation,
//! ordered live-document set, document-ID semantics and semantic/control space
//! kind. Their dimensions, producers, quantization and physical tombstone rows
//! may differ. The aggregate Vector receipt authenticates every container byte
//! and the shared document set/source cut. Per-tier metadata must describe the
//! embedded FSVI images' own lengths, hashes, dimensions and physical counts.

use std::fmt;
use std::sync::Arc;

use frankensearch_core::generation::{
    ExactComponentReceiptV1, GenerationComponentReceiptV1, GenerationComponentRole,
    SourceCheckpointV1,
};
use frankensearch_core::{Cx, SearchError, SearchResult};
use sha2::{Digest as _, Sha256};

use crate::exact_component_adapters::vector_component_receipt;
use crate::{FsviV2IdentityBinding, ValidatedFsviBytes};

/// Encoded overhead of one complete FSVC v1 fast/quality vector cohort.
pub const VECTOR_COHORT_HEADER_BYTES_V1: usize = 192;
const HEADER_BYTES_WIRE: u16 = 192;
const MAGIC: &[u8; 4] = b"FSVC";
const VERSION: u16 = 1;
const FAST_ENTRY: usize = 80;
const QUALITY_ENTRY: usize = 136;
const CHUNK_BYTES: usize = 1024 * 1024;

/// Immutable complete vector cohort with two independently admitted owners.
///
/// Admission is synchronous and checks cancellation between bounded chunks and
/// validation stages. It opens no paths and creates no runtime or tasks. The
/// encoded-image limit does not bound process RSS: parsed owners retain separate
/// immutable payload allocations alongside the aggregate image.
pub struct ExactVectorCohortV1 {
    bytes: Arc<[u8]>,
    tiers: AdmittedTiers,
    receipt: ExactComponentReceiptV1,
}

impl fmt::Debug for ExactVectorCohortV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExactVectorCohortV1")
            .field("receipt", &self.receipt)
            .finish_non_exhaustive()
    }
}

struct AdmittedTiers {
    fast: Arc<ValidatedFsviBytes>,
    quality: Arc<ValidatedFsviBytes>,
    fast_receipt: ExactComponentReceiptV1,
    quality_receipt: ExactComponentReceiptV1,
}

impl ExactVectorCohortV1 {
    /// Admit and pack two independently selected complete FSVI v2 images.
    ///
    /// The shared coverage is derived from actual live rows. Empty complete
    /// cohorts are representable; consumers may impose a stricter readiness
    /// policy. Tombstones need not have identical physical positions or counts.
    ///
    /// # Errors
    /// Refuses invalid images, producer/generation/coverage disagreement,
    /// incompatible document-ID semantics or space kinds, image-budget overflow,
    /// allocation failure and cancellation.
    pub fn build(
        cx: &Cx,
        fast: (&FsviV2IdentityBinding, Arc<[u8]>),
        quality: (&FsviV2IdentityBinding, Arc<[u8]>),
        source: SourceCheckpointV1,
        max_image_bytes: usize,
    ) -> SearchResult<Self> {
        checkpoint(cx)?;
        let total = checked_total(fast.1.len(), quality.1.len(), max_image_bytes)?;
        let fast_bytes = Arc::clone(&fast.1);
        let quality_bytes = Arc::clone(&quality.1);
        let tiers = admit_tiers(cx, fast, quality, source)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(total)
            .map_err(|_| refused("allocation"))?;
        bytes.resize(VECTOR_COHORT_HEADER_BYTES_V1, 0);
        bytes[..4].copy_from_slice(MAGIC);
        bytes[4..6].copy_from_slice(&VERSION.to_le_bytes());
        bytes[6..8].copy_from_slice(&HEADER_BYTES_WIRE.to_le_bytes());
        bytes[8..40].copy_from_slice(&source.to_bytes());
        bytes[40..72].copy_from_slice(&tiers.fast_receipt.docset_digest);
        bytes[72..80].copy_from_slice(&tiers.fast_receipt.live_document_count.to_le_bytes());
        write_entry(
            &mut bytes,
            FAST_ENTRY,
            1,
            VECTOR_COHORT_HEADER_BYTES_V1,
            &tiers.fast_receipt.bytes,
        )?;
        write_entry(
            &mut bytes,
            QUALITY_ENTRY,
            2,
            VECTOR_COHORT_HEADER_BYTES_V1 + fast_bytes.len(),
            &tiers.quality_receipt.bytes,
        )?;
        append_image(cx, &mut bytes, &fast_bytes)?;
        append_image(cx, &mut bytes, &quality_bytes)?;
        Self::from_admitted(cx, bytes.into(), tiers)
    }

    /// Decode a complete selected cohort against both caller-owned bindings.
    ///
    /// The enclosing authority normally authenticates the aggregate bytes. This
    /// additionally checks framing, both internal images, full coverage and the
    /// independently supplied source checkpoint.
    ///
    /// # Errors
    /// Refuses unknown versions, malformed or overflowing framing, corruption,
    /// substituted tiers, source/coverage/generation drift, an exceeded image
    /// budget, allocation failure and cancellation.
    pub fn admit(
        cx: &Cx,
        bytes: Arc<[u8]>,
        expected_fast: &FsviV2IdentityBinding,
        expected_quality: &FsviV2IdentityBinding,
        source: SourceCheckpointV1,
        max_image_bytes: usize,
    ) -> SearchResult<Self> {
        checkpoint(cx)?;
        if bytes.len() > max_image_bytes {
            return Err(refused("image-budget"));
        }
        if bytes.len() < VECTOR_COHORT_HEADER_BYTES_V1
            || &bytes[..4] != MAGIC
            || read_u16(&bytes, 4)? != VERSION
            || read_u16(&bytes, 6)? != HEADER_BYTES_WIRE
        {
            return Err(refused("header"));
        }
        if bytes[8..40] != source.to_bytes() {
            return Err(refused("source-checkpoint"));
        }
        let fast = read_entry(&bytes, FAST_ENTRY, 1, VECTOR_COHORT_HEADER_BYTES_V1)?;
        let quality = read_entry(&bytes, QUALITY_ENTRY, 2, fast.end)?;
        if quality.end != bytes.len() {
            return Err(refused("trailing-bytes"));
        }
        checked_total(fast.payload.len(), quality.payload.len(), max_image_bytes)?;
        if hash_image(cx, fast.payload)? != fast.sha256
            || hash_image(cx, quality.payload)? != quality.sha256
        {
            return Err(refused("payload-digest"));
        }
        let tiers = admit_tiers(
            cx,
            (expected_fast, copy_image(cx, fast.payload)?),
            (expected_quality, copy_image(cx, quality.payload)?),
            source,
        )?;
        if bytes[40..72] != tiers.fast_receipt.docset_digest
            || read_u64(&bytes, 72)? != tiers.fast_receipt.live_document_count
        {
            return Err(refused("header-coverage"));
        }
        Self::from_admitted(cx, bytes, tiers)
    }

    fn from_admitted(cx: &Cx, bytes: Arc<[u8]>, tiers: AdmittedTiers) -> SearchResult<Self> {
        let receipt = ExactComponentReceiptV1 {
            role: GenerationComponentRole::Vector,
            bytes: GenerationComponentReceiptV1 {
                byte_len: u64::try_from(bytes.len()).map_err(|_| refused("image-length"))?,
                sha256: hash_image(cx, &bytes)?,
            },
            docset_digest: tiers.fast_receipt.docset_digest,
            live_document_count: tiers.fast_receipt.live_document_count,
            source_checkpoint: tiers.fast_receipt.source_checkpoint,
        };
        receipt
            .validate()
            .map_err(|_| refused("aggregate-receipt"))?;
        checkpoint(cx)?;
        Ok(Self {
            bytes,
            tiers,
            receipt,
        })
    }

    /// Share the canonical aggregate image without reopening a pathname.
    #[must_use]
    pub fn bytes(&self) -> Arc<[u8]> {
        Arc::clone(&self.bytes)
    }

    /// Independently admitted fast FSVI owner.
    #[must_use]
    pub const fn fast(&self) -> &Arc<ValidatedFsviBytes> {
        &self.tiers.fast
    }

    /// Independently admitted quality FSVI owner.
    #[must_use]
    pub const fn quality(&self) -> &Arc<ValidatedFsviBytes> {
        &self.tiers.quality
    }

    /// Aggregate Vector receipt for the exact authority component join.
    #[must_use]
    pub const fn receipt(&self) -> &ExactComponentReceiptV1 {
        &self.receipt
    }

    /// Receipt of the embedded fast image, for its individual metadata artifact.
    #[must_use]
    pub const fn fast_receipt(&self) -> &ExactComponentReceiptV1 {
        &self.tiers.fast_receipt
    }

    /// Receipt of the embedded quality image, for its individual metadata artifact.
    #[must_use]
    pub const fn quality_receipt(&self) -> &ExactComponentReceiptV1 {
        &self.tiers.quality_receipt
    }
}

fn admit_tiers(
    cx: &Cx,
    fast: (&FsviV2IdentityBinding, Arc<[u8]>),
    quality: (&FsviV2IdentityBinding, Arc<[u8]>),
    source: SourceCheckpointV1,
) -> SearchResult<AdmittedTiers> {
    checkpoint(cx)?;
    let fast_identity = &fast.0.frozen_identity().identity;
    let quality_identity = &quality.0.frozen_identity().identity;
    if fast.0.generation() != quality.0.generation() {
        return Err(refused("generation"));
    }
    if fast_identity.input.doc_id_semantics != quality_identity.input.doc_id_semantics {
        return Err(refused("document-id-semantics"));
    }
    if fast_identity.space.kind != quality_identity.space.kind {
        return Err(refused("space-kind"));
    }
    let fast =
        Arc::new(ValidatedFsviBytes::from_arc(fast.1, fast.0).map_err(|_| refused("fast-image"))?);
    checkpoint(cx)?;
    let quality = Arc::new(
        ValidatedFsviBytes::from_arc(quality.1, quality.0).map_err(|_| refused("quality-image"))?,
    );
    checkpoint(cx)?;
    let fast_receipt = owner_receipt(cx, &fast, source)?;
    let quality_receipt = owner_receipt(cx, &quality, source)?;
    if fast_receipt.docset_digest != quality_receipt.docset_digest
        || fast_receipt.live_document_count != quality_receipt.live_document_count
    {
        return Err(refused("complete-coverage"));
    }
    Ok(AdmittedTiers {
        fast,
        quality,
        fast_receipt,
        quality_receipt,
    })
}

fn owner_receipt(
    cx: &Cx,
    owner: &ValidatedFsviBytes,
    source: SourceCheckpointV1,
) -> SearchResult<ExactComponentReceiptV1> {
    let mut documents = Vec::new();
    documents
        .try_reserve_exact(owner.live_count())
        .map_err(|_| refused("allocation"))?;
    for physical in 0..owner.record_count() {
        checkpoint(cx)?;
        let row = owner.row(physical)?;
        if row.flags().is_live() {
            documents.push(owner.doc_id_at(physical)?);
        }
    }
    vector_component_receipt(owner.witness(), documents, source)
        .map_err(|_| refused("tier-receipt"))
}

struct Entry<'a> {
    payload: &'a [u8],
    end: usize,
    sha256: [u8; 32],
}

fn read_entry(
    bytes: &[u8],
    position: usize,
    role: u8,
    expected_offset: usize,
) -> SearchResult<Entry<'_>> {
    if bytes[position] != role || bytes[position + 1..position + 8] != [0; 7] {
        return Err(refused("entry-role"));
    }
    let offset =
        usize::try_from(read_u64(bytes, position + 8)?).map_err(|_| refused("entry-offset"))?;
    let length =
        usize::try_from(read_u64(bytes, position + 16)?).map_err(|_| refused("entry-length"))?;
    if offset != expected_offset || length == 0 {
        return Err(refused("entry-layout"));
    }
    let end = offset
        .checked_add(length)
        .ok_or_else(|| refused("entry-overflow"))?;
    let payload = bytes
        .get(offset..end)
        .ok_or_else(|| refused("entry-bounds"))?;
    let sha256 = bytes[position + 24..position + 56]
        .try_into()
        .map_err(|_| refused("entry-digest"))?;
    Ok(Entry {
        payload,
        end,
        sha256,
    })
}

fn write_entry(
    bytes: &mut [u8],
    position: usize,
    role: u8,
    offset: usize,
    receipt: &GenerationComponentReceiptV1,
) -> SearchResult<()> {
    bytes[position] = role;
    bytes[position + 8..position + 16].copy_from_slice(
        &u64::try_from(offset)
            .map_err(|_| refused("entry-offset"))?
            .to_le_bytes(),
    );
    bytes[position + 16..position + 24].copy_from_slice(&receipt.byte_len.to_le_bytes());
    bytes[position + 24..position + 56].copy_from_slice(&receipt.sha256);
    Ok(())
}

fn read_u16(bytes: &[u8], offset: usize) -> SearchResult<u16> {
    Ok(u16::from_le_bytes(
        bytes
            .get(offset..offset + 2)
            .ok_or_else(|| refused("header"))?
            .try_into()
            .map_err(|_| refused("header"))?,
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> SearchResult<u64> {
    Ok(u64::from_le_bytes(
        bytes
            .get(offset..offset + 8)
            .ok_or_else(|| refused("header"))?
            .try_into()
            .map_err(|_| refused("header"))?,
    ))
}

fn checked_total(fast: usize, quality: usize, limit: usize) -> SearchResult<usize> {
    VECTOR_COHORT_HEADER_BYTES_V1
        .checked_add(fast)
        .and_then(|size| size.checked_add(quality))
        .filter(|size| *size <= limit)
        .ok_or_else(|| refused("image-budget"))
}

fn copy_image(cx: &Cx, bytes: &[u8]) -> SearchResult<Arc<[u8]>> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(bytes.len())
        .map_err(|_| refused("allocation"))?;
    append_image(cx, &mut copy, bytes)?;
    Ok(copy.into())
}

fn append_image(cx: &Cx, output: &mut Vec<u8>, bytes: &[u8]) -> SearchResult<()> {
    for chunk in bytes.chunks(CHUNK_BYTES) {
        checkpoint(cx)?;
        output.extend_from_slice(chunk);
    }
    checkpoint(cx)
}

fn hash_image(cx: &Cx, bytes: &[u8]) -> SearchResult<[u8; 32]> {
    let mut hasher = Sha256::new();
    for chunk in bytes.chunks(CHUNK_BYTES) {
        checkpoint(cx)?;
        hasher.update(chunk);
    }
    checkpoint(cx)?;
    Ok(hasher.finalize().into())
}

fn checkpoint(cx: &Cx) -> SearchResult<()> {
    if cx.checkpoint().is_err() || cx.is_cancel_requested() {
        return Err(SearchError::Cancelled {
            phase: "vector_cohort.admission".to_owned(),
            reason: "operation cancelled".to_owned(),
        });
    }
    Ok(())
}

fn refused(reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: "vector_cohort".to_owned(),
        value: reason.to_owned(),
        reason: "complete fast/quality vector cohort admission failed".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    use frankensearch_core::generation::{
        ArtifactGenerationIdentityV1, CommitRange, EmbeddingArtifactIdentityV1,
        EmbeddingIdentityBundleV1, EmbeddingSpaceKindV1, QuantizationFormat,
    };

    use crate::VectorIndex;

    const BUDGET: usize = 4 * 1024 * 1024;
    const LIVE: [(&str, bool); 2] = [("zeta", true), ("alpha", true)];

    fn source() -> SourceCheckpointV1 {
        SourceCheckpointV1::derive(&CommitRange { low: 4, high: 9 })
    }

    fn binding(name: &str, dimension: u32) -> FsviV2IdentityBinding {
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(name, dimension);
        identity.storage.format = "fsvi-v2".to_owned();
        identity.storage.quantization = QuantizationFormat::F32;
        identity.storage.endianness = "little-endian".to_owned();
        FsviV2IdentityBinding::new(
            ArtifactGenerationIdentityV1::new(7, [0x17; 16]).unwrap(),
            identity.freeze().unwrap(),
        )
        .unwrap()
    }

    fn write_image(
        directory: &Path,
        name: &str,
        binding: &FsviV2IdentityBinding,
        rows: &[(&str, bool)],
    ) -> Arc<[u8]> {
        let path = directory.join(name);
        let mut writer = VectorIndex::create_v2(&path, binding.clone()).unwrap();
        for (index, (id, live)) in rows.iter().enumerate() {
            let mut vector = vec![0.0; binding.dimension()];
            vector[index % binding.dimension()] = 1.0;
            if *live {
                writer.write_record(id, &vector).unwrap();
            } else {
                writer.write_tombstone_record(id, &vector).unwrap();
            }
        }
        writer.finish().unwrap();
        std::fs::read(path).unwrap().into()
    }

    struct Fixture {
        directory: tempfile::TempDir,
        fast: FsviV2IdentityBinding,
        quality: FsviV2IdentityBinding,
        fast_bytes: Arc<[u8]>,
        quality_bytes: Arc<[u8]>,
    }

    impl Fixture {
        fn new(quality_dimension: u32) -> Self {
            let directory = tempfile::tempdir().unwrap();
            let fast = binding("fast-control", 2);
            let quality = binding("quality-control", quality_dimension);
            let fast_bytes = write_image(directory.path(), "fast.fsvi", &fast, &LIVE);
            let quality_bytes = write_image(directory.path(), "quality.fsvi", &quality, &LIVE);
            Self {
                directory,
                fast,
                quality,
                fast_bytes,
                quality_bytes,
            }
        }

        fn build(&self) -> ExactVectorCohortV1 {
            ExactVectorCohortV1::build(
                &Cx::for_testing(),
                (&self.fast, Arc::clone(&self.fast_bytes)),
                (&self.quality, Arc::clone(&self.quality_bytes)),
                source(),
                BUDGET,
            )
            .unwrap()
        }

        fn admit(&self, bytes: Arc<[u8]>) -> SearchResult<ExactVectorCohortV1> {
            ExactVectorCohortV1::admit(
                &Cx::for_testing(),
                bytes,
                &self.fast,
                &self.quality,
                source(),
                BUDGET,
            )
        }
    }

    #[test]
    fn complete_cohort_roundtrips_independent_dimensions_and_retains_exact_images() {
        let fixture = Fixture::new(3);
        let built = fixture.build();
        let bytes = built.bytes();
        let expected_len =
            VECTOR_COHORT_HEADER_BYTES_V1 + fixture.fast_bytes.len() + fixture.quality_bytes.len();
        assert_eq!(bytes.len(), expected_len);
        assert_eq!(
            built.receipt().bytes.byte_len,
            u64::try_from(expected_len).unwrap()
        );
        assert_eq!(
            built.receipt().bytes.sha256,
            <[u8; 32]>::from(Sha256::digest(&bytes))
        );
        assert_eq!(built.fast().dimension(), 2);
        assert_eq!(built.quality().dimension(), 3);
        assert_eq!(built.receipt().live_document_count, 2);
        assert_eq!(built.receipt().source_checkpoint, source().to_bytes());
        assert_eq!(
            built.fast_receipt().docset_digest,
            built.quality_receipt().docset_digest
        );
        assert_eq!(
            built.fast_receipt().bytes.sha256,
            built.fast().witness().whole_image_sha256
        );
        assert_eq!(
            built.quality_receipt().bytes.sha256,
            built.quality().witness().whole_image_sha256
        );
        assert_ne!(
            built.fast_receipt().bytes.sha256,
            built.quality_receipt().bytes.sha256
        );

        let reopened = fixture.admit(Arc::clone(&bytes)).unwrap();
        assert_eq!(reopened.receipt(), built.receipt());
        assert_eq!(reopened.fast().witness(), built.fast().witness());
        assert_eq!(reopened.quality().witness(), built.quality().witness());
        assert_eq!(reopened.bytes(), bytes);

        // Aggregate admission has no dependence on the files that produced it.
        std::fs::rename(
            fixture.directory.path().join("fast.fsvi"),
            fixture.directory.path().join("moved.fsvi"),
        )
        .unwrap();
        std::fs::write(
            fixture.directory.path().join("fast.fsvi"),
            b"unrelated replacement",
        )
        .unwrap();
        let retained = fixture.admit(bytes).unwrap();
        assert_eq!(retained.fast().witness(), built.fast().witness());
        assert_eq!(retained.fast().vector_at_f32(0).unwrap().len(), 2);
        assert_eq!(retained.quality().vector_at_f32(0).unwrap().len(), 3);
    }

    #[test]
    fn explicitly_identical_providers_and_empty_complete_cohorts_are_representable() {
        let fixture = Fixture::new(2);
        let cohort = ExactVectorCohortV1::build(
            &Cx::for_testing(),
            (&fixture.fast, Arc::clone(&fixture.fast_bytes)),
            (&fixture.fast, Arc::clone(&fixture.fast_bytes)),
            source(),
            BUDGET,
        )
        .unwrap();
        assert_eq!(cohort.fast_receipt(), cohort.quality_receipt());
        ExactVectorCohortV1::admit(
            &Cx::for_testing(),
            cohort.bytes(),
            &fixture.fast,
            &fixture.fast,
            source(),
            BUDGET,
        )
        .unwrap();

        let empty = write_image(fixture.directory.path(), "empty.fsvi", &fixture.fast, &[]);
        let empty = ExactVectorCohortV1::build(
            &Cx::for_testing(),
            (&fixture.fast, Arc::clone(&empty)),
            (&fixture.fast, empty),
            source(),
            BUDGET,
        )
        .unwrap();
        assert_eq!(empty.receipt().live_document_count, 0);
        let admitted = ExactVectorCohortV1::admit(
            &Cx::for_testing(),
            empty.bytes(),
            &fixture.fast,
            &fixture.fast,
            source(),
            BUDGET,
        )
        .unwrap();
        assert_eq!(admitted.fast().live_count(), 0);
        assert_eq!(admitted.quality().live_count(), 0);
    }

    #[test]
    fn every_header_byte_and_payload_digest_is_checked() {
        let fixture = Fixture::new(3);
        let encoded = fixture.build().bytes();
        for offset in 0..VECTOR_COHORT_HEADER_BYTES_V1 {
            let mut changed = encoded.to_vec();
            changed[offset] ^= 1;
            assert!(
                fixture.admit(changed.into()).is_err(),
                "header byte {offset}"
            );
        }
        for length in 0..VECTOR_COHORT_HEADER_BYTES_V1 {
            assert!(
                fixture.admit(encoded[..length].to_vec().into()).is_err(),
                "truncated at {length}"
            );
        }
        for offset in [
            VECTOR_COHORT_HEADER_BYTES_V1,
            VECTOR_COHORT_HEADER_BYTES_V1 + fixture.fast_bytes.len() - 1,
            VECTOR_COHORT_HEADER_BYTES_V1 + fixture.fast_bytes.len(),
            encoded.len() - 1,
        ] {
            let mut changed = encoded.to_vec();
            changed[offset] ^= 1;
            assert!(
                fixture.admit(changed.into()).is_err(),
                "payload byte {offset}"
            );
            assert!(
                fixture.admit(encoded[..offset].to_vec().into()).is_err(),
                "truncated payload {offset}"
            );
        }
        let mut trailing = encoded.to_vec();
        trailing.push(0);
        assert!(fixture.admit(trailing.into()).is_err());

        for position in [FAST_ENTRY, QUALITY_ENTRY] {
            for field in [8, 16] {
                let mut changed = encoded.to_vec();
                changed[position + field..position + field + 8]
                    .copy_from_slice(&u64::MAX.to_le_bytes());
                assert!(
                    fixture.admit(changed.into()).is_err(),
                    "overflowed field {position}/{field}"
                );
            }
        }
        fixture.admit(encoded).unwrap();
    }

    #[test]
    fn recomputed_container_hashes_do_not_bypass_either_fsvi_admission() {
        let fixture = Fixture::new(3);
        let encoded = fixture.build().bytes();
        for (entry, offset, length) in [
            (
                FAST_ENTRY,
                VECTOR_COHORT_HEADER_BYTES_V1,
                fixture.fast_bytes.len(),
            ),
            (
                QUALITY_ENTRY,
                VECTOR_COHORT_HEADER_BYTES_V1 + fixture.fast_bytes.len(),
                fixture.quality_bytes.len(),
            ),
        ] {
            let mut changed = encoded.to_vec();
            changed[offset..offset + 4].copy_from_slice(b"nope");
            let hash: [u8; 32] = Sha256::digest(&changed[offset..offset + length]).into();
            changed[entry + 24..entry + 56].copy_from_slice(&hash);
            assert!(fixture.admit(changed.into()).is_err());
        }
        assert!(
            ExactVectorCohortV1::build(
                &Cx::for_testing(),
                (&fixture.fast, Arc::from(&b"not an fsvi image"[..])),
                (&fixture.quality, Arc::clone(&fixture.quality_bytes)),
                source(),
                BUDGET,
            )
            .is_err()
        );
    }

    #[test]
    fn both_bindings_are_independent_even_when_dimensions_agree() {
        let fixture = Fixture::new(2);
        let encoded = fixture.build().bytes();
        assert!(
            ExactVectorCohortV1::admit(
                &Cx::for_testing(),
                Arc::clone(&encoded),
                &fixture.quality,
                &fixture.fast,
                source(),
                BUDGET,
            )
            .is_err()
        );
        let mut identity = fixture.quality.frozen_identity().identity.clone();
        identity
            .producer
            .implementation_revision
            .push_str("-foreign");
        let foreign =
            FsviV2IdentityBinding::new(fixture.quality.generation(), identity.freeze().unwrap())
                .unwrap();
        assert_eq!(foreign.dimension(), fixture.quality.dimension());
        assert!(
            ExactVectorCohortV1::admit(
                &Cx::for_testing(),
                Arc::clone(&encoded),
                &fixture.fast,
                &foreign,
                source(),
                BUDGET,
            )
            .is_err()
        );
        assert!(
            ExactVectorCohortV1::build(
                &Cx::for_testing(),
                (&fixture.fast, Arc::clone(&fixture.fast_bytes)),
                (&foreign, Arc::clone(&fixture.quality_bytes)),
                source(),
                BUDGET,
            )
            .is_err()
        );
        fixture.admit(encoded).unwrap();
    }

    #[test]
    fn complete_coverage_rejects_partial_and_same_count_substitution_but_allows_tombstones() {
        let fixture = Fixture::new(3);
        for (name, rows) in [
            ("partial.fsvi", vec![("zeta", true)]),
            ("foreign.fsvi", vec![("zeta", true), ("different", true)]),
            ("deleted.fsvi", vec![("zeta", true), ("alpha", false)]),
        ] {
            let quality = write_image(fixture.directory.path(), name, &fixture.quality, &rows);
            assert!(
                ExactVectorCohortV1::build(
                    &Cx::for_testing(),
                    (&fixture.fast, Arc::clone(&fixture.fast_bytes)),
                    (&fixture.quality, quality),
                    source(),
                    BUDGET,
                )
                .is_err(),
                "{name}"
            );
        }
        let quality = write_image(
            fixture.directory.path(),
            "tombstones.fsvi",
            &fixture.quality,
            &[("zeta", true), ("dead", false), ("alpha", true)],
        );
        let cohort = ExactVectorCohortV1::build(
            &Cx::for_testing(),
            (&fixture.fast, Arc::clone(&fixture.fast_bytes)),
            (&fixture.quality, quality),
            source(),
            BUDGET,
        )
        .unwrap();
        assert_eq!(cohort.fast().record_count(), 2);
        assert_eq!(cohort.quality().record_count(), 3);
        assert_eq!(cohort.receipt().live_document_count, 2);
        fixture.admit(cohort.bytes()).unwrap();
    }

    #[test]
    fn cohort_rejects_cross_generation_doc_id_semantics_and_mixed_provenance() {
        let fixture = Fixture::new(3);
        let different_generation = FsviV2IdentityBinding::new(
            ArtifactGenerationIdentityV1::new(7, [0x18; 16]).unwrap(),
            fixture.quality.frozen_identity().clone(),
        )
        .unwrap();
        let mut different_semantics = fixture.quality.frozen_identity().identity.clone();
        different_semantics
            .input
            .doc_id_semantics
            .push_str("-another-cohort");
        different_semantics.space.input_contract_fingerprint =
            different_semantics.input.fingerprint();
        different_semantics.producer.space_fingerprint = different_semantics.space.fingerprint();
        let different_semantics = FsviV2IdentityBinding::new(
            fixture.quality.generation(),
            different_semantics.freeze().unwrap(),
        )
        .unwrap();
        let mut semantic = fixture.quality.frozen_identity().identity.clone();
        semantic.space.kind = EmbeddingSpaceKindV1::Semantic;
        semantic.space.hash_control = None;
        semantic.space.artifact_manifest_fingerprint = "a".repeat(64);
        semantic.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
            role: "weights".to_owned(),
            sha256: "b".repeat(64),
            size: 1,
        }];
        semantic.producer.space_fingerprint = semantic.space.fingerprint();
        let semantic =
            FsviV2IdentityBinding::new(fixture.quality.generation(), semantic.freeze().unwrap())
                .unwrap();
        for (name, quality_binding) in [
            ("other-generation.fsvi", different_generation),
            ("other-semantics.fsvi", different_semantics),
            ("semantic.fsvi", semantic),
        ] {
            let quality = write_image(fixture.directory.path(), name, &quality_binding, &LIVE);
            ValidatedFsviBytes::from_arc(Arc::clone(&quality), &quality_binding).unwrap();
            assert!(
                ExactVectorCohortV1::build(
                    &Cx::for_testing(),
                    (&fixture.fast, Arc::clone(&fixture.fast_bytes)),
                    (&quality_binding, quality),
                    source(),
                    BUDGET,
                )
                .is_err(),
                "{name}"
            );
        }
    }

    #[test]
    fn source_budget_and_cancellation_apply_before_success_or_large_allocation() {
        let fixture = Fixture::new(3);
        let built = fixture.build();
        let bytes = built.bytes();
        let cx = Cx::for_testing();
        let other_source = SourceCheckpointV1::derive(&CommitRange { low: 4, high: 10 });
        assert!(
            ExactVectorCohortV1::admit(
                &cx,
                Arc::clone(&bytes),
                &fixture.fast,
                &fixture.quality,
                other_source,
                BUDGET
            )
            .is_err()
        );
        for budget in [0, VECTOR_COHORT_HEADER_BYTES_V1, bytes.len() - 1] {
            assert!(
                ExactVectorCohortV1::admit(
                    &cx,
                    Arc::clone(&bytes),
                    &fixture.fast,
                    &fixture.quality,
                    source(),
                    budget
                )
                .is_err()
            );
            assert!(
                ExactVectorCohortV1::build(
                    &cx,
                    (&fixture.fast, Arc::clone(&fixture.fast_bytes)),
                    (&fixture.quality, Arc::clone(&fixture.quality_bytes)),
                    source(),
                    budget
                )
                .is_err()
            );
        }
        ExactVectorCohortV1::admit(
            &cx,
            Arc::clone(&bytes),
            &fixture.fast,
            &fixture.quality,
            source(),
            bytes.len(),
        )
        .unwrap();
        ExactVectorCohortV1::build(
            &cx,
            (&fixture.fast, Arc::clone(&fixture.fast_bytes)),
            (&fixture.quality, Arc::clone(&fixture.quality_bytes)),
            source(),
            bytes.len(),
        )
        .unwrap();
        assert!(checked_total(usize::MAX, 1, usize::MAX).is_err());
        assert!(checked_total(1, usize::MAX, usize::MAX).is_err());
        cx.set_cancel_requested(true);
        assert!(matches!(
            ExactVectorCohortV1::admit(
                &cx,
                bytes,
                &fixture.fast,
                &fixture.quality,
                source(),
                BUDGET
            ),
            Err(SearchError::Cancelled { .. })
        ));
        assert!(matches!(
            ExactVectorCohortV1::build(
                &cx,
                (&fixture.fast, fixture.fast_bytes),
                (&fixture.quality, fixture.quality_bytes),
                source(),
                BUDGET
            ),
            Err(SearchError::Cancelled { .. })
        ));
    }
}
