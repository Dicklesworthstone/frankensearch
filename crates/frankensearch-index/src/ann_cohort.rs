//! Independently bound fast/quality native graphs in one authority Ann component.
//!
//! An admitted [`ExactVectorCohortV1`] supplies the complete source cohort and
//! the two immutable vector owners. Every present graph requires its own
//! separately authenticated [`NativeHnswGenerationReceiptV2`]; no receipt or
//! graph is inferred from the other tier. A missing graph means exact retrieval
//! only when the caller's selected metadata explicitly declares that absence.
//!
//! # FSAC version 1 wire contract
//!
//! All integers are little-endian. The fixed header is exactly 192 bytes:
//!
//! | Offset | Length | Field |
//! |---:|---:|---|
//! | 0 | 4 | `FSAC` magic |
//! | 4 | 2 | Version, exactly 1 |
//! | 6 | 2 | Header length, exactly 192 |
//! | 8 | 32 | Source checkpoint from the admitted vector cohort |
//! | 40 | 32 | Canonical ordered live-document-set digest |
//! | 72 | 8 | Common live-document count |
//! | 80 | 56 | Fast entry |
//! | 136 | 56 | Quality entry |
//!
//! Each entry contains a one-byte role (fast=1, quality=2), one-byte presence
//! (0=exact, 1=HNSW), six zero bytes, an eight-byte offset, an eight-byte length,
//! and a 32-byte SHA-256. At least one graph is present. The fast offset is 192;
//! the quality offset is exactly the fast offset plus its length. Absent graphs
//! have zero length and digest, while present graphs have a nonzero length.
//! Payloads are unchanged FSHNSW images. Gaps, padding, alternate role ordering,
//! hidden absent payloads, unknown flags/versions and trailing bytes are refused.
//!
//! Graphs may have different dimensions, producers, parameters, seeds and
//! physical row counts. Each graph is structurally admitted against its own
//! exact retained FSVI owner. The enclosing Vector cohort already requires the
//! same artifact generation, document-ID semantics, space kind and complete
//! ordered live coverage. The aggregate Ann receipt authenticates all framing
//! and payload bytes plus that same source checkpoint and document set.

use std::fmt;
use std::sync::Arc;

use frankensearch_core::generation::{
    ExactComponentReceiptV1, GenerationComponentReceiptV1, GenerationComponentRole,
};
use frankensearch_core::{Cx, SearchError, SearchResult};
use sha2::{Digest as _, Sha256};

use crate::ValidatedFsviBytes;
use crate::native_hnsw::{NativeHnswGenerationReceiptV2, ValidatedNativeHnsw};
use crate::vector_cohort::ExactVectorCohortV1;

/// Encoded overhead of a complete FSAC v1 fast/quality graph cohort.
pub const ANN_COHORT_HEADER_BYTES_V1: usize = 192;
const HEADER_BYTES_WIRE: u16 = 192;
const MAGIC: &[u8; 4] = b"FSAC";
const VERSION: u16 = 1;
const FAST_ENTRY: usize = 80;
const QUALITY_ENTRY: usize = 136;
const CHUNK_BYTES: usize = 1024 * 1024;

/// Immutable independently admitted graphs over one complete vector cohort.
///
/// Construction and admission are synchronous, open no paths and start no
/// runtime or tasks. Cancellation is checked around graph admission and between
/// bounded copying/hashing chunks. The existing graph parser is synchronous;
/// its internal structural validation has the same execution contract as
/// [`ValidatedNativeHnsw::from_bytes`]. The image budget bounds encoded bytes,
/// not decoded graph adjacency or total process memory.
pub struct NativeAnnCohortV1 {
    bytes: Arc<[u8]>,
    fast: Option<ValidatedNativeHnsw>,
    quality: Option<ValidatedNativeHnsw>,
    receipt: ExactComponentReceiptV1,
}

impl fmt::Debug for NativeAnnCohortV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeAnnCohortV1")
            .field("fast_hnsw", &self.fast.is_some())
            .field("quality_hnsw", &self.quality.is_some())
            .field("receipt", &self.receipt)
            .finish_non_exhaustive()
    }
}

impl NativeAnnCohortV1 {
    /// Validate and pack the explicitly configured graphs for a full cohort.
    ///
    /// Exact/HNSW, HNSW/exact and HNSW/HNSW are represented independently.
    /// Exact/exact has no Ann component and is rejected by this constructor.
    /// Receipts must come from the caller's selected graph artifacts; the
    /// encoded container does not discover or select replacement receipts.
    ///
    /// # Errors
    /// Refuses an empty graph set, incompatible receipts/owners, malformed
    /// graphs, image-budget overflow, allocation failure and cancellation.
    pub fn build(
        cx: &Cx,
        cohort: &ExactVectorCohortV1,
        fast: Option<(&NativeHnswGenerationReceiptV2, Arc<[u8]>)>,
        quality: Option<(&NativeHnswGenerationReceiptV2, Arc<[u8]>)>,
        max_image_bytes: usize,
    ) -> SearchResult<Self> {
        checkpoint(cx)?;
        if fast.is_none() && quality.is_none() {
            return Err(refused("empty-graph-cohort"));
        }
        let fast_len = fast.as_ref().map_or(0, |(_, bytes)| bytes.len());
        let quality_len = quality.as_ref().map_or(0, |(_, bytes)| bytes.len());
        let total = checked_total(fast_len, quality_len, max_image_bytes)?;
        let fast_graph = admit_graph(cx, cohort.fast(), fast.as_ref())?;
        let quality_graph = admit_graph(cx, cohort.quality(), quality.as_ref())?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(total)
            .map_err(|_| refused("allocation"))?;
        bytes.resize(ANN_COHORT_HEADER_BYTES_V1, 0);
        bytes[..4].copy_from_slice(MAGIC);
        bytes[4..6].copy_from_slice(&VERSION.to_le_bytes());
        bytes[6..8].copy_from_slice(&HEADER_BYTES_WIRE.to_le_bytes());
        bytes[8..40].copy_from_slice(&cohort.receipt().source_checkpoint);
        bytes[40..72].copy_from_slice(&cohort.receipt().docset_digest);
        bytes[72..80].copy_from_slice(&cohort.receipt().live_document_count.to_le_bytes());
        write_entry(
            cx,
            &mut bytes,
            FAST_ENTRY,
            1,
            ANN_COHORT_HEADER_BYTES_V1,
            fast.as_ref().map(|(_, bytes)| bytes.as_ref()),
        )?;
        write_entry(
            cx,
            &mut bytes,
            QUALITY_ENTRY,
            2,
            ANN_COHORT_HEADER_BYTES_V1 + fast_len,
            quality.as_ref().map(|(_, bytes)| bytes.as_ref()),
        )?;
        for (_, image) in [fast, quality].into_iter().flatten() {
            append_image(cx, &mut bytes, &image)?;
        }
        Self::from_admitted(cx, bytes.into(), cohort, fast_graph, quality_graph)
    }

    /// Admit retained aggregate bytes using the caller's authenticated tier plan.
    ///
    /// Receipt presence must match the encoded graph presence exactly. A corrupt
    /// or omitted declared graph is an error, never a switch to exact search.
    /// All framing, lengths, digests and common coverage are checked before
    /// allocating child images. Each graph then receives ordinary owner-bound
    /// native admission, including source-map and seeded topology validation.
    ///
    /// # Errors
    /// Refuses malformed/noncanonical framing, substituted policies or graphs,
    /// source/coverage disagreement, incompatible owners/receipts, an exceeded
    /// image budget, allocation failure and cancellation.
    pub fn admit(
        cx: &Cx,
        bytes: Arc<[u8]>,
        cohort: &ExactVectorCohortV1,
        fast_receipt: Option<&NativeHnswGenerationReceiptV2>,
        quality_receipt: Option<&NativeHnswGenerationReceiptV2>,
        max_image_bytes: usize,
    ) -> SearchResult<Self> {
        checkpoint(cx)?;
        if bytes.len() > max_image_bytes {
            return Err(refused("image-budget"));
        }
        if bytes.len() < ANN_COHORT_HEADER_BYTES_V1
            || &bytes[..4] != MAGIC
            || read_u16(&bytes, 4)? != VERSION
            || read_u16(&bytes, 6)? != HEADER_BYTES_WIRE
        {
            return Err(refused("header"));
        }
        if fast_receipt.is_none() && quality_receipt.is_none() {
            return Err(refused("empty-graph-cohort"));
        }
        if bytes[8..40] != cohort.receipt().source_checkpoint {
            return Err(refused("source-checkpoint"));
        }
        if bytes[40..72] != cohort.receipt().docset_digest
            || read_u64(&bytes, 72)? != cohort.receipt().live_document_count
        {
            return Err(refused("header-coverage"));
        }
        let fast = read_entry(
            &bytes,
            FAST_ENTRY,
            1,
            fast_receipt.is_some(),
            ANN_COHORT_HEADER_BYTES_V1,
        )?;
        let quality = read_entry(
            &bytes,
            QUALITY_ENTRY,
            2,
            quality_receipt.is_some(),
            fast.end,
        )?;
        if quality.end != bytes.len() {
            return Err(refused("trailing-bytes"));
        }
        checked_total(fast.payload.len(), quality.payload.len(), max_image_bytes)?;
        for entry in [&fast, &quality] {
            if !entry.payload.is_empty() && hash_image(cx, entry.payload)? != entry.sha256 {
                return Err(refused("payload-digest"));
            }
        }
        let fast_image = fast_receipt
            .map(|receipt| copy_image(cx, fast.payload).map(|bytes| (receipt, bytes)))
            .transpose()?;
        let quality_image = quality_receipt
            .map(|receipt| copy_image(cx, quality.payload).map(|bytes| (receipt, bytes)))
            .transpose()?;
        let fast_graph = admit_graph(cx, cohort.fast(), fast_image.as_ref())?;
        let quality_graph = admit_graph(cx, cohort.quality(), quality_image.as_ref())?;
        Self::from_admitted(cx, bytes, cohort, fast_graph, quality_graph)
    }

    fn from_admitted(
        cx: &Cx,
        bytes: Arc<[u8]>,
        cohort: &ExactVectorCohortV1,
        fast: Option<ValidatedNativeHnsw>,
        quality: Option<ValidatedNativeHnsw>,
    ) -> SearchResult<Self> {
        let receipt = ExactComponentReceiptV1 {
            role: GenerationComponentRole::Ann,
            bytes: GenerationComponentReceiptV1 {
                byte_len: u64::try_from(bytes.len()).map_err(|_| refused("image-length"))?,
                sha256: hash_image(cx, &bytes)?,
            },
            docset_digest: cohort.receipt().docset_digest,
            live_document_count: cohort.receipt().live_document_count,
            source_checkpoint: cohort.receipt().source_checkpoint,
        };
        receipt
            .validate()
            .map_err(|_| refused("aggregate-receipt"))?;
        checkpoint(cx)?;
        Ok(Self {
            bytes,
            fast,
            quality,
            receipt,
        })
    }

    /// Share the canonical immutable Ann component without reopening a pathname.
    #[must_use]
    pub fn bytes(&self) -> Arc<[u8]> {
        Arc::clone(&self.bytes)
    }

    /// Fast graph, or `None` for the explicitly selected exact fast policy.
    #[must_use]
    pub const fn fast(&self) -> Option<&ValidatedNativeHnsw> {
        self.fast.as_ref()
    }

    /// Quality graph, or `None` for the explicitly selected exact quality policy.
    #[must_use]
    pub const fn quality(&self) -> Option<&ValidatedNativeHnsw> {
        self.quality.as_ref()
    }

    /// Aggregate Ann receipt for the ordinary complete-component authority join.
    #[must_use]
    pub const fn receipt(&self) -> &ExactComponentReceiptV1 {
        &self.receipt
    }

    /// Move both admitted graphs into retained readers without parsing again.
    ///
    /// Each present graph retains its original vector owner independently of
    /// this envelope and of the caller's vector-cohort handle.
    #[must_use]
    pub fn into_graphs(self) -> (Option<ValidatedNativeHnsw>, Option<ValidatedNativeHnsw>) {
        (self.fast, self.quality)
    }
}

fn admit_graph(
    cx: &Cx,
    owner: &Arc<ValidatedFsviBytes>,
    image: Option<&(&NativeHnswGenerationReceiptV2, Arc<[u8]>)>,
) -> SearchResult<Option<ValidatedNativeHnsw>> {
    checkpoint(cx)?;
    let graph = image
        .map(|(receipt, bytes)| {
            ValidatedNativeHnsw::from_bytes(Arc::clone(owner), Arc::clone(bytes), receipt)
        })
        .transpose()?;
    checkpoint(cx)?;
    Ok(graph)
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
    present: bool,
    expected_offset: usize,
) -> SearchResult<Entry<'_>> {
    if bytes[position] != role
        || bytes[position + 1] != u8::from(present)
        || bytes[position + 2..position + 8] != [0; 6]
    {
        return Err(refused("entry-policy"));
    }
    let offset =
        usize::try_from(read_u64(bytes, position + 8)?).map_err(|_| refused("entry-offset"))?;
    let length =
        usize::try_from(read_u64(bytes, position + 16)?).map_err(|_| refused("entry-length"))?;
    if offset != expected_offset || (length != 0) != present {
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
    if !present && sha256 != [0; 32] {
        return Err(refused("absent-digest"));
    }
    Ok(Entry {
        payload,
        end,
        sha256,
    })
}

fn write_entry(
    cx: &Cx,
    bytes: &mut [u8],
    position: usize,
    role: u8,
    offset: usize,
    payload: Option<&[u8]>,
) -> SearchResult<()> {
    bytes[position] = role;
    bytes[position + 1] = u8::from(payload.is_some());
    bytes[position + 8..position + 16].copy_from_slice(
        &u64::try_from(offset)
            .map_err(|_| refused("entry-offset"))?
            .to_le_bytes(),
    );
    if let Some(payload) = payload {
        bytes[position + 16..position + 24].copy_from_slice(
            &u64::try_from(payload.len())
                .map_err(|_| refused("entry-length"))?
                .to_le_bytes(),
        );
        bytes[position + 24..position + 56].copy_from_slice(&hash_image(cx, payload)?);
    }
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
    ANN_COHORT_HEADER_BYTES_V1
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
            phase: "ann_cohort.admission".to_owned(),
            reason: "operation cancelled".to_owned(),
        });
    }
    Ok(())
}

fn refused(reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: "ann_cohort".to_owned(),
        value: reason.to_owned(),
        reason: "complete fast/quality graph cohort admission failed".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    use frankensearch_core::generation::{
        ArtifactGenerationIdentityV1, CommitRange, EmbeddingIdentityBundleV1, QuantizationFormat,
        SourceCheckpointV1,
    };

    use crate::native_hnsw::HnswParams;
    use crate::{FsviV2IdentityBinding, VectorIndex};

    const BUDGET: usize = 4 * 1024 * 1024;
    const FAST_ROWS: &[(&str, bool)] = &[("zeta", true), ("gone", false), ("alpha", true)];
    const QUALITY_ROWS: &[(&str, bool)] = &[
        ("zeta", true),
        ("old-one", false),
        ("old-two", false),
        ("alpha", true),
    ];

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

    fn write_vectors(
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

    fn fast_policy() -> HnswParams {
        HnswParams {
            m: 4,
            m0: 8,
            ef_construction: 16,
            ef_search: 4,
        }
    }

    fn quality_policy() -> HnswParams {
        HnswParams {
            m: 6,
            m0: 12,
            ef_construction: 24,
            ef_search: 8,
        }
    }

    fn write_graph(
        owner: &Arc<ValidatedFsviBytes>,
        path: &Path,
        params: HnswParams,
        seed: u64,
    ) -> (Arc<[u8]>, NativeHnswGenerationReceiptV2) {
        let graph = ValidatedNativeHnsw::build(&Cx::for_testing(), Arc::clone(owner), params, seed)
            .unwrap();
        let receipt = graph.save(path).unwrap();
        (std::fs::read(path).unwrap().into(), receipt)
    }

    struct Fixture {
        directory: tempfile::TempDir,
        fast_binding: FsviV2IdentityBinding,
        quality_binding: FsviV2IdentityBinding,
        cohort: ExactVectorCohortV1,
        fast_image: Arc<[u8]>,
        quality_image: Arc<[u8]>,
        fast_receipt: NativeHnswGenerationReceiptV2,
        quality_receipt: NativeHnswGenerationReceiptV2,
    }

    impl Fixture {
        fn new() -> Self {
            Self::with_rows(FAST_ROWS, QUALITY_ROWS, false)
        }

        fn with_rows(
            fast_rows: &[(&str, bool)],
            quality_rows: &[(&str, bool)],
            identical: bool,
        ) -> Self {
            let directory = tempfile::tempdir().unwrap();
            let fast_binding = binding("fast-control", 2);
            let quality_binding = if identical {
                fast_binding.clone()
            } else {
                binding("quality-control", 3)
            };
            let fast_vectors =
                write_vectors(directory.path(), "fast.fsvi", &fast_binding, fast_rows);
            let quality_vectors = write_vectors(
                directory.path(),
                "quality.fsvi",
                &quality_binding,
                quality_rows,
            );
            let cohort = ExactVectorCohortV1::build(
                &Cx::for_testing(),
                (&fast_binding, fast_vectors),
                (&quality_binding, quality_vectors),
                source(),
                BUDGET,
            )
            .unwrap();
            let (fast_image, fast_receipt) = write_graph(
                cohort.fast(),
                &directory.path().join("fast.fshnsw"),
                fast_policy(),
                101,
            );
            let (quality_image, quality_receipt) = write_graph(
                cohort.quality(),
                &directory.path().join("quality.fshnsw"),
                quality_policy(),
                903,
            );
            Self {
                directory,
                fast_binding,
                quality_binding,
                cohort,
                fast_image,
                quality_image,
                fast_receipt,
                quality_receipt,
            }
        }

        fn build(&self, fast: bool, quality: bool) -> SearchResult<NativeAnnCohortV1> {
            NativeAnnCohortV1::build(
                &Cx::for_testing(),
                &self.cohort,
                fast.then(|| (&self.fast_receipt, Arc::clone(&self.fast_image))),
                quality.then(|| (&self.quality_receipt, Arc::clone(&self.quality_image))),
                BUDGET,
            )
        }

        fn admit(
            &self,
            bytes: Arc<[u8]>,
            fast: bool,
            quality: bool,
        ) -> SearchResult<NativeAnnCohortV1> {
            NativeAnnCohortV1::admit(
                &Cx::for_testing(),
                bytes,
                &self.cohort,
                fast.then_some(&self.fast_receipt),
                quality.then_some(&self.quality_receipt),
                BUDGET,
            )
        }
    }

    #[test]
    fn dual_graphs_roundtrip_independent_policies_and_remain_bound_after_path_changes() {
        let fixture = Fixture::new();
        let built = fixture.build(true, true).unwrap();
        let image = built.bytes();
        assert_eq!(built.receipt().role, GenerationComponentRole::Ann);
        assert_eq!(
            built.receipt().docset_digest,
            fixture.cohort.receipt().docset_digest
        );
        assert_eq!(built.receipt().source_checkpoint, source().to_bytes());
        assert_eq!(built.receipt().live_document_count, 2);
        assert_eq!(
            image.len(),
            ANN_COHORT_HEADER_BYTES_V1 + fixture.fast_image.len() + fixture.quality_image.len()
        );
        assert_eq!(
            &image
                [ANN_COHORT_HEADER_BYTES_V1..ANN_COHORT_HEADER_BYTES_V1 + fixture.fast_image.len()],
            fixture.fast_image.as_ref()
        );
        assert_eq!(
            &image[ANN_COHORT_HEADER_BYTES_V1 + fixture.fast_image.len()..],
            fixture.quality_image.as_ref()
        );
        std::fs::rename(
            fixture.directory.path().join("fast.fshnsw"),
            fixture.directory.path().join("archived-fast.fshnsw"),
        )
        .unwrap();
        std::fs::rename(
            fixture.directory.path().join("quality.fshnsw"),
            fixture.directory.path().join("archived-quality.fshnsw"),
        )
        .unwrap();
        let admitted = fixture.admit(image, true, true).unwrap();
        assert_eq!(admitted.receipt(), built.receipt());
        assert!(
            admitted
                .fast()
                .unwrap()
                .matches_build_policy(fast_policy(), 101)
        );
        assert!(
            admitted
                .quality()
                .unwrap()
                .matches_build_policy(quality_policy(), 903)
        );
        assert_eq!(
            admitted.fast().unwrap().owner_witness(),
            fixture.cohort.fast().witness()
        );
        assert_eq!(
            admitted.quality().unwrap().owner_witness(),
            fixture.cohort.quality().witness()
        );
        assert_eq!(admitted.fast().unwrap().len(), 3);
        assert_eq!(admitted.quality().unwrap().len(), 4);
        let (fast, quality) = admitted.into_graphs();
        let fast = fast.unwrap();
        let quality = quality.unwrap();
        let fast_hits = fast
            .search(&Cx::for_testing(), &[1.0, 0.0], 10, Some(16))
            .unwrap();
        let quality_hits = quality
            .search(&Cx::for_testing(), &[1.0, 0.0, 0.0], 10, Some(16))
            .unwrap();
        for hits in [&fast_hits, &quality_hits] {
            assert_eq!(hits.len(), 2);
            assert!(hits.iter().all(|hit| hit.flags().is_live()));
            assert!(hits.iter().any(|hit| hit.doc_id() == "zeta"));
            assert!(hits.iter().any(|hit| hit.doc_id() == "alpha"));
        }
    }

    #[test]
    fn mixed_policies_roundtrip_and_cannot_be_silently_changed() {
        let fixture = Fixture::new();
        for (fast, quality) in [(true, false), (false, true), (true, true)] {
            let built = fixture.build(fast, quality).unwrap();
            let admitted = fixture.admit(built.bytes(), fast, quality).unwrap();
            assert_eq!(admitted.fast().is_some(), fast);
            assert_eq!(admitted.quality().is_some(), quality);
            assert_eq!(admitted.receipt(), built.receipt());
            assert!(fixture.admit(built.bytes(), !fast, quality).is_err());
            assert!(fixture.admit(built.bytes(), fast, !quality).is_err());
        }
        assert!(fixture.build(false, false).is_err());
        assert!(
            fixture
                .admit(fixture.build(true, true).unwrap().bytes(), false, false)
                .is_err()
        );
    }

    #[test]
    fn identical_providers_and_empty_complete_cohorts_remain_explicit() {
        let identical = Fixture::with_rows(FAST_ROWS, QUALITY_ROWS, true);
        assert_eq!(
            identical
                .cohort
                .fast()
                .witness()
                .identity_bundle_fingerprint,
            identical
                .cohort
                .quality()
                .witness()
                .identity_bundle_fingerprint
        );
        let built = identical.build(true, true).unwrap();
        assert!(identical.admit(built.bytes(), true, true).is_ok());

        let empty = Fixture::with_rows(&[], &[], false);
        let built = empty.build(true, true).unwrap();
        let admitted = empty.admit(built.bytes(), true, true).unwrap();
        assert_eq!(admitted.receipt().live_document_count, 0);
        assert!(admitted.fast().unwrap().is_empty());
        assert!(admitted.quality().unwrap().is_empty());
    }

    #[test]
    fn every_framing_region_and_payload_digest_is_checked() {
        let fixture = Fixture::new();
        let image = fixture.build(true, true).unwrap().bytes();
        for offset in [
            0,
            4,
            6,
            8,
            40,
            72,
            FAST_ENTRY,
            FAST_ENTRY + 1,
            FAST_ENTRY + 2,
            FAST_ENTRY + 8,
            FAST_ENTRY + 16,
            FAST_ENTRY + 24,
            QUALITY_ENTRY,
            QUALITY_ENTRY + 1,
            QUALITY_ENTRY + 2,
            QUALITY_ENTRY + 8,
            QUALITY_ENTRY + 16,
            QUALITY_ENTRY + 24,
            ANN_COHORT_HEADER_BYTES_V1,
            image.len() - 1,
        ] {
            let mut corrupted = image.to_vec();
            corrupted[offset] ^= 0x80;
            assert!(
                fixture.admit(corrupted.into(), true, true).is_err(),
                "offset {offset}"
            );
        }
        for length in [0, 7, ANN_COHORT_HEADER_BYTES_V1 - 1, image.len() - 1] {
            assert!(
                fixture
                    .admit(Arc::from(&image[..length]), true, true)
                    .is_err()
            );
        }
        let mut trailing = image.to_vec();
        trailing.push(0);
        assert!(fixture.admit(trailing.into(), true, true).is_err());
        let mut overflowing = image.to_vec();
        overflowing[FAST_ENTRY + 16..FAST_ENTRY + 24].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(fixture.admit(overflowing.into(), true, true).is_err());
    }

    #[test]
    fn absent_entries_cannot_hide_payloads_digests_or_alternate_offsets() {
        let fixture = Fixture::new();
        for (fast, quality, absent) in [(false, true, FAST_ENTRY), (true, false, QUALITY_ENTRY)] {
            let image = fixture.build(fast, quality).unwrap().bytes();
            for field in [absent + 1, absent + 8, absent + 16, absent + 24] {
                let mut hidden = image.to_vec();
                hidden[field] ^= 1;
                assert!(fixture.admit(hidden.into(), fast, quality).is_err());
            }
        }
    }

    #[test]
    fn graph_receipts_and_complete_vector_owners_cannot_be_substituted() {
        let fixture = Fixture::new();
        let image = fixture.build(true, true).unwrap().bytes();
        assert!(
            NativeAnnCohortV1::admit(
                &Cx::for_testing(),
                Arc::clone(&image),
                &fixture.cohort,
                Some(&fixture.quality_receipt),
                Some(&fixture.fast_receipt),
                BUDGET,
            )
            .is_err()
        );
        assert!(
            NativeAnnCohortV1::build(
                &Cx::for_testing(),
                &fixture.cohort,
                Some((&fixture.quality_receipt, Arc::clone(&fixture.quality_image))),
                Some((&fixture.fast_receipt, Arc::clone(&fixture.fast_image))),
                BUDGET,
            )
            .is_err()
        );
        let different_physical_rows = Fixture::with_rows(
            &[("zeta", true), ("alpha", true)],
            &[("zeta", true), ("alpha", true)],
            false,
        );
        assert_eq!(
            different_physical_rows.cohort.receipt().docset_digest,
            fixture.cohort.receipt().docset_digest
        );
        assert!(
            NativeAnnCohortV1::admit(
                &Cx::for_testing(),
                Arc::clone(&image),
                &different_physical_rows.cohort,
                Some(&fixture.fast_receipt),
                Some(&fixture.quality_receipt),
                BUDGET,
            )
            .is_err()
        );

        // A self-consistent envelope checksum cannot replace the independently
        // selected native receipt's hash or its owner/structural checks.
        let mut resealed = image.to_vec();
        let fast_end = ANN_COHORT_HEADER_BYTES_V1 + fixture.fast_image.len();
        resealed[fast_end - 1] ^= 1;
        let digest = hash_image(
            &Cx::for_testing(),
            &resealed[ANN_COHORT_HEADER_BYTES_V1..fast_end],
        )
        .unwrap();
        resealed[FAST_ENTRY + 24..FAST_ENTRY + 56].copy_from_slice(&digest);
        assert!(fixture.admit(resealed.into(), true, true).is_err());
    }

    #[test]
    fn source_and_live_coverage_are_bound_to_the_selected_vector_cohort() {
        let fixture = Fixture::new();
        let image = fixture.build(true, true).unwrap().bytes();
        let other_source = ExactVectorCohortV1::build(
            &Cx::for_testing(),
            (
                &fixture.fast_binding,
                std::fs::read(fixture.directory.path().join("fast.fsvi"))
                    .unwrap()
                    .into(),
            ),
            (
                &fixture.quality_binding,
                std::fs::read(fixture.directory.path().join("quality.fsvi"))
                    .unwrap()
                    .into(),
            ),
            SourceCheckpointV1::derive(&CommitRange { low: 4, high: 10 }),
            BUDGET,
        )
        .unwrap();
        assert!(
            NativeAnnCohortV1::admit(
                &Cx::for_testing(),
                Arc::clone(&image),
                &other_source,
                Some(&fixture.fast_receipt),
                Some(&fixture.quality_receipt),
                BUDGET,
            )
            .is_err()
        );
        let different_documents =
            Fixture::with_rows(&[("different", true)], &[("different", true)], false);
        assert!(
            NativeAnnCohortV1::admit(
                &Cx::for_testing(),
                image,
                &different_documents.cohort,
                Some(&fixture.fast_receipt),
                Some(&fixture.quality_receipt),
                BUDGET,
            )
            .is_err()
        );
    }

    #[test]
    fn exact_image_budgets_and_cancellation_are_enforced_before_returning_graphs() {
        let fixture = Fixture::new();
        let image = fixture.build(true, true).unwrap().bytes();
        let admit = |cx: &Cx, budget| {
            NativeAnnCohortV1::admit(
                cx,
                Arc::clone(&image),
                &fixture.cohort,
                Some(&fixture.fast_receipt),
                Some(&fixture.quality_receipt),
                budget,
            )
        };
        assert!(admit(&Cx::for_testing(), image.len()).is_ok());
        assert!(admit(&Cx::for_testing(), image.len() - 1).is_err());
        assert!(
            NativeAnnCohortV1::build(
                &Cx::for_testing(),
                &fixture.cohort,
                Some((&fixture.fast_receipt, Arc::clone(&fixture.fast_image))),
                Some((&fixture.quality_receipt, Arc::clone(&fixture.quality_image))),
                image.len() - 1,
            )
            .is_err()
        );
        assert!(checked_total(usize::MAX, 1, usize::MAX).is_err());
        let cancelled = Cx::for_testing();
        cancelled.set_cancel_requested(true);
        assert!(matches!(
            admit(&cancelled, BUDGET),
            Err(SearchError::Cancelled { .. })
        ));
        assert!(matches!(
            NativeAnnCohortV1::build(
                &cancelled,
                &fixture.cohort,
                Some((&fixture.fast_receipt, Arc::clone(&fixture.fast_image))),
                None,
                BUDGET,
            ),
            Err(SearchError::Cancelled { .. })
        ));
    }
}
