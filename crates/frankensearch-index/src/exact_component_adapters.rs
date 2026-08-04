//! Adapters from engine-specific witnesses to engine-neutral exact-generation
//! component receipts (`bd-cycdy`).
//!
//! [`crate::FsviV2Witness`] and
//! [`crate::native_hnsw::NativeHnswGenerationReceiptV2`] already carry
//! everything the vector and ANN clauses of the exact-generation contract need,
//! but nothing converted them into the
//! [`ExactComponentReceiptV1`][frankensearch_core::generation::ExactComponentReceiptV1]
//! that
//! [`ExactGenerationComponentsV1::admit`][frankensearch_core::generation::ExactGenerationComponentsV1::admit]
//! consumes — so that join was only ever exercised by receipts constructed by
//! hand in core's own tests.
//!
//! These adapters live in the index crate rather than in core, so core keeps
//! zero engine dependencies.
//!
//! # Why the docset digest is recomputed rather than copied
//!
//! Both witnesses expose a field literally named `ordered_live_docset_digest`,
//! and it is the wrong value to put in an
//! [`ExactComponentReceiptV1::docset_digest`][frankensearch_core::generation::ExactComponentReceiptV1::docset_digest].
//! The witness digest is computed under the FSVI-local domain
//! `frankensearch.fsvi-v2.ordered-live-docset.v1`, while the receipt field is
//! defined under core's
//! [`CANONICAL_DOCSET_DIGEST_DOMAIN_V1`][frankensearch_core::generation::CANONICAL_DOCSET_DIGEST_DOMAIN_V1].
//! Two domains over the same ordered identifiers produce two different digests.
//!
//! Copying the witness value across would therefore be a re-labelling: the
//! receipt would claim a canonical digest it does not hold, and every component
//! of a generation would agree with the vector component only if it made the
//! same mistake. Cross-engine agreement built on a shared error is worse than
//! no agreement, because it looks like corroboration. So the adapters take the
//! owner's ordered live document identifiers and recompute through
//! [`CanonicalDocsetV1`][frankensearch_core::generation::CanonicalDocsetV1],
//! and `adapter_docset_digest_is_canonical_not_fsvi_domain` pins that the two
//! digests genuinely differ.

use frankensearch_core::generation::{
    CanonicalDocsetV1, ExactComponentReceiptV1, GenerationAuthorityErrorV1,
    GenerationComponentReceiptV1, GenerationComponentRole,
};

use crate::FsviV2Witness;
use crate::native_hnsw::NativeHnswGenerationReceiptV2;

/// Decode a lowercase-hex SHA-256 carried as text by an engine receipt.
///
/// The ANN receipt stores its digests as `String` for artifact readability,
/// while the neutral receipt is typed `[u8; 32]`. A malformed value is a typed
/// error rather than a truncation or a zero fill: a silently zeroed digest
/// would be indistinguishable from the all-zero placeholder that
/// [`ExactComponentReceiptV1::validate`][frankensearch_core::generation::ExactComponentReceiptV1::validate]
/// exists to reject.
fn sha256_from_hex(
    label: &'static str,
    text: &str,
) -> Result<[u8; 32], GenerationAuthorityErrorV1> {
    let bytes = text.as_bytes();
    if bytes.len() != 64 || !bytes.iter().all(u8::is_ascii_hexdigit) {
        return Err(GenerationAuthorityErrorV1::InvalidField { field: label });
    }
    let mut digest = [0_u8; 32];
    for (index, slot) in digest.iter_mut().enumerate() {
        let pair = &text[index * 2..index * 2 + 2];
        *slot = u8::from_str_radix(pair, 16)
            .map_err(|_| GenerationAuthorityErrorV1::InvalidField { field: label })?;
    }
    Ok(digest)
}

/// Build the canonical docset digest from the owner's ordered live identifiers.
///
/// # Errors
///
/// Propagates [`CanonicalDocsetV1::from_ordered_live_documents`]'s typed error
/// for an empty identifier or a duplicate. A duplicate is rejected rather than
/// deduplicated, so a corrupted owner cannot be made to agree with a healthy
/// one.
fn canonical_docset_digest<I, S>(
    ordered_live_documents: I,
) -> Result<[u8; 32], GenerationAuthorityErrorV1>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    Ok(CanonicalDocsetV1::from_ordered_live_documents(ordered_live_documents)?.digest())
}

/// Derive the vector-component receipt from a validated FSVI v2 witness.
///
/// `ordered_live_documents` must be the owner's live document identifiers in
/// exact generation order — the same sequence the witness's own digest was
/// taken over. It is a parameter rather than something read back out of the
/// witness precisely because the witness only retains a digest, and a digest
/// cannot be re-domained.
///
/// # Errors
///
/// Returns [`GenerationAuthorityErrorV1`] when the identifiers are not a valid
/// canonical docset, or when the resulting receipt fails its own validation.
pub fn vector_component_receipt<I, S>(
    witness: &FsviV2Witness,
    ordered_live_documents: I,
    source_checkpoint: [u8; 32],
) -> Result<ExactComponentReceiptV1, GenerationAuthorityErrorV1>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let receipt = ExactComponentReceiptV1 {
        role: GenerationComponentRole::Vector,
        bytes: GenerationComponentReceiptV1 {
            byte_len: witness.byte_len,
            sha256: witness.whole_image_sha256,
        },
        docset_digest: canonical_docset_digest(ordered_live_documents)?,
        live_document_count: witness.live_count,
        source_checkpoint,
    };
    receipt.validate()?;
    Ok(receipt)
}

/// Derive the ANN-component receipt from a native HNSW generation receipt.
///
/// The component bytes are the FSHNSW graph file, not the FSVI image: the ANN
/// component's identity is its own sidecar, and binding it to the vector image
/// would make a rebuilt graph indistinguishable from an unchanged one.
///
/// `ordered_live_documents` carries the same requirement and the same reason as
/// [`vector_component_receipt`].
///
/// # Errors
///
/// Returns [`GenerationAuthorityErrorV1`] for a malformed hex digest, an
/// invalid canonical docset, or a receipt that fails its own validation.
pub fn ann_component_receipt<I, S>(
    receipt: &NativeHnswGenerationReceiptV2,
    ordered_live_documents: I,
    source_checkpoint: [u8; 32],
) -> Result<ExactComponentReceiptV1, GenerationAuthorityErrorV1>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let component = ExactComponentReceiptV1 {
        role: GenerationComponentRole::Ann,
        bytes: GenerationComponentReceiptV1 {
            byte_len: receipt.graph_byte_len,
            sha256: sha256_from_hex("graph_sha256", &receipt.graph_sha256)?,
        },
        docset_digest: canonical_docset_digest(ordered_live_documents)?,
        live_document_count: receipt.point_count,
        source_checkpoint,
    };
    component.validate()?;
    Ok(component)
}

#[cfg(test)]
mod tests {
    use super::canonical_docset_digest;
    use frankensearch_core::generation::CanonicalDocsetV1;
    use sha2::{Digest, Sha256};

    /// Mirror of the FSVI v2 ordered-live-docset digest, byte for byte with the
    /// production construction in `lib.rs`: domain length and bytes, then the
    /// live count as big-endian u64, then each identifier length-prefixed the
    /// same way. Reproduced here rather than called because the production
    /// routine reads a mapped image; the point is to obtain a genuine
    /// FSVI-domain digest to compare against, not to re-test that routine.
    fn fsvi_domain_digest(documents: &[&str]) -> [u8; 32] {
        const ORDERED_DOCSET_DIGEST_DOMAIN: &[u8] = b"frankensearch.fsvi-v2.ordered-live-docset.v1";
        let mut hasher = Sha256::new();
        hasher.update(
            u64::try_from(ORDERED_DOCSET_DIGEST_DOMAIN.len())
                .unwrap_or(u64::MAX)
                .to_be_bytes(),
        );
        hasher.update(ORDERED_DOCSET_DIGEST_DOMAIN);
        hasher.update((documents.len() as u64).to_be_bytes());
        for document in documents {
            hasher.update((document.len() as u64).to_be_bytes());
            hasher.update(document.as_bytes());
        }
        hasher.finalize().into()
    }

    /// THE LOAD-BEARING TEST for bd-cycdy.
    ///
    /// Both witnesses expose a field named `ordered_live_docset_digest`, and it
    /// is tempting to copy it straight into the neutral receipt. It is computed
    /// under a different domain, so the two digests over the SAME ordered
    /// identifiers must differ — and the adapter must emit the canonical one.
    /// Without this, a re-labelled receipt would claim a canonical digest it
    /// does not hold, and components would agree only by sharing the mistake.
    #[test]
    fn adapter_docset_digest_is_canonical_not_fsvi_domain() {
        let documents = ["doc-a", "doc-b", "doc-c"];

        let canonical = CanonicalDocsetV1::from_ordered_live_documents(documents)
            .expect("canonical docset")
            .digest();
        let fsvi = fsvi_domain_digest(&documents);

        assert_ne!(
            canonical, fsvi,
            "the canonical and FSVI-domain digests must differ over the same ordered \
             identifiers; if they ever coincide, copying the witness value would silently \
             become correct and this adapter's whole reason for existing would be invisible"
        );

        let emitted = canonical_docset_digest(documents).expect("adapter digest");
        assert_eq!(
            emitted, canonical,
            "the adapter must emit the CANONICAL digest"
        );
        assert_ne!(
            emitted, fsvi,
            "the adapter must never emit the FSVI-domain digest"
        );
    }

    /// Ordering is load-bearing: the canonical digest length-prefixes each
    /// identifier precisely so a reordering cannot collide with a healthy set.
    #[test]
    fn canonical_digest_is_order_sensitive() {
        let forward = canonical_docset_digest(["doc-a", "doc-b"]).expect("forward");
        let reversed = canonical_docset_digest(["doc-b", "doc-a"]).expect("reversed");
        assert_ne!(
            forward, reversed,
            "a reordered docset must not produce the same canonical digest"
        );
    }

    /// A duplicate identifier is rejected rather than deduplicated: collapsing
    /// one would let a corrupted owner agree with a healthy one.
    #[test]
    fn duplicate_identifiers_are_rejected_not_deduplicated() {
        assert!(
            canonical_docset_digest(["doc-a", "doc-a"]).is_err(),
            "a duplicate identifier must be a typed error"
        );
    }

    // -----------------------------------------------------------------------
    // The two PUBLIC adapters. The three tests above exercise only the private
    // canonical_docset_digest helper, so until here `vector_component_receipt`
    // and `ann_component_receipt` -- the functions this module exists for --
    // were never called by anything. An adapter mapping the wrong witness
    // field, emitting the wrong role, or failing validate() would have left
    // every test above green.
    // -----------------------------------------------------------------------

    use frankensearch_core::generation::{
        ComponentJoinErrorV1, ExactComponentReceiptV1, ExactGenerationComponentsV1,
        GenerationComponentReceiptV1, GenerationComponentRole,
    };

    use super::{ann_component_receipt, sha256_from_hex, vector_component_receipt};
    use crate::native_hnsw::{NativeHnswGenerationReceiptV2, NativeHnswParamsIdentityV1};
    use crate::{FsviV2Witness, Quantization};
    use frankensearch_core::generation::ArtifactGenerationIdentityV1;

    const DOCS: [&str; 3] = ["doc-a", "doc-b", "doc-c"];
    const CHECKPOINT: [u8; 32] = [0x5c; 32];

    /// Distinct byte markers per field, so a mis-mapped adapter produces a
    /// visibly wrong value rather than a coincidentally right one.
    const WITNESS_IMAGE_SHA: [u8; 32] = [0xA1; 32];
    const WITNESS_FSVI_DOCSET_DIGEST: [u8; 32] = [0xA2; 32];
    const WITNESS_CONTENT_DIGEST: [u8; 32] = [0xA3; 32];

    fn witness_fixture() -> FsviV2Witness {
        FsviV2Witness {
            schema_version: 1,
            fsvi_version: 2,
            byte_len: 8_192,
            whole_image_sha256: WITNESS_IMAGE_SHA,
            generation: ArtifactGenerationIdentityV1::new(7, [0x4d; 16])
                .expect("valid test generation"),
            identity_bundle_fingerprint: [0xB1; 32],
            space_fingerprint: [0xB2; 32],
            producer_fingerprint: [0xB3; 32],
            input_fingerprint: [0xB4; 32],
            storage_fingerprint: [0xB5; 32],
            generation_fingerprint: [0xB6; 32],
            // Deliberately NOT the canonical digest: the adapter must recompute.
            ordered_live_docset_digest: WITNESS_FSVI_DOCSET_DIGEST,
            vector_content_digest: WITNESS_CONTENT_DIGEST,
            dimension: 4,
            quantization: Quantization::F32,
            record_count: 5,
            live_count: DOCS.len() as u64,
            tombstone_count: 2,
        }
    }

    fn hex32(byte: u8) -> String {
        use std::fmt::Write as _;

        let mut text = String::with_capacity(64);
        for _ in 0..32 {
            write!(text, "{byte:02x}").expect("writing to a String cannot fail");
        }
        text
    }

    fn ann_receipt_fixture() -> NativeHnswGenerationReceiptV2 {
        NativeHnswGenerationReceiptV2 {
            schema_version: 2,
            artifact_generation: ArtifactGenerationIdentityV1::new(7, [0x4d; 16])
                .expect("valid test generation"),
            artifact_generation_fingerprint: hex32(0xC1),
            embedding_identity_fingerprint: hex32(0xC2),
            embedding_space_fingerprint: hex32(0xC3),
            embedding_producer_fingerprint: hex32(0xC4),
            embedding_input_fingerprint: hex32(0xC5),
            vector_storage_fingerprint: hex32(0xC6),
            vector_content_digest: hex32(0xC7),
            ordered_live_docset_digest: hex32(0xC8),
            // The FSVI image digest, which the ANN component must NOT bind to.
            fsvi_whole_image_sha256: hex32(0xA1),
            fsvi_physical_row_count: 5,
            graph_basename: "fast.fshnsw".to_owned(),
            // Deliberately different from the FSVI image's 8_192.
            graph_byte_len: 4_096,
            graph_sha256: hex32(0xD1),
            native_format_version: 1,
            params: NativeHnswParamsIdentityV1 {
                m: 16,
                m0: 32,
                ef_construction: 64,
                ef_search: 32,
            },
            seed: 42,
            point_count: DOCS.len() as u64,
            entry_point: Some(0),
            max_level: 3,
            payload_crc32: 0x1234_5678,
            header_crc32: 0x8765_4321,
            topology_sha256: hex32(0xD2),
            receipt_sha256: hex32(0xD3),
        }
    }

    /// The vector adapter maps from the WITNESS, field by field. Each assertion
    /// names a specific source, so a plausible-but-wrong mapping (content
    /// digest for image digest, `record_count` for `live_count`) fails here.
    #[test]
    fn the_vector_adapter_maps_every_field_from_the_witness() {
        let witness = witness_fixture();
        let receipt = vector_component_receipt(&witness, DOCS, CHECKPOINT)
            .expect("a valid witness produces a vector receipt");

        assert_eq!(receipt.role, GenerationComponentRole::Vector);
        assert_eq!(receipt.bytes.byte_len, witness.byte_len);
        assert_eq!(receipt.bytes.sha256, witness.whole_image_sha256);
        assert_ne!(
            receipt.bytes.sha256, WITNESS_CONTENT_DIGEST,
            "the component bytes are the whole IMAGE, not the vector content digest"
        );
        assert_eq!(
            receipt.live_document_count, witness.live_count,
            "live count, not record_count"
        );
        assert_ne!(receipt.live_document_count, witness.record_count);
        assert_eq!(receipt.source_checkpoint, CHECKPOINT);

        // The load-bearing mapping, restated against a real witness: the
        // adapter must recompute, never re-label.
        assert_eq!(
            receipt.docset_digest,
            CanonicalDocsetV1::from_ordered_live_documents(DOCS)
                .expect("canonical")
                .digest()
        );
        assert_ne!(
            receipt.docset_digest, witness.ordered_live_docset_digest,
            "re-labelling the witness's fsvi-domain digest would be the whole bug"
        );
    }

    /// The ANN adapter binds its OWN sidecar, not the FSVI image. Binding the
    /// image would make a rebuilt graph indistinguishable from an unchanged
    /// one, which is exactly what the ANN component exists to detect.
    #[test]
    fn the_ann_adapter_binds_the_graph_not_the_fsvi_image() {
        let ann = ann_receipt_fixture();
        let receipt = ann_component_receipt(&ann, DOCS, CHECKPOINT)
            .expect("a valid ANN receipt produces an ANN component");

        assert_eq!(receipt.role, GenerationComponentRole::Ann);
        assert_eq!(receipt.bytes.byte_len, ann.graph_byte_len);
        assert_eq!(
            receipt.bytes.sha256,
            sha256_from_hex("graph_sha256", &ann.graph_sha256).expect("fixture hex")
        );

        // The fixture makes the graph and the FSVI image differ in BOTH length
        // and digest, so an adapter that grabbed the image would be caught by
        // either assertion alone.
        assert_ne!(
            receipt.bytes.byte_len, 8_192,
            "that is the FSVI image length"
        );
        assert_ne!(
            receipt.bytes.sha256,
            sha256_from_hex("fsvi", &ann.fsvi_whole_image_sha256).expect("fixture hex"),
            "the ANN component must not bind the FSVI whole-image digest"
        );

        assert_eq!(receipt.live_document_count, ann.point_count);
        assert_eq!(receipt.source_checkpoint, CHECKPOINT);
        assert_ne!(
            receipt.docset_digest,
            sha256_from_hex("docset", &ann.ordered_live_docset_digest).expect("fixture hex"),
            "the ANN adapter must recompute the canonical digest too"
        );
    }

    fn metadata_component(
        docset_digest: [u8; 32],
        checkpoint: [u8; 32],
    ) -> ExactComponentReceiptV1 {
        ExactComponentReceiptV1 {
            role: GenerationComponentRole::Metadata,
            bytes: GenerationComponentReceiptV1 {
                byte_len: 512,
                sha256: [0xE1; 32],
            },
            docset_digest,
            live_document_count: DOCS.len() as u64,
            source_checkpoint: checkpoint,
        }
    }

    fn lexical_component(docset_digest: [u8; 32], checkpoint: [u8; 32]) -> ExactComponentReceiptV1 {
        ExactComponentReceiptV1 {
            role: GenerationComponentRole::Lexical,
            bytes: GenerationComponentReceiptV1 {
                byte_len: 1_024,
                sha256: [0xE2; 32],
            },
            docset_digest,
            live_document_count: DOCS.len() as u64,
            source_checkpoint: checkpoint,
        }
    }

    /// THE INTEGRATION THIS BEAD EXISTS FOR. The join was previously exercised
    /// only by receipts core constructed by hand; this admits ADAPTER-PRODUCED
    /// vector and ANN receipts, which is the thing that proves the adapters
    /// speak the join's language rather than merely compiling against its
    /// types.
    #[test]
    fn adapter_produced_receipts_admit_as_one_generation() {
        let vector =
            vector_component_receipt(&witness_fixture(), DOCS, CHECKPOINT).expect("vector receipt");
        let ann =
            ann_component_receipt(&ann_receipt_fixture(), DOCS, CHECKPOINT).expect("ann receipt");
        let canonical = vector.docset_digest;

        let admitted = ExactGenerationComponentsV1::admit(
            vector.clone(),
            lexical_component(canonical, CHECKPOINT),
            Some(ann.clone()),
            metadata_component(canonical, CHECKPOINT),
        )
        .expect("adapter-produced receipts describe one generation");

        assert_eq!(admitted.vector(), &vector);
        assert_eq!(admitted.ann(), Some(&ann));
        assert_eq!(admitted.docset_digest(), canonical);
        assert!(admitted.has_ann());

        // ANN is optional: dropping it still leaves an exact generation.
        ExactGenerationComponentsV1::admit(
            vector,
            lexical_component(canonical, CHECKPOINT),
            None,
            metadata_component(canonical, CHECKPOINT),
        )
        .expect("a generation without an ANN sidecar is still exact");
    }

    /// Drift controls, each mutation applied ALONE and rejected on the role
    /// that drifted. An adapter fed a different document set must not be able
    /// to pass itself off as agreeing with the anchor.
    #[test]
    fn an_adapter_fed_a_different_docset_rejects_on_its_own_role() {
        let anchor = vector_component_receipt(&witness_fixture(), DOCS, CHECKPOINT)
            .expect("anchor vector receipt");
        let canonical = anchor.docset_digest;

        // ANN built from a different ordered set, everything else identical.
        let drifted_ann = ann_component_receipt(
            &ann_receipt_fixture(),
            ["doc-a", "doc-c", "doc-b"],
            CHECKPOINT,
        )
        .expect("ann receipt over a reordered set");
        assert_ne!(drifted_ann.docset_digest, canonical);
        let observed = ExactGenerationComponentsV1::admit(
            anchor.clone(),
            lexical_component(canonical, CHECKPOINT),
            Some(drifted_ann),
            metadata_component(canonical, CHECKPOINT),
        );
        assert!(
            matches!(
                observed,
                Err(ComponentJoinErrorV1::DocsetDrift { role: "ann" })
            ),
            "observed {observed:?}"
        );

        // A vector anchor built from a different set moves the anchor itself,
        // so the OTHERS drift against it -- the first mandatory role checked
        // after the anchor is lexical.
        let drifted_anchor =
            vector_component_receipt(&witness_fixture(), ["doc-a", "doc-b"], CHECKPOINT)
                .expect("vector receipt over a shorter set");
        assert_ne!(drifted_anchor.docset_digest, canonical);
        let observed = ExactGenerationComponentsV1::admit(
            drifted_anchor,
            lexical_component(canonical, CHECKPOINT),
            None,
            metadata_component(canonical, CHECKPOINT),
        );
        assert!(
            matches!(
                observed,
                Err(ComponentJoinErrorV1::DocsetDrift { role: "lexical" })
            ),
            "the anchor defines truth, so a drifted anchor makes the others wrong: {observed:?}"
        );
    }

    /// A checkpoint that disagrees rejects on the drifting role, with the
    /// document set held fixed so the checkpoint is the only variable.
    #[test]
    fn an_adapter_given_a_different_checkpoint_rejects_on_its_own_role() {
        let vector =
            vector_component_receipt(&witness_fixture(), DOCS, CHECKPOINT).expect("vector receipt");
        let canonical = vector.docset_digest;
        let other_checkpoint = [0x5d; 32];

        let ann = ann_component_receipt(&ann_receipt_fixture(), DOCS, other_checkpoint)
            .expect("ann receipt at another checkpoint");
        assert_eq!(
            ann.docset_digest, canonical,
            "documents must be identical, or this is docset drift instead"
        );

        let observed = ExactGenerationComponentsV1::admit(
            vector,
            lexical_component(canonical, CHECKPOINT),
            Some(ann),
            metadata_component(canonical, CHECKPOINT),
        );
        assert!(
            matches!(
                observed,
                Err(ComponentJoinErrorV1::CheckpointDrift { role: "ann" })
            ),
            "observed {observed:?}"
        );
    }

    /// `sha256_from_hex` had no test. A silently zeroed or truncated digest
    /// would be indistinguishable from the all-zero placeholder that
    /// `validate()` exists to reject, so every malformed shape must be a typed
    /// error instead.
    #[test]
    fn malformed_hex_digests_are_typed_errors_not_silent_zeros() {
        let valid = hex32(0xD1);
        sha256_from_hex("control", &valid).expect("64 lowercase hex digits decode");

        for (label, text) in [
            ("empty", String::new()),
            ("too short", valid[..62].to_owned()),
            ("too long", format!("{valid}00")),
            ("non-hex", "z".repeat(64)),
            ("spaces", " ".repeat(64)),
        ] {
            assert!(
                sha256_from_hex("graph_sha256", &text).is_err(),
                "{label} must be a typed error"
            );
        }

        // Uppercase is accepted -- is_ascii_hexdigit admits it and
        // from_str_radix decodes it -- and decodes to the same bytes. Pinned so
        // the behaviour is deliberate rather than incidental.
        assert_eq!(
            sha256_from_hex("upper", &valid.to_uppercase()).expect("uppercase decodes"),
            sha256_from_hex("lower", &valid).expect("lowercase decodes")
        );

        // A well-formed all-zero digest decodes, and is then caught downstream
        // by the receipt's own validation rather than here.
        let zeros = "0".repeat(64);
        assert_eq!(
            sha256_from_hex("zeros", &zeros).expect("all-zero hex is well formed"),
            [0_u8; 32]
        );
        let mut ann = ann_receipt_fixture();
        ann.graph_sha256 = zeros;
        assert!(
            ann_component_receipt(&ann, DOCS, CHECKPOINT).is_err(),
            "an all-zero graph digest cannot identify real bytes"
        );
    }

    /// An empty or duplicated document set never yields a receipt from either
    /// adapter, so a corrupted owner cannot enter the join at all.
    #[test]
    fn neither_adapter_produces_a_receipt_from_an_invalid_docset() {
        let witness = witness_fixture();
        let ann = ann_receipt_fixture();

        assert!(vector_component_receipt(&witness, ["doc-a", "doc-a"], CHECKPOINT).is_err());
        assert!(ann_component_receipt(&ann, ["doc-a", "doc-a"], CHECKPOINT).is_err());
        assert!(vector_component_receipt(&witness, ["doc-a", ""], CHECKPOINT).is_err());
        assert!(ann_component_receipt(&ann, ["", "doc-b"], CHECKPOINT).is_err());

        // Control: the same adapters accept the valid set, so the refusals are
        // attributable to the docset and not to the fixtures.
        vector_component_receipt(&witness, DOCS, CHECKPOINT).expect("valid docset");
        ann_component_receipt(&ann, DOCS, CHECKPOINT).expect("valid docset");
    }

    /// A zero-placeholder checkpoint is refused at construction. Without this
    /// an adapter could emit a receipt that only fails later, inside `admit()`,
    /// as an anonymous `InvalidComponent`.
    #[test]
    fn a_zero_placeholder_checkpoint_never_leaves_an_adapter() {
        assert!(vector_component_receipt(&witness_fixture(), DOCS, [0; 32]).is_err());
        assert!(ann_component_receipt(&ann_receipt_fixture(), DOCS, [0; 32]).is_err());
    }
}
