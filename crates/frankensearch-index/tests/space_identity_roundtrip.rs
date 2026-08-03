//! Real FSVI v2 identity roundtrip through the production `create_v2` writer
//! (bd-9xuj T2-C1 rebuild).
//!
//! The producer-conformance proof here is REAL, not synthetic: the artifact
//! is written by [`VectorIndex::create_v2`] — the actual production FSVI v2
//! writer path — admitted through [`VectorIndex::open_admitted_v2`] (the only
//! way a v2 artifact is legitimately opened), and the query-side verifiers
//! join against the space fingerprint decoded from that artifact's own header
//! bytes.
//!
//! Note on the `LegacyUnidentified` boundary: the only production call sites
//! of `create_v2` today are the two staging writes in the fusion crate's
//! `RefreshWorker::stage_identity_bound_generation` (one per tier), whose
//! output is staged and non-canonical until publication;
//! `TwoTierIndexBuilder::finish` still routes through the legacy v1
//! constructors with `identity_v2: None`, so every canonical production
//! index is v1 and identity-less. Such artifacts never reach the verifiers
//! exercised below — the seams route them as typed `LegacyUnidentified`
//! reindex instead. This test proves the identity-bearing path those legacy
//! artifacts must be reindexed onto.

use std::path::{Path, PathBuf};

use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1, QuantizationFormat,
};
use frankensearch_core::{BoundQueryEmbedding, SearchError, SpaceIdentityAdmission};
use frankensearch_index::{FsviV2IdentityBinding, VectorIndex};

fn temp_index_path(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("frankensearch_space_identity_roundtrip");
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir.join(format!("{name}.fsvi"))
}

fn cleanup(path: &Path) {
    let _ = std::fs::remove_file(path);
    let wal_path = path.with_extension("fsvi.wal");
    let _ = std::fs::remove_file(&wal_path);
}

/// Lowercase hex encoding of raw fingerprint bytes (test-local; the crate's
/// internal encoder is deliberately not public).
fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        })
}

fn normalized(dim: usize, seed: f32) -> Vec<f32> {
    let raw: Vec<f32> = (0..dim).map(|i| seed + i as f32).collect();
    let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    raw.iter().map(|x| x / norm).collect()
}

/// Identity bundle for `model_id` with canonical FSVI v2 storage.
fn fsvi_v2_identity(model_id: &str, dimension: u32) -> EmbeddingIdentityBundleV1 {
    let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(model_id, dimension);
    "fsvi-v2".clone_into(&mut identity.storage.format);
    identity.storage.quantization = QuantizationFormat::F16;
    "little-endian".clone_into(&mut identity.storage.endianness);
    identity
}

#[test]
fn create_v2_roundtrip_joins_and_certifies_through_real_artifact() -> Result<(), String> {
    let path = temp_index_path("create_v2_conformance");
    cleanup(&path);
    let dim = 8_usize;

    // WRITE: the actual production v2 writer path.
    let artifact_identity = fsvi_v2_identity("roundtrip-model", 8);
    let generation =
        ArtifactGenerationIdentityV1::new(21, [0x7a; 16]).expect("valid test generation");
    let binding = FsviV2IdentityBinding::new(
        generation,
        artifact_identity
            .freeze()
            .expect("freeze artifact identity"),
    )
    .expect("valid FSVI v2 identity binding");
    let mut writer = VectorIndex::create_v2(&path, binding.clone()).expect("create_v2 writer");
    for row in 0..4_u8 {
        writer
            .write_record(&format!("doc-{row}"), &normalized(dim, f32::from(row + 1)))
            .expect("write v2 row");
    }
    writer.finish().expect("finish v2 artifact");

    // ADMIT: the only legitimate open path for a v2 artifact.
    let admitted =
        VectorIndex::open_admitted_v2(&path, &binding).expect("exact admission of the v2 artifact");
    let header = admitted.identity_v2();
    let artifact_space_hex = hex(&header.space_fingerprint);
    assert_eq!(
        artifact_space_hex,
        artifact_identity.space.fingerprint(),
        "artifact header must preserve the core-side space fingerprint bit-for-bit"
    );

    // QUERY SIDE: same model, in-memory storage — the seam's legitimate
    // storage difference — bound and validated at bind time.
    let query_identity = EmbeddingIdentityBundleV1::explicit_test_model("roundtrip-model", 8);
    assert_ne!(
        query_identity.fingerprint(),
        artifact_identity.fingerprint(),
        "full-bundle fingerprints legitimately differ across the seam (storage)"
    );
    let bound = BoundQueryEmbedding::new(normalized(dim, 1.0), query_identity.clone())
        .expect("bind query embedding");

    // JOIN: space-scoped verification against the REAL artifact fingerprint.
    bound
        .verify_space_identity(&artifact_space_hex, "quality")
        .expect("same space must join against the artifact's header fingerprint");

    // CERTIFY: the full admission law against the artifact's frozen bundle.
    let admission = bound
        .verify_producer_conformance(&binding.frozen_identity().identity, "quality")
        .expect("same producer must be admitted");
    assert_eq!(admission, SpaceIdentityAdmission::SameProducer);

    // A conformance-certified foreign producer: same space and golden
    // certificate, different attested backend — admitted, telemetry-typed.
    let mut foreign = query_identity;
    foreign.producer.backend = "alternate-conformant-backend".to_owned();
    let foreign_bound = BoundQueryEmbedding::new(normalized(dim, 2.0), foreign)
        .expect("bind foreign-producer embedding");
    let admission = foreign_bound
        .verify_producer_conformance(&binding.frozen_identity().identity, "quality")
        .expect("certified foreign producer must be admitted");
    assert_eq!(admission.code(), "conformance_compatible_producer");
    let rendered = format!("{admission:?}");
    let SpaceIdentityAdmission::ConformanceCompatibleProducer {
        expected_producer_fingerprint,
        ..
    } = admission
    else {
        return Err(format!(
            "expected ConformanceCompatibleProducer, got {rendered}"
        ));
    };
    assert_eq!(
        expected_producer_fingerprint,
        binding.frozen_identity().identity.producer.fingerprint(),
        "telemetry must carry the artifact producer's real fingerprint"
    );

    // REJECT: a wrong model at the same dimension, against the same real
    // artifact fingerprint — the case dimension checks can never catch.
    let wrong = BoundQueryEmbedding::new(
        normalized(dim, 3.0),
        EmbeddingIdentityBundleV1::explicit_test_model("other-model", 8),
    )
    .expect("bind wrong-space embedding");
    let error = wrong
        .verify_space_identity(&artifact_space_hex, "quality")
        .expect_err("wrong space must reject against the artifact fingerprint");
    let rendered = format!("{error:?}");
    let SearchError::InvalidConfig { field, .. } = error else {
        return Err(format!("expected InvalidConfig, got {rendered}"));
    };
    assert_eq!(field, "query_embedding.quality.space_identity");

    cleanup(&path);
    Ok(())
}
