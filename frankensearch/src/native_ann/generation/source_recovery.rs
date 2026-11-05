//! Authenticated retained source for replacing damaged derived components.
//!
//! This route admits the selected metadata and complete Quill source cohort
//! without opening a vector or ANN image. It deliberately provides no search
//! interface. A caller must build and admit a new complete generation before
//! publishing against the retained physical authority pair.

use std::collections::BTreeMap;
use std::sync::Arc;

use frankensearch_core::generation::{
    ActivationManifest, AuthorityRefV1, EmbedderTierTag, ExactComponentReceiptV1,
    GenerationComponentRole, GenerationManifest, InvariantKind, SourceCheckpointV1,
    validate_manifest,
};
use frankensearch_index::FsviV2IdentityBinding;
use frankensearch_index::generation_root::generation_reader::{
    OpenedGenerationSourceV1, component_object_name_v1,
};
use frankensearch_quill::{
    DEFAULT_SCHEMA, KeeperSnapshot, QuillConfig, QuillIndexError, QuillSnapshotIndex,
};
use sha2::{Digest as _, Sha256};

use super::{NativeGenerationTopology, checkpoint, lower_hex, refused};
use crate::{Cx, IndexableDocument, SearchResult};

/// Authenticated source from an explicitly selected, possibly damaged generation.
///
/// The fixed-authority capability authenticates the exact metadata and lexical
/// images. Admission additionally proves their source checkpoint, complete
/// ordered live document set, counts and caller-selected frozen identities.
/// The vector and ANN components are not opened or certified. In particular,
/// this handle cannot become a [`super::NativeGenerationSnapshot`], issue a
/// query, or silently repair an ordinary snapshot open.
///
/// A repair exports bounded source documents, builds new independently admitted
/// components, and publishes against [`Self::retained_source`]'s captured pair.
/// Keeping this value retains the original immutable source even if another
/// writer publishes while the replacement is being built.
pub struct NativeGenerationSource {
    authority: Arc<OpenedGenerationSourceV1>,
    components: SourceComponents,
    primary: FsviV2IdentityBinding,
    quality: Option<FsviV2IdentityBinding>,
    tier: EmbedderTierTag,
}

impl NativeGenerationSource {
    /// Admit the source selected by an explicit recovery authority capability.
    ///
    /// The bindings must come from the caller's expected selection; no producer
    /// is inferred from damaged vector bytes or substituted from metadata.
    /// `quality` is present only for a complete fast-plus-quality topology.
    /// Image admission is bounded independently from later source extraction.
    ///
    /// # Errors
    /// Refuses identity, generation, topology, source or metadata inconsistencies,
    /// malformed lexical images, image budget overflow and cancellation.
    pub fn admit(
        cx: &Cx,
        authority: Arc<OpenedGenerationSourceV1>,
        primary: &FsviV2IdentityBinding,
        quality: Option<&FsviV2IdentityBinding>,
        tier: EmbedderTierTag,
        max_image_bytes: usize,
    ) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.generation.recovery_admit")?;
        if authority.retained_image_bytes() > u64::try_from(max_image_bytes).unwrap_or(u64::MAX) {
            return Err(refused("source-recovery", "encoded-image-limit"));
        }
        let components = admit_source_components(
            cx,
            authority.manifest(),
            primary,
            quality,
            tier,
            authority.lexical_bytes(),
            authority.metadata_bytes(),
            max_image_bytes,
        )?;
        checkpoint(cx, "native_ann.generation.recovery_ready")?;
        Ok(Self {
            authority,
            components,
            primary: primary.clone(),
            quality: quality.cloned(),
            tier,
        })
    }

    /// Original source capability, including the exact publication expectation.
    #[must_use]
    pub const fn retained_source(&self) -> &Arc<OpenedGenerationSourceV1> {
        &self.authority
    }

    /// Exact authority selected for this recovery.
    #[must_use]
    pub fn authority(&self) -> AuthorityRefV1 {
        self.authority.head().authority
    }

    /// Authenticated intended topology; this is not a claim of healthy vectors.
    #[must_use]
    pub const fn topology(&self) -> NativeGenerationTopology {
        if self.quality.is_some() {
            NativeGenerationTopology::FullProgressive
        } else {
            match self.tier {
                EmbedderTierTag::Fast => NativeGenerationTopology::FastOnly,
                EmbedderTierTag::Quality => NativeGenerationTopology::QualityOnly,
            }
        }
    }

    /// Primary tier selected by the caller and authenticated metadata.
    #[must_use]
    pub const fn tier(&self) -> EmbedderTierTag {
        self.tier
    }

    /// Complete authenticated metadata of the damaged selection.
    /// Its vector/ANN descriptions are historical claims, not admitted images.
    #[must_use]
    pub const fn metadata(&self) -> &GenerationManifest {
        &self.components.metadata
    }

    /// Exact expected primary identity and original artifact generation.
    #[must_use]
    pub const fn binding(&self) -> &FsviV2IdentityBinding {
        &self.primary
    }

    /// Independent expected quality binding for a complete two-tier generation.
    #[must_use]
    pub const fn quality_binding(&self) -> Option<&FsviV2IdentityBinding> {
        self.quality.as_ref()
    }

    /// Recover all live documents in authenticated lexical generation order.
    ///
    /// The byte limit covers the aggregate stored IDs, content, titles and
    /// canonical metadata payloads before decoding or allocating those fields.
    /// Holes and tombstones are omitted; the complete admitted live count must
    /// be recovered. No partial source collection is returned on failure.
    ///
    /// # Errors
    /// Refuses malformed or missing stored fields, noncanonical metadata,
    /// count mismatches, source budget overflow, allocation failure and cancellation.
    pub fn source_documents(
        &self,
        cx: &Cx,
        max_source_bytes: usize,
    ) -> SearchResult<Vec<IndexableDocument>> {
        self.components.source_documents(cx, max_source_bytes)
    }
}

struct SourceComponents {
    metadata: GenerationManifest,
    lexical: QuillSnapshotIndex,
}

#[allow(clippy::too_many_arguments)]
fn admit_source_components(
    cx: &Cx,
    activation: &ActivationManifest,
    primary: &FsviV2IdentityBinding,
    quality: Option<&FsviV2IdentityBinding>,
    tier: EmbedderTierTag,
    lexical_bytes: Arc<[u8]>,
    metadata_bytes: Arc<[u8]>,
    max_image_bytes: usize,
) -> SearchResult<SourceComponents> {
    checkpoint(cx, "native_ann.generation.recovery_components")?;
    if activation.generation() != primary.generation()
        || quality.is_some_and(|binding| binding.generation() != activation.generation())
    {
        return Err(refused("source-recovery", "generation-mismatch"));
    }
    if let Some(quality) = quality {
        let primary_identity = &primary.frozen_identity().identity;
        let quality_identity = &quality.frozen_identity().identity;
        if tier != EmbedderTierTag::Fast
            || primary_identity.input.doc_id_semantics != quality_identity.input.doc_id_semantics
            || primary_identity.space.kind != quality_identity.space.kind
        {
            return Err(refused("source-recovery", "incompatible-tier-contracts"));
        }
    }
    let total = activation
        .canonical_bytes()
        .len()
        .checked_add(lexical_bytes.len())
        .and_then(|total| total.checked_add(metadata_bytes.len()));
    if total.is_none_or(|total| total > max_image_bytes) {
        return Err(refused("source-recovery", "encoded-image-limit"));
    }
    let declared = activation.components();
    for (image, receipt) in [
        (&lexical_bytes, declared.lexical),
        (&metadata_bytes, declared.metadata),
    ] {
        let mut digest = Sha256::new();
        for chunk in image.chunks(1024 * 1024) {
            checkpoint(cx, "native_ann.generation.recovery_hash")?;
            digest.update(chunk);
        }
        if u64::try_from(image.len()).ok() != Some(receipt.byte_len)
            || <[u8; 32]>::from(digest.finalize()) != receipt.sha256
        {
            return Err(refused("source-recovery", "component-image-mismatch"));
        }
    }
    checkpoint(cx, "native_ann.generation.recovery_metadata")?;
    let metadata: GenerationManifest = serde_json::from_slice(&metadata_bytes)
        .map_err(|_| refused("source-recovery", "invalid-metadata-json"))?;
    let tier_count = if quality.is_some() { 2 } else { 1 };
    if metadata.vector_artifacts.len() != tier_count
        || metadata.lexical_artifacts.len() != 1
        || metadata.embedders.len() != tier_count
        || !metadata.repair_descriptors.is_empty()
        || !validate_manifest(&metadata).is_valid()
        || metadata.generation_id != activation.generation().fingerprint()
    {
        return Err(refused("source-recovery", "invalid-metadata-contract"));
    }
    let lexical_artifact = &metadata.lexical_artifacts[0];
    if lexical_artifact.document_count != metadata.total_documents
        || lexical_artifact.size_bytes != declared.lexical.byte_len
        || lexical_artifact.checksum != lower_hex(&declared.lexical.sha256)
        || lexical_artifact.path
            != component_object_name_v1(GenerationComponentRole::Lexical, declared.lexical.sha256)
    {
        return Err(refused("source-recovery", "lexical-metadata-mismatch"));
    }
    let mut physical_rows = 0u64;
    for (position, (binding, expected_tier)) in std::iter::once((primary, tier))
        .chain(quality.map(|binding| (binding, EmbedderTierTag::Quality)))
        .enumerate()
    {
        let artifact = &metadata.vector_artifacts[position];
        let name = match expected_tier {
            EmbedderTierTag::Fast => "fast",
            EmbedderTierTag::Quality => "quality",
        };
        if artifact.embedder_tier != expected_tier
            || metadata.embedders.get(name) != Some(binding.frozen_identity())
            || usize::try_from(artifact.dimension).ok() != Some(binding.dimension())
            || artifact.vector_count < metadata.total_documents
            || (quality.is_none()
                && (artifact.size_bytes != declared.vector.byte_len
                    || artifact.checksum != lower_hex(&declared.vector.sha256)))
        {
            return Err(refused("source-recovery", "expected-identity-mismatch"));
        }
        physical_rows = physical_rows
            .checked_add(artifact.vector_count)
            .ok_or_else(|| refused("source-recovery", "record-count-overflow"))?;
    }
    for invariant in &metadata.activation_invariants {
        match &invariant.kind {
            InvariantKind::AllArtifactsVerified | InvariantKind::EmbedderRevisionMatch => {}
            InvariantKind::VectorCountConsistency { expected_total }
                if *expected_total == physical_rows => {}
            _ => {
                return Err(refused(
                    "source-recovery",
                    "unsupported-activation-predicate",
                ));
            }
        }
    }
    let source = SourceCheckpointV1::derive(&metadata.commit_range);
    if source.to_bytes() != activation.source_checkpoint_sha256() {
        return Err(refused("source-recovery", "source-checkpoint-mismatch"));
    }
    let receipt = ExactComponentReceiptV1 {
        role: GenerationComponentRole::Lexical,
        bytes: declared.lexical,
        docset_digest: activation.document_set_sha256(),
        live_document_count: metadata.total_documents,
        source_checkpoint: source.to_bytes(),
    };
    // This independently enumerates every live IDMAP row and checks the exact
    // ordered docset and count. A source subset cannot become repair authority.
    let admitted = KeeperSnapshot::from_exact_image(
        cx,
        lexical_bytes,
        DEFAULT_SCHEMA,
        source,
        &receipt,
        max_image_bytes,
    );
    checkpoint(cx, "native_ann.generation.recovery_lexical")?;
    let lexical = admitted.map_err(|_| refused("source-recovery", "lexical-admission-failed"))?;
    let reader = QuillSnapshotIndex::new(cx, lexical, QuillConfig::default());
    checkpoint(cx, "native_ann.generation.recovery_reader")?;
    Ok(SourceComponents {
        metadata,
        lexical: reader.map_err(|_| refused("source-recovery", "lexical-reader-failed"))?,
    })
}

impl SourceComponents {
    fn source_documents(
        &self,
        cx: &Cx,
        max_source_bytes: usize,
    ) -> SearchResult<Vec<IndexableDocument>> {
        checkpoint(cx, "native_ann.generation.recovery_source")?;
        let snapshot = self
            .lexical
            .snapshot()
            .map_err(|_| refused("source-recovery", "lexical-snapshot-failed"))?;
        let count = usize::try_from(self.metadata.total_documents)
            .map_err(|_| refused("source-recovery", "live-count-overflow"))?;
        if snapshot.doc_count() != self.metadata.total_documents || count > max_source_bytes {
            return Err(refused("source-recovery", "stored-payload-limit"));
        }
        let mut fields = [0; 4];
        for (field, name) in fields
            .iter_mut()
            .zip(["id", "content", "title", "metadata_json"])
        {
            *field = DEFAULT_SCHEMA
                .fields
                .iter()
                .find(|field| field.name == name && field.stored)
                .ok_or_else(|| refused("source-recovery", "unsupported-schema"))?
                .id;
        }
        let mut documents = Vec::new();
        documents
            .try_reserve_exact(count)
            .map_err(|_| refused("source-recovery", "source-allocation-failed"))?;
        let mut remaining = max_source_bytes;
        for segment in snapshot.segments() {
            let manifest = segment.manifest();
            for global_id in manifest.docid_lo..manifest.docid_hi {
                checkpoint(cx, "native_ann.generation.recovery_source_row")?;
                let global_id = u32::try_from(global_id)
                    .map_err(|_| refused("source-recovery", "invalid-document-id"))?;
                let Some(id) = segment.materialize_document_id(global_id) else {
                    continue;
                };
                let mut values: [Vec<u8>; 4] = std::array::from_fn(|_| Vec::new());
                for (value, field) in values.iter_mut().zip(fields) {
                    checkpoint(cx, "native_ann.generation.recovery_source_field")?;
                    let bytes = self
                        .lexical
                        .stored_field_value_bounded(field, global_id, remaining)
                        .map_err(|error| match error {
                            QuillIndexError::StoredFieldBudgetExceeded { .. } => {
                                refused("source-recovery", "stored-payload-limit")
                            }
                            _ => refused("source-recovery", "invalid-stored-field"),
                        })?
                        .ok_or_else(|| refused("source-recovery", "missing-stored-field"))?;
                    remaining = remaining
                        .checked_sub(bytes.len())
                        .ok_or_else(|| refused("source-recovery", "stored-payload-limit"))?;
                    *value = bytes;
                }
                let [stored_id, content, title, metadata_json] = values;
                let stored_id = String::from_utf8(stored_id)
                    .map_err(|_| refused("source-recovery", "invalid-stored-utf8"))?;
                if stored_id != id.as_str() {
                    return Err(refused("source-recovery", "stored-id-mismatch"));
                }
                let content = String::from_utf8(content)
                    .map_err(|_| refused("source-recovery", "invalid-stored-utf8"))?;
                let title = String::from_utf8(title)
                    .map_err(|_| refused("source-recovery", "invalid-stored-utf8"))?;
                let metadata: BTreeMap<String, String> = serde_json::from_slice(&metadata_json)
                    .map_err(|_| refused("source-recovery", "invalid-source-metadata"))?;
                if serde_json::to_vec(&metadata)
                    .map_err(|_| refused("source-recovery", "invalid-source-metadata"))?
                    != metadata_json
                {
                    return Err(refused("source-recovery", "noncanonical-source-metadata"));
                }
                documents.push(IndexableDocument {
                    id: stored_id,
                    content,
                    title: (!title.is_empty()).then_some(title),
                    metadata: metadata.into_iter().collect(),
                });
            }
        }
        checkpoint(cx, "native_ann.generation.recovery_source_complete")?;
        if documents.len() != count {
            return Err(refused("source-recovery", "live-count-mismatch"));
        }
        Ok(documents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use frankensearch_core::generation::{
        ActivationManifestV2, ArtifactGenerationIdentityV1, CanonicalDocsetV1, CommitRange,
        EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1, EmbeddingSpaceKindV1,
        GenerationAuthorityActionV1, GenerationComponentReceiptV1, compute_manifest_hash,
    };
    use frankensearch_core::traits::{ModelCategory, SearchFuture};

    use crate::native_ann::builder::{
        NativeBuildPrecision, NativeBuildRetrieval, NativeBuiltExactGeneration, NativeIndexBuilder,
    };
    use crate::{Embedder, GenerationCandidateV1, SearchError};

    const BUDGET: usize = 16 * 1024 * 1024;

    struct Provider {
        identity: EmbeddingIdentityBundleV1,
        calls: AtomicUsize,
    }

    impl Provider {
        fn new(name: &str, dimension: u32) -> Self {
            let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(name, dimension);
            identity.space.kind = EmbeddingSpaceKindV1::Semantic;
            identity.space.hash_control = None;
            identity.space.artifact_manifest_fingerprint = "a".repeat(64);
            identity.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
                role: "weights".to_owned(),
                sha256: "b".repeat(64),
                size: 1,
            }];
            identity.producer.space_fingerprint = identity.space.fingerprint();
            identity
                .validate()
                .expect("valid synthetic semantic identity");
            Self {
                identity,
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl Embedder for Provider {
        fn embed<'a>(&'a self, _: &'a Cx, _: &'a str) -> SearchFuture<'a, Vec<f32>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                let mut vector = vec![0.0; self.dimension()];
                vector[0] = 1.0;
                Ok(vector)
            })
        }

        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(&self.identity)
        }

        fn dimension(&self) -> usize {
            usize::try_from(self.identity.space.dimension).unwrap()
        }

        fn id(&self) -> &str {
            &self.identity.space.logical_model_id
        }

        fn model_name(&self) -> &str {
            self.id()
        }

        fn is_semantic(&self) -> bool {
            true
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::TransformerEmbedder
        }
    }

    struct Fixture {
        _directory: tempfile::TempDir,
        built: NativeBuiltExactGeneration,
        candidate: GenerationCandidateV1,
        primary: Arc<Provider>,
        quality: Arc<Provider>,
        tier: EmbedderTierTag,
    }

    impl Fixture {
        fn admit(&self, cx: &Cx, budget: usize) -> SearchResult<SourceComponents> {
            self.admit_images(
                cx,
                self.candidate.manifest(),
                self.built
                    .component_bytes(GenerationComponentRole::Lexical)
                    .unwrap(),
                self.built
                    .component_bytes(GenerationComponentRole::Metadata)
                    .unwrap(),
                budget,
            )
        }

        fn admit_images(
            &self,
            cx: &Cx,
            manifest: &ActivationManifest,
            lexical: Arc<[u8]>,
            metadata: Arc<[u8]>,
            budget: usize,
        ) -> SearchResult<SourceComponents> {
            admit_source_components(
                cx,
                manifest,
                self.built.binding(),
                self.built.quality_binding(),
                self.tier,
                lexical,
                metadata,
                budget,
            )
        }
    }

    async fn fixture(cx: &Cx, tier: EmbedderTierTag, full: bool) -> Fixture {
        let directory = tempfile::tempdir().unwrap();
        let primary = Arc::new(Provider::new("source-primary", 2));
        let quality = Arc::new(Provider::new("source-quality", 3));
        let mut builder = NativeIndexBuilder::new(
            directory.path().join("build"),
            ArtifactGenerationIdentityV1::new(7, [0x31; 16]).unwrap(),
            primary.clone(),
        )
        .unwrap()
        .with_fast_storage(NativeBuildPrecision::F16, NativeBuildRetrieval::Exact)
        .add_document(
            IndexableDocument::new("alpha", "recover α unchanged")
                .with_title("Retained title")
                .with_metadata("z", "last")
                .with_metadata("a", "first"),
        )
        .add_document(IndexableDocument::new("beta", "another document"));
        if full {
            builder = builder
                .with_quality_embedder(quality.clone())
                .unwrap()
                .with_quality_storage(NativeBuildPrecision::F32, NativeBuildRetrieval::Exact)
                .unwrap();
        }
        let source = CommitRange { low: 3, high: 9 };
        let built = if full {
            builder
                .build_progressive_generation(cx, source, BUDGET)
                .await
        } else {
            builder
                .build_exact_generation(cx, tier, source, BUDGET)
                .await
        }
        .expect("complete source fixture");
        let candidate = built
            .candidate(
                [0x32; 16],
                None,
                GenerationAuthorityActionV1::Activate,
                [0x33; 32],
            )
            .unwrap();
        primary.calls.store(0, Ordering::SeqCst);
        quality.calls.store(0, Ordering::SeqCst);
        Fixture {
            _directory: directory,
            built,
            candidate,
            primary,
            quality,
            tier,
        }
    }

    fn reseal(
        fixture: &Fixture,
        metadata: &[u8],
        source: [u8; 32],
        docset: [u8; 32],
    ) -> ActivationManifest {
        let mut components = fixture.candidate.manifest().components();
        components.metadata = GenerationComponentReceiptV1 {
            byte_len: u64::try_from(metadata.len()).unwrap(),
            sha256: Sha256::digest(metadata).into(),
        };
        ActivationManifest::V2(
            ActivationManifestV2::new(
                1,
                None,
                GenerationAuthorityActionV1::Activate,
                fixture.built.binding().generation(),
                [0x33; 32],
                source,
                docset,
                components,
            )
            .unwrap(),
        )
    }

    #[test]
    fn source_admission_recovers_every_field_for_each_supported_topology_without_providers() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (tier, full) in [
                (EmbedderTierTag::Fast, false),
                (EmbedderTierTag::Quality, false),
                (EmbedderTierTag::Fast, true),
            ] {
                let fixture = fixture(&cx, tier, full).await;
                let images = fixture.candidate.manifest().components();
                let exact_budget = fixture.candidate.manifest_bytes().len()
                    + usize::try_from(images.lexical.byte_len + images.metadata.byte_len).unwrap();
                let admitted = fixture.admit(&cx, exact_budget).unwrap();
                assert!(fixture.admit(&cx, exact_budget - 1).is_err());
                let documents = admitted.source_documents(&cx, BUDGET).unwrap();
                assert_eq!(documents.len(), 2);
                let alpha = documents
                    .iter()
                    .find(|document| document.id == "alpha")
                    .unwrap();
                let beta = documents
                    .iter()
                    .find(|document| document.id == "beta")
                    .unwrap();
                assert_eq!(alpha.content, "recover α unchanged");
                assert_eq!(alpha.title.as_deref(), Some("Retained title"));
                assert_eq!(alpha.metadata["a"], "first");
                assert_eq!(alpha.metadata["z"], "last");
                assert_eq!(beta.content, "another document");
                assert!(beta.title.is_none());
                assert!(beta.metadata.is_empty());
                assert_eq!(
                    CanonicalDocsetV1::from_ordered_live_documents(
                        documents.iter().map(|document| document.id.as_str()),
                    )
                    .unwrap()
                    .digest(),
                    fixture.candidate.manifest().document_set_sha256(),
                );
                let exact_source_bytes: usize = documents
                    .iter()
                    .map(|document| {
                        let metadata: BTreeMap<_, _> = document.metadata.iter().collect();
                        document.id.len()
                            + document.content.len()
                            + document.title.as_ref().map_or(0, String::len)
                            + serde_json::to_vec(&metadata).unwrap().len()
                    })
                    .sum();
                assert_eq!(
                    admitted
                        .source_documents(&cx, exact_source_bytes)
                        .unwrap()
                        .len(),
                    2
                );
                assert!(
                    admitted
                        .source_documents(&cx, exact_source_bytes - 1)
                        .is_err()
                );
                assert_eq!(fixture.primary.calls.load(Ordering::SeqCst), 0);
                assert_eq!(fixture.quality.calls.load(Ordering::SeqCst), 0);
            }
        });
    }

    #[test]
    fn source_admission_refuses_foreign_producers_and_authenticated_source_mismatches() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fixture = fixture(&cx, EmbedderTierTag::Fast, true).await;
            let lexical = fixture
                .built
                .component_bytes(GenerationComponentRole::Lexical)
                .unwrap();
            let metadata = fixture
                .built
                .component_bytes(GenerationComponentRole::Metadata)
                .unwrap();
            let selected = fixture.candidate.manifest();
            for binding in [
                fixture.built.binding(),
                fixture.built.quality_binding().unwrap(),
            ] {
                let mut identity = binding.frozen_identity().identity.clone();
                identity
                    .producer
                    .implementation_revision
                    .push_str("-foreign");
                let foreign =
                    FsviV2IdentityBinding::new(binding.generation(), identity.freeze().unwrap())
                        .unwrap();
                let (primary, quality) =
                    if binding.dimension() == fixture.built.binding().dimension() {
                        (&foreign, fixture.built.quality_binding())
                    } else {
                        (fixture.built.binding(), Some(&foreign))
                    };
                assert!(
                    admit_source_components(
                        &cx,
                        selected,
                        primary,
                        quality,
                        fixture.tier,
                        Arc::clone(&lexical),
                        Arc::clone(&metadata),
                        BUDGET,
                    )
                    .is_err()
                );
            }
            for (source, docset) in [
                ([0x91; 32], selected.document_set_sha256()),
                (selected.source_checkpoint_sha256(), [0x92; 32]),
            ] {
                let manifest = reseal(&fixture, &metadata, source, docset);
                assert!(
                    fixture
                        .admit_images(
                            &cx,
                            &manifest,
                            Arc::clone(&lexical),
                            Arc::clone(&metadata),
                            BUDGET,
                        )
                        .is_err()
                );
            }
            let mut wrong_count = fixture.built.metadata().clone();
            wrong_count.total_documents = 1;
            wrong_count.lexical_artifacts[0].document_count = 1;
            wrong_count.manifest_hash = compute_manifest_hash(&wrong_count).unwrap();
            let wrong_count: Arc<[u8]> = serde_json::to_vec(&wrong_count).unwrap().into();
            let manifest = reseal(
                &fixture,
                &wrong_count,
                selected.source_checkpoint_sha256(),
                selected.document_set_sha256(),
            );
            assert!(
                fixture
                    .admit_images(&cx, &manifest, Arc::clone(&lexical), wrong_count, BUDGET)
                    .is_err()
            );
            let mut corrupt_lexical = lexical.to_vec();
            corrupt_lexical[0] ^= 1;
            assert!(
                fixture
                    .admit_images(&cx, selected, corrupt_lexical.into(), metadata, BUDGET)
                    .is_err()
            );
            assert_eq!(fixture.primary.calls.load(Ordering::SeqCst), 0);
            assert_eq!(fixture.quality.calls.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn source_recovery_cancellation_precedes_budget_refusals_and_returns_no_partial_documents() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fixture = fixture(&cx, EmbedderTierTag::Fast, false).await;
            let admitted = fixture.admit(&cx, BUDGET).unwrap();
            cx.set_cancel_requested(true);
            assert!(matches!(
                fixture.admit(&cx, 0),
                Err(SearchError::Cancelled { .. })
            ));
            assert!(matches!(
                admitted.source_documents(&cx, 0),
                Err(SearchError::Cancelled { .. })
            ));
            cx.set_cancel_requested(false);
            assert_eq!(admitted.source_documents(&cx, BUDGET).unwrap().len(), 2);
            assert_eq!(fixture.primary.calls.load(Ordering::SeqCst), 0);
        });
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn selected_source_survives_damaged_vectors_without_admitting_a_partial_search_snapshot() {
        use std::fs::{self, File, OpenOptions};
        use std::io::Write as _;
        use std::os::unix::fs::{OpenOptionsExt as _, PermissionsExt as _};

        use frankensearch_core::generation::GenerationRootSecurityProfileV1;
        use frankensearch_index::generation_root::authority_publisher::{
            AUTHORITY_PUBLISHER_LOCK_BYTES_V1, ExpectedAuthorityPairV1, PublicationOutcomeV1,
        };
        use frankensearch_index::generation_root::generation_reader::{
            SnapshotOpenOutcomeV1, SourceOpenOutcomeV1, activation_manifest_path_v1,
            component_object_path_v1,
        };
        use frankensearch_index::generation_root::{
            GENERATION_ROOT_AUTHORITY_FILE_BYTES, GenerationRootAnchorLayout,
            QualifiedGenerationRoot,
        };

        use crate::{GenerationActivationV1, GenerationPublisherV1};

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fixture = fixture(&cx, EmbedderTierTag::Fast, true).await;
            // Qualified roots require the ordinary persistent filesystem rather
            // than a possible tmpfs test directory.
            let directory = tempfile::Builder::new()
                .prefix(".frankensearch-source-recovery-")
                .permissions(fs::Permissions::from_mode(0o700))
                .tempdir_in(std::env::var_os("HOME").unwrap())
                .unwrap();
            for (name, size) in [
                ("LOCK", AUTHORITY_PUBLISHER_LOCK_BYTES_V1),
                ("AUTHORITY", GENERATION_ROOT_AUTHORITY_FILE_BYTES),
            ] {
                let mut file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .mode(0o600)
                    .open(directory.path().join(name))
                    .unwrap();
                file.write_all(&vec![0; usize::try_from(size).unwrap()])
                    .unwrap();
                file.sync_all().unwrap();
            }
            File::open(directory.path()).unwrap().sync_all().unwrap();
            File::open(directory.path().parent().unwrap())
                .unwrap()
                .sync_all()
                .unwrap();
            let root = QualifiedGenerationRoot::admit(
                directory.path(),
                GenerationRootAnchorLayout::new(AUTHORITY_PUBLISHER_LOCK_BYTES_V1).unwrap(),
            )
            .expect("qualified test root");
            for (role, receipt) in fixture.candidate.manifest().components().iter() {
                root.stage_immutable_file(
                    &component_object_path_v1(role, receipt.sha256).unwrap(),
                    &fixture.built.component_bytes(role).unwrap(),
                )
                .unwrap();
            }
            root.stage_immutable_file(
                &activation_manifest_path_v1(fixture.candidate.authority().object_id).unwrap(),
                fixture.candidate.manifest_bytes(),
            )
            .unwrap();
            let profile = GenerationRootSecurityProfileV1::CooperativeLocal;
            let root_id = [0xa1; 16];
            let mut publisher =
                GenerationPublisherV1::new(&root, root_id, [0xa2; 16], profile).unwrap();
            let attempt = publisher
                .prepare_attempt(
                    &fixture.candidate,
                    ExpectedAuthorityPairV1::default(),
                    [0xa3; 16],
                    [0xa4; 16],
                )
                .unwrap();
            let published = publisher
                .publish_prepared(&cx, &fixture.candidate, &attempt, None, BUDGET as u64)
                .unwrap();
            assert!(matches!(
                published.publication,
                PublicationOutcomeV1::Committed { .. }
            ));
            let Some(GenerationActivationV1::Installed(complete)) = published.activation else {
                panic!("fixture publication must admit a complete generation");
            };
            let original_pair = complete.authority_pair();
            drop(complete);
            let receipts = fixture.candidate.manifest().components();
            let vector_path = directory.path().join(component_object_name_v1(
                GenerationComponentRole::Vector,
                receipts.vector.sha256,
            ));
            fs::set_permissions(&vector_path, fs::Permissions::from_mode(0o600)).unwrap();
            fs::write(&vector_path, b"damaged retained vector").unwrap();
            fs::set_permissions(&vector_path, fs::Permissions::from_mode(0o400)).unwrap();
            assert!(matches!(
                root.open_expected_generation_snapshot_with_budget(
                    root_id,
                    profile,
                    None,
                    fixture.candidate.authority(),
                    BUDGET as u64,
                )
                .unwrap(),
                SnapshotOpenOutcomeV1::Refused(_)
            ));
            let SourceOpenOutcomeV1::Opened(opened) = root
                .open_expected_generation_source_for_recovery_with_budget(
                    root_id,
                    fixture.candidate.authority(),
                    fixture.built.binding().generation(),
                    BUDGET as u64,
                )
                .unwrap()
            else {
                panic!("intact authenticated source must remain recoverable");
            };
            let source = NativeGenerationSource::admit(
                &cx,
                opened,
                fixture.built.binding(),
                fixture.built.quality_binding(),
                fixture.tier,
                BUDGET,
            )
            .unwrap();
            assert_eq!(source.authority(), fixture.candidate.authority());
            assert_eq!(source.retained_source().authority_pair(), original_pair);
            assert_eq!(source.topology(), NativeGenerationTopology::FullProgressive);
            assert_eq!(source.tier(), EmbedderTierTag::Fast);
            assert_eq!(
                source.binding().frozen_identity(),
                fixture.built.binding().frozen_identity()
            );
            assert_eq!(source.quality_binding().unwrap().dimension(), 3);
            assert_eq!(source.metadata().total_documents, 2);
            assert_eq!(
                source
                    .source_documents(&cx, BUDGET)
                    .unwrap()
                    .iter()
                    .find(|document| document.id == "alpha")
                    .unwrap()
                    .title
                    .as_deref(),
                Some("Retained title")
            );
            assert_eq!(fs::read(&vector_path).unwrap(), b"damaged retained vector");
            let metadata_path = directory.path().join(component_object_name_v1(
                GenerationComponentRole::Metadata,
                receipts.metadata.sha256,
            ));
            fs::set_permissions(&metadata_path, fs::Permissions::from_mode(0o600)).unwrap();
            fs::write(&metadata_path, b"damaged later metadata").unwrap();
            fs::set_permissions(&metadata_path, fs::Permissions::from_mode(0o400)).unwrap();
            assert!(!matches!(
                root.open_expected_generation_source_for_recovery_with_budget(
                    root_id,
                    fixture.candidate.authority(),
                    fixture.built.binding().generation(),
                    BUDGET as u64,
                ),
                Ok(SourceOpenOutcomeV1::Opened(_))
            ));
            // Source recovery keeps the bytes it admitted, without reopening a
            // now-damaged pathname or manufacturing a current search reader.
            assert_eq!(source.source_documents(&cx, BUDGET).unwrap().len(), 2);
            assert_eq!(fixture.primary.calls.load(Ordering::SeqCst), 0);
            assert_eq!(fixture.quality.calls.load(Ordering::SeqCst), 0);
        });
    }
}
