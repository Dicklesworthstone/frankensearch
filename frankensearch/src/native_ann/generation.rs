//! Exact retrieval over one authority-admitted composite generation.
//!
//! All engine readers consume the retained component images. No component is
//! rediscovered through a pathname, and a later authority activation cannot
//! change the bytes used by an already-admitted reader.

use std::sync::Arc;

use frankensearch_core::generation::{
    ActivationManifest, AuthorityRefV1, CanonicalDocsetV1, EmbedderTierTag,
    ExactComponentReceiptV1, ExactGenerationComponentsV1, GenerationComponentRole,
    GenerationManifest, InvariantKind, SourceCheckpointV1, validate_manifest,
};
use frankensearch_index::exact_component_adapters::vector_component_receipt;
use frankensearch_index::generation_root::generation_reader::OpenedGenerationSnapshotV1;
use frankensearch_index::{FsviV2IdentityBinding, ValidatedFsviBytes};
use frankensearch_quill::{KeeperSnapshot, QuillConfig, QuillSnapshotIndex, SchemaDescriptor};
use sha2::Digest as _;

use super::{NativeAnnIndex, NativeRetrievalMode, checkpoint, invalid, lower_hex};
use crate::{Cx, Embedder, ScoredResult, SearchResult};

#[cfg(test)]
mod tests;

/// Query-capable readers for one fixed-authority, single-tier exact generation.
///
/// The authority-selected vector, lexical and metadata images are parsed and
/// joined before the handle becomes usable. The full FSVI generation and frozen
/// identity, the actual ordered live document sets, and the metadata's source
/// checkpoint must agree. Query-time producer admission is delegated to the
/// existing typed native retrieval path, including zero-result requests.
///
/// This first composite route supports one independently identified vector tier
/// (fast or quality) and an explicitly absent ANN component. Multi-tier/sharded
/// descriptors and declared ANN are refused, never silently reduced to a
/// different topology. The metadata artifact list describes exactly the two
/// retained search images; it is not a source of paths to open. Vector artifact
/// counts describe physical rows, including tombstones; the common document set
/// and metadata total count describe live documents. The existing metadata
/// readiness policy requires at least one live document in this route.
///
/// Keeping this handle retains the original authority capability and all engine
/// bytes. Publishing or rolling back another generation does not change an
/// in-flight or intentionally retained old reader. Obtain a fresh admitted root
/// snapshot and construct another handle to observe a new generation.
pub struct NativeGenerationSnapshot {
    authority: Arc<OpenedGenerationSnapshotV1>,
    readers: CompositeReaders,
}

pub(super) struct CompositeReaders {
    vector: NativeAnnIndex,
    lexical: QuillSnapshotIndex,
    metadata: GenerationManifest,
    tier: EmbedderTierTag,
}

impl NativeGenerationSnapshot {
    /// Admit the actual search components retained by a qualified root snapshot.
    ///
    /// `expected_vector` is the caller's selected generation/embedding/storage
    /// binding. It must not be inferred from the untrusted component it admits.
    /// `schema` identifies the caller's lexical schema. `max_image_bytes` bounds
    /// the combined encoded component and activation-manifest images, not total
    /// process RSS; parsed search structures have additional memory costs.
    ///
    /// All operations are synchronous with cancellation checkpoints between
    /// parsing/validation stages. Large admissions belong on the caller's
    /// bounded blocking lane. No runtime or background task is created.
    ///
    /// # Errors
    ///
    /// Rejects cancellation, unsupported topology, budget exhaustion, corrupt
    /// component bytes, source/document-set/identity disagreement, unsupported
    /// activation predicates, and inconsistent metadata artifact declarations.
    /// Admission errors contain bounded reason codes rather than raw metadata.
    pub fn admit_exact(
        cx: &Cx,
        authority: Arc<OpenedGenerationSnapshotV1>,
        expected_vector: &FsviV2IdentityBinding,
        schema: SchemaDescriptor,
        config: QuillConfig,
        max_image_bytes: usize,
    ) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.generation.admit")?;
        let component = |role| {
            authority.closure().bytes(role).ok_or_else(|| {
                invalid(
                    "generation.component",
                    "missing",
                    "the selected authority must retain every mandatory component",
                )
            })
        };
        let readers = admit_components(
            cx,
            authority.manifest(),
            expected_vector,
            schema,
            config,
            [
                component(GenerationComponentRole::Vector)?,
                component(GenerationComponentRole::Lexical)?,
                component(GenerationComponentRole::Metadata)?,
            ],
            max_image_bytes,
        )?;
        checkpoint(cx, "native_ann.generation.admitted")?;
        Ok(Self { authority, readers })
    }

    /// The exact authority retained by this reader, including rollback sequence.
    #[must_use]
    pub fn authority(&self) -> AuthorityRefV1 {
        self.authority.head().authority
    }

    /// The original admitted root capability, never a fresh current-head lookup.
    #[must_use]
    pub const fn retained_snapshot(&self) -> &Arc<OpenedGenerationSnapshotV1> {
        &self.authority
    }

    /// The single vector tier independently declared by the selected metadata.
    #[must_use]
    pub const fn tier(&self) -> EmbedderTierTag {
        self.readers.tier
    }

    /// The actual retrieval mode of this handle.
    #[must_use]
    pub const fn retrieval_mode(&self) -> NativeRetrievalMode {
        self.readers.vector.retrieval_mode()
    }

    /// The validated metadata image belonging to this exact authority.
    #[must_use]
    pub const fn metadata(&self) -> &GenerationManifest {
        &self.readers.metadata
    }

    /// Search and hydrate against the selected vector and Quill images.
    ///
    /// The configured producer is checked before inference and its bound output
    /// is checked again before scoring. Quality-only generations run the quality
    /// lane directly; they do not require or invent a fast tier.
    ///
    /// # Errors
    ///
    /// Propagates typed producer/embedding refusal, search/hydration failure and
    /// cancellation. A failed semantic arm never becomes lexical-only success.
    pub async fn search_text(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        self.readers.search_text(cx, embedder, text, k).await
    }
}

impl CompositeReaders {
    async fn search_text(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        checkpoint(cx, "native_ann.generation.search")?;
        let live_count =
            usize::try_from(self.vector.owner_witness().live_count).unwrap_or(usize::MAX);
        let k = k.min(live_count);
        match self.tier {
            EmbedderTierTag::Fast => {
                self.vector
                    .search_hybrid_text(cx, embedder, &self.lexical, text, k)
                    .await
            }
            EmbedderTierTag::Quality => {
                self.vector
                    .search_hybrid_quality_text(cx, embedder, &self.lexical, text, k)
                    .await
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn admit_components(
    cx: &Cx,
    activation: &ActivationManifest,
    expected_vector: &FsviV2IdentityBinding,
    schema: SchemaDescriptor,
    config: QuillConfig,
    images: [Arc<[u8]>; 3],
    max_image_bytes: usize,
) -> SearchResult<CompositeReaders> {
    checkpoint(cx, "native_ann.generation.components")?;
    let declared = activation.components();
    if declared.ann.is_some() {
        return Err(refused("ann", "declared-accelerator"));
    }
    if activation.generation() != expected_vector.generation() {
        return Err(refused("generation", "binding-mismatch"));
    }
    let image_bytes = images
        .iter()
        .try_fold(activation.canonical_bytes().len(), |total, image| {
            total.checked_add(image.len())
        });
    if image_bytes.is_none_or(|total| total > max_image_bytes) {
        return Err(refused("budget", "encoded-image-limit"));
    }
    let [vector_bytes, lexical_bytes, metadata_bytes] = images;
    for (image, receipt) in [
        (&vector_bytes, declared.vector),
        (&lexical_bytes, declared.lexical),
        (&metadata_bytes, declared.metadata),
    ] {
        let mut hasher = sha2::Sha256::new();
        for chunk in image.chunks(1024 * 1024) {
            checkpoint(cx, "native_ann.generation.component_hash")?;
            hasher.update(chunk);
        }
        if u64::try_from(image.len()).ok() != Some(receipt.byte_len)
            || <[u8; 32]>::from(hasher.finalize()) != receipt.sha256
        {
            return Err(refused("component", "image-mismatch"));
        }
    }
    checkpoint(cx, "native_ann.generation.metadata")?;
    let metadata: GenerationManifest =
        serde_json::from_slice(&metadata_bytes).map_err(|_| refused("metadata", "invalid-json"))?;
    if metadata.vector_artifacts.len() != 1
        || metadata.lexical_artifacts.len() != 1
        || metadata.embedders.len() != 1
        || !metadata.repair_descriptors.is_empty()
    {
        return Err(refused("metadata", "unsupported-component-shape"));
    }
    if !validate_manifest(&metadata).is_valid() {
        return Err(refused("metadata", "invalid-manifest"));
    }
    let checkpoint_value = SourceCheckpointV1::derive(&metadata.commit_range);
    if checkpoint_value.to_bytes() != activation.source_checkpoint_sha256() {
        return Err(refused("checkpoint", "source-mismatch"));
    }
    checkpoint(cx, "native_ann.generation.vector")?;
    let owner = Arc::new(
        ValidatedFsviBytes::from_arc(vector_bytes, expected_vector)
            .map_err(|_| refused("vector", "admission-failed"))?,
    );
    let witness = owner.witness();
    let mut identifiers = Vec::new();
    for row in 0..owner.record_count() {
        checkpoint(cx, "native_ann.generation.document")?;
        let row = owner.row(row)?;
        if row.flags().is_live() {
            identifiers.push(row.doc_id().to_owned());
        }
    }
    let vector_receipt = vector_component_receipt(
        witness,
        identifiers.iter().map(String::as_str),
        checkpoint_value,
    )
    .map_err(|_| refused("vector", "invalid-document-set"))?;
    if vector_receipt.bytes != declared.vector
        || vector_receipt.docset_digest != activation.document_set_sha256()
    {
        return Err(refused("vector", "selected-document-set-mismatch"));
    }
    let documents = CanonicalDocsetV1::from_ordered_live_documents(identifiers)
        .map_err(|_| refused("metadata", "invalid-document-set"))?;
    let metadata_receipt =
        ExactComponentReceiptV1::for_metadata_manifest(&metadata, &metadata_bytes, &documents)
            .map_err(|_| refused("metadata", "document-count-mismatch"))?;
    metadata_receipt
        .verify_metadata_manifest(&metadata, &metadata_bytes)
        .map_err(|_| refused("metadata", "receipt-mismatch"))?;
    validate_artifact_metadata(
        &metadata,
        expected_vector,
        activation,
        owner.record_count(),
        witness.live_count,
    )?;
    let lexical_receipt = ExactComponentReceiptV1 {
        role: GenerationComponentRole::Lexical,
        bytes: declared.lexical,
        docset_digest: activation.document_set_sha256(),
        live_document_count: witness.live_count,
        source_checkpoint: checkpoint_value.to_bytes(),
    };
    checkpoint(cx, "native_ann.generation.lexical")?;
    let lexical_result = KeeperSnapshot::from_exact_image(
        cx,
        lexical_bytes,
        schema,
        checkpoint_value,
        &lexical_receipt,
        max_image_bytes,
    );
    checkpoint(cx, "native_ann.generation.lexical_complete")?;
    let lexical = lexical_result.map_err(|_| refused("lexical", "admission-failed"))?;
    ExactGenerationComponentsV1::admit(vector_receipt, lexical_receipt, None, metadata_receipt)
        .map_err(|_| refused("components", "incoherent-generation"))?;
    checkpoint(cx, "native_ann.generation.readers")?;
    let vector = NativeAnnIndex::exact(cx, owner)?;
    let lexical_result = QuillSnapshotIndex::new(cx, lexical, config);
    checkpoint(cx, "native_ann.generation.ready")?;
    let lexical = lexical_result.map_err(|_| refused("lexical", "invalid-search-configuration"))?;
    let tier = metadata.vector_artifacts[0].embedder_tier;
    Ok(CompositeReaders {
        vector,
        lexical,
        metadata,
        tier,
    })
}

fn validate_artifact_metadata(
    metadata: &GenerationManifest,
    binding: &FsviV2IdentityBinding,
    activation: &ActivationManifest,
    record_count: usize,
    live_count: u64,
) -> SearchResult<()> {
    let declared = activation.components();
    let vector = &metadata.vector_artifacts[0];
    let lexical = &metadata.lexical_artifacts[0];
    let tier = match vector.embedder_tier {
        EmbedderTierTag::Fast => "fast",
        EmbedderTierTag::Quality => "quality",
    };
    if metadata.embedders.get(tier) != Some(binding.frozen_identity())
        || usize::try_from(vector.dimension).ok() != Some(binding.dimension())
        || usize::try_from(vector.vector_count).ok() != Some(record_count)
        || vector.size_bytes != declared.vector.byte_len
        || vector.checksum != lower_hex(&declared.vector.sha256)
        || lexical.document_count != live_count
        || lexical.size_bytes != declared.lexical.byte_len
        || lexical.checksum != lower_hex(&declared.lexical.sha256)
        || metadata.total_documents != live_count
    {
        return Err(refused("metadata", "artifact-binding-mismatch"));
    }
    for invariant in &metadata.activation_invariants {
        match &invariant.kind {
            InvariantKind::AllArtifactsVerified | InvariantKind::EmbedderRevisionMatch => {}
            InvariantKind::VectorCountConsistency { expected_total }
                if usize::try_from(*expected_total).ok() == Some(record_count) => {}
            _ => return Err(refused("metadata", "unsupported-activation-predicate")),
        }
    }
    Ok(())
}

fn refused(field: &str, code: &str) -> crate::SearchError {
    invalid(
        &format!("generation.{field}"),
        code,
        "the retained components do not satisfy the selected exact-generation contract",
    )
}
