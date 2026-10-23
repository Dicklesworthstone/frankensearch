//! Source-owned exact components for the fixed generation-authority publisher.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use frankensearch_core::generation::{
    ActivationInvariant, AuthorityRefV1, CanonicalDocsetV1, CommitRange, EmbedderTierTag,
    ExactComponentReceiptV1, ExactGenerationComponentsV1,
    GENERATION_ACTIVATION_MANIFEST_MAX_BYTES_V2, GenerationAuthorityActionV1,
    GenerationComponentRole, GenerationManifest, InvariantKind, LexicalArtifact,
    MANIFEST_SCHEMA_VERSION, SourceCheckpointV1, VectorArtifact, compute_manifest_hash,
    require_valid, validate_manifest,
};
use frankensearch_fusion::generation_publication::{
    GenerationCandidateV1, GenerationPublicationErrorV1,
};
use frankensearch_index::exact_component_adapters::vector_component_receipt;
use frankensearch_index::generation_root::generation_reader::component_object_name_v1;
use frankensearch_index::{FsviV2IdentityBinding, ValidatedFsviBytes};
use frankensearch_quill::{DEFAULT_SCHEMA, QuillConfig};

use super::super::{checkpoint, invalid, lower_hex};
use super::{NativeBuildRetrieval, NativeBuiltTier, NativeIndexBuilder, hybrid};
use crate::{Cx, SearchResult};

/// Complete owned vector, lexical, and metadata images from one source cohort.
///
/// The only constructor builds all three components and joins their actual
/// document-set/checkpoint receipts. No arbitrary reader or caller-provided
/// checksum can construct this payload. Images remain usable after the build
/// directory is moved or removed. Publication still requires the qualified-root
/// publisher; neither construction nor candidate creation changes AUTHORITY.
pub struct NativeBuiltExactGeneration {
    binding: FsviV2IdentityBinding,
    metadata: GenerationManifest,
    components: ExactGenerationComponentsV1,
    vector: Arc<[u8]>,
    lexical: Arc<[u8]>,
    metadata_bytes: Arc<[u8]>,
    max_image_bytes: usize,
}

impl NativeIndexBuilder {
    /// Build a single exact vector tier and its complete immutable Quill cohort.
    ///
    /// `tier` labels the explicitly supplied primary provider as fast or quality;
    /// a quality-only generation does not load or invent a fast provider. A
    /// second provider, HNSW, an empty cohort, and an unrepresentable source range are
    /// refused before inference or file creation. The source range is supplied
    /// by the caller that owns the source cut, not inferred from document IDs.
    ///
    /// The encoded-image budget includes all three components and reserves the
    /// maximum activation-manifest size. It is not a process-memory or work-file
    /// limit. Quill is populated in the admitted FSVI's physical live-row order,
    /// which can differ from source-ID order. Its stored fields are checked
    /// against the retained source before exporting the immutable image.
    ///
    /// Filesystem work runs on the caller's execution lane, as in [`Self::build`].
    /// No authority is published and no task or runtime is spawned here.
    ///
    /// # Errors
    /// Refuses unsupported build plans, invalid source/metadata, oversized or
    /// changed component images, receipt drift, inference failures, and
    /// cancellation. An unsuccessful build may leave an unselected work directory.
    pub async fn build_exact_generation(
        self,
        cx: &Cx,
        tier: EmbedderTierTag,
        source: CommitRange,
        max_image_bytes: usize,
    ) -> SearchResult<NativeBuiltExactGeneration> {
        checkpoint(cx, "native_ann.builder.generation_start")?;
        if self.quality.is_some() || !matches!(self.fast.retrieval, NativeBuildRetrieval::Exact) {
            return Err(invalid(
                "builder.generation.plan",
                "unsupported",
                "exact authority generations require exactly one provider and no ANN",
            ));
        }
        if self.documents.is_empty()
            || source
                .high
                .checked_sub(source.low)
                .and_then(|span| span.checked_add(1))
                .is_none()
        {
            return Err(invalid(
                "builder.generation.source",
                "empty-or-unrepresentable",
                "a nonempty source cohort and a representable inclusive commit range are required",
            ));
        }
        let component_budget =
            remaining_budget(max_image_bytes, GENERATION_ACTIVATION_MANIFEST_MAX_BYTES_V2)?;
        let started = timestamp_millis()?;
        let vectors = Box::pin(self.build(cx)).await?;
        let (vector, owner) = capture_vector(cx, vectors.fast(), component_budget)?;
        let source_checkpoint = SourceCheckpointV1::derive(&source);
        let mut ordered = Vec::with_capacity(owner.live_count());
        for physical in 0..owner.record_count() {
            checkpoint(cx, "native_ann.builder.generation_document")?;
            let row = owner.row(physical)?;
            if row.flags().is_live() {
                ordered.push(vectors.document(row.doc_id()).ok_or_else(|| {
                    invalid(
                        "builder.generation.source_join",
                        "missing",
                        "an admitted vector row is absent from the retained source",
                    )
                })?);
            }
        }
        let vector_receipt = vector_component_receipt(
            owner.witness(),
            ordered.iter().map(|document| document.id.as_str()),
            source_checkpoint,
        )
        .map_err(|error| invalid("builder.generation.vector", "receipt", &error.to_string()))?;
        let documents = CanonicalDocsetV1::from_ordered_live_documents(
            ordered.iter().map(|document| document.id.as_str()),
        )
        .map_err(|error| invalid("builder.generation.docset", "invalid", &error.to_string()))?;
        let lexical_budget = remaining_budget(component_budget, vector.len())?;
        let (writer, reader, _seal) =
            hybrid::create_lexical(cx, &vectors.directory().join("lexical"), ordered).await?;
        hybrid::cohort::validate(cx, &reader, vectors.documents())?;
        checkpoint(cx, "native_ann.builder.generation_lexical")?;
        let snapshot = writer.snapshot()?;
        let image_result = snapshot.export_exact_image(cx, source_checkpoint, lexical_budget);
        checkpoint(cx, "native_ann.builder.generation_lexical_complete")?;
        let image = image_result
            .map_err(|error| invalid("builder.generation.lexical", "image", &error.to_string()))?;
        let lexical = image.bytes();
        let binding = vectors.fast().binding().clone();
        let metadata = build_metadata(
            &binding,
            tier,
            source,
            started,
            owner.witness().record_count,
            &vector_receipt,
            image.receipt(),
        )?;
        let metadata_bytes: Arc<[u8]> = serde_json::to_vec(&metadata)
            .map_err(|error| {
                invalid(
                    "builder.generation.metadata",
                    "serialization",
                    &error.to_string(),
                )
            })?
            .into();
        let metadata_budget = remaining_budget(lexical_budget, lexical.len())?;
        if metadata_bytes.len() > metadata_budget {
            return Err(image_budget_error());
        }
        let metadata_receipt =
            ExactComponentReceiptV1::for_metadata_manifest(&metadata, &metadata_bytes, &documents)
                .map_err(|error| {
                    invalid("builder.generation.metadata", "receipt", &error.to_string())
                })?;
        let components = ExactGenerationComponentsV1::admit(
            vector_receipt,
            image.receipt().clone(),
            None,
            metadata_receipt,
        )
        .map_err(|error| invalid("builder.generation.components", "join", &error.to_string()))?;
        checkpoint(cx, "native_ann.builder.generation_complete")?;
        Ok(NativeBuiltExactGeneration {
            binding,
            metadata,
            components,
            vector,
            lexical,
            metadata_bytes,
            max_image_bytes,
        })
    }
}

impl NativeBuiltExactGeneration {
    /// Actual generation and frozen embedding/storage identity of the vector image.
    #[must_use]
    pub const fn binding(&self) -> &FsviV2IdentityBinding {
        &self.binding
    }

    /// Complete metadata derived from the admitted components.
    #[must_use]
    pub const fn metadata(&self) -> &GenerationManifest {
        &self.metadata
    }

    /// Share immutable component bytes without reopening the build directory.
    /// The ANN role is absent for this explicitly exact generation.
    #[must_use]
    pub fn component_bytes(&self, role: GenerationComponentRole) -> Option<Arc<[u8]>> {
        match role {
            GenerationComponentRole::Vector => Some(Arc::clone(&self.vector)),
            GenerationComponentRole::Lexical => Some(Arc::clone(&self.lexical)),
            GenerationComponentRole::Metadata => Some(Arc::clone(&self.metadata_bytes)),
            GenerationComponentRole::Ann => None,
        }
    }

    /// Derive a publication candidate from the independently joined receipts.
    ///
    /// # Errors
    /// Propagates invalid authority identities, predecessor/action errors, and
    /// publication size limits. This operation does not stage or select anything.
    pub fn candidate(
        &self,
        object_id: [u8; 16],
        predecessor: Option<AuthorityRefV1>,
        action: GenerationAuthorityActionV1,
        writer_fence: [u8; 32],
    ) -> Result<GenerationCandidateV1, GenerationPublicationErrorV1> {
        let candidate = GenerationCandidateV1::new(
            object_id,
            predecessor,
            action,
            self.binding.generation(),
            writer_fence,
            &self.components,
        )?;
        let limit = u64::try_from(self.max_image_bytes).unwrap_or(u64::MAX);
        if candidate.owned_image_bytes() > limit {
            return Err(GenerationPublicationErrorV1::AdmissionBudget {
                required: candidate.owned_image_bytes(),
                limit,
            });
        }
        Ok(candidate)
    }

    /// Exercise the production retained reader over this exact candidate before
    /// staging or publishing it. This proves query admission without discovering
    /// a root or creating a second publication authority.
    ///
    /// # Errors
    /// Refuses candidate/component drift, invalid engine images or metadata, and
    /// cancellation using the same admission checks as selected-root serving.
    pub fn validate_candidate(
        &self,
        cx: &Cx,
        candidate: &GenerationCandidateV1,
    ) -> SearchResult<()> {
        super::super::generation::admit_components(
            cx,
            candidate.manifest(),
            &self.binding,
            DEFAULT_SCHEMA,
            QuillConfig::default(),
            [
                Arc::clone(&self.vector),
                Arc::clone(&self.lexical),
                Arc::clone(&self.metadata_bytes),
            ],
            self.max_image_bytes,
        )?;
        Ok(())
    }
}

fn capture_vector(
    cx: &Cx,
    tier: &NativeBuiltTier,
    budget: usize,
) -> SearchResult<(Arc<[u8]>, ValidatedFsviBytes)> {
    let retained = tier.index().owner_witness();
    let len = usize::try_from(retained.byte_len).map_err(|_| image_budget_error())?;
    if len > budget {
        return Err(image_budget_error());
    }
    checkpoint(cx, "native_ann.builder.generation_vector_open")?;
    let mut file = File::open(tier.vector_path())?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() != retained.byte_len {
        return Err(invalid(
            "builder.generation.vector",
            "changed",
            "the completed vector file differs from the retained build witness",
        ));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|_| image_budget_error())?;
    let mut chunk = [0_u8; 16 * 1024];
    while bytes.len() < len {
        checkpoint(cx, "native_ann.builder.generation_vector_read")?;
        let count = chunk.len().min(len - bytes.len());
        file.read_exact(&mut chunk[..count])?;
        bytes.extend_from_slice(&chunk[..count]);
    }
    if file.read(&mut chunk[..1])? != 0 {
        return Err(image_budget_error());
    }
    checkpoint(cx, "native_ann.builder.generation_vector_admit")?;
    let bytes: Arc<[u8]> = bytes.into();
    let owner = ValidatedFsviBytes::from_arc(Arc::clone(&bytes), tier.binding())
        .map_err(|error| invalid("builder.generation.vector", "invalid", &error.to_string()))?;
    if owner.witness() != retained {
        return Err(invalid(
            "builder.generation.vector",
            "changed",
            "the captured vector bytes do not match the complete retained build witness",
        ));
    }
    checkpoint(cx, "native_ann.builder.generation_vector_complete")?;
    Ok((bytes, owner))
}

fn build_metadata(
    binding: &FsviV2IdentityBinding,
    tier: EmbedderTierTag,
    source: CommitRange,
    started: u64,
    record_count: u64,
    vector: &ExactComponentReceiptV1,
    lexical: &ExactComponentReceiptV1,
) -> SearchResult<GenerationManifest> {
    let tier_name = match tier {
        EmbedderTierTag::Fast => "fast",
        EmbedderTierTag::Quality => "quality",
    };
    let mut metadata = GenerationManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        generation_id: binding.generation().fingerprint(),
        manifest_hash: String::new(),
        commit_range: source,
        build_started_at: started,
        build_completed_at: timestamp_millis()?,
        embedders: BTreeMap::from([(tier_name.to_owned(), binding.frozen_identity().clone())]),
        vector_artifacts: vec![VectorArtifact {
            path: component_object_name_v1(GenerationComponentRole::Vector, vector.bytes.sha256),
            size_bytes: vector.bytes.byte_len,
            checksum: lower_hex(&vector.bytes.sha256),
            vector_count: record_count,
            dimension: binding.frozen_identity().identity.space.dimension,
            embedder_tier: tier,
        }],
        lexical_artifacts: vec![LexicalArtifact {
            path: component_object_name_v1(GenerationComponentRole::Lexical, lexical.bytes.sha256),
            size_bytes: lexical.bytes.byte_len,
            checksum: lower_hex(&lexical.bytes.sha256),
            document_count: lexical.live_document_count,
        }],
        repair_descriptors: Vec::new(),
        activation_invariants: vec![
            ActivationInvariant {
                id: "exact_artifacts".to_owned(),
                description: "Every component matches its independently derived receipt".to_owned(),
                kind: InvariantKind::AllArtifactsVerified,
            },
            ActivationInvariant {
                id: "exact_producer".to_owned(),
                description: "The retained vector identity matches the configured producer"
                    .to_owned(),
                kind: InvariantKind::EmbedderRevisionMatch,
            },
            ActivationInvariant {
                id: "complete_vectors".to_owned(),
                description: "Every source document has exactly one stored vector".to_owned(),
                kind: InvariantKind::VectorCountConsistency {
                    expected_total: record_count,
                },
            },
        ],
        total_documents: vector.live_document_count,
        metadata: BTreeMap::new(),
    };
    metadata.manifest_hash = compute_manifest_hash(&metadata)?;
    require_valid(&validate_manifest(&metadata))?;
    Ok(metadata)
}

fn timestamp_millis() -> SearchResult<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
        .filter(|millis| *millis != 0)
        .ok_or_else(|| {
            invalid(
                "builder.generation.clock",
                "invalid",
                "a positive Unix timestamp is required",
            )
        })
}

fn remaining_budget(total: usize, used: usize) -> SearchResult<usize> {
    total
        .checked_sub(used)
        .filter(|remaining| *remaining != 0)
        .ok_or_else(image_budget_error)
}

fn image_budget_error() -> crate::SearchError {
    invalid(
        "builder.generation.image_budget",
        "exceeded",
        "encoded components and the reserved activation manifest must fit the image budget",
    )
}

#[cfg(test)]
mod tests;
