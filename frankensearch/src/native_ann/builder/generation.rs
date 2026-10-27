//! Source-owned native components for the fixed generation-authority publisher.

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
use frankensearch_index::ann_cohort::{ANN_COHORT_HEADER_BYTES_V1, NativeAnnCohortV1};
use frankensearch_index::exact_component_adapters::{
    ann_component_receipt, vector_component_receipt,
};
use frankensearch_index::generation_root::generation_reader::component_object_name_v1;
use frankensearch_index::native_hnsw::{NativeHnswGenerationReceiptV2, ValidatedNativeHnsw};
use frankensearch_index::vector_cohort::{ExactVectorCohortV1, VECTOR_COHORT_HEADER_BYTES_V1};
use frankensearch_index::{FsviV2IdentityBinding, ValidatedFsviBytes};
use frankensearch_quill::{DEFAULT_SCHEMA, QuillConfig};

use super::super::generation::{NATIVE_ANN_RECEIPT_KEY, NATIVE_QUALITY_ANN_RECEIPT_KEY};
use super::super::{NativeRetrievalMode, checkpoint, invalid, lower_hex};
use super::scope::PreparedNativeGenerationScope;
use super::{NativeBuildRetrieval, NativeBuiltTier, NativeIndexBuilder, hybrid};
use crate::{Cx, IndexableDocument, SearchResult};

/// Complete owned vector, lexical, metadata, and optional native ANN images.
///
/// The only constructor builds every component from one source cohort and joins its actual
/// document-set/checkpoint receipts. No arbitrary reader or caller-provided
/// checksum can construct this payload. Images remain usable after the build
/// directory is moved or removed. Publication still requires the qualified-root
/// publisher; neither construction nor candidate creation changes AUTHORITY.
pub struct NativeBuiltExactGeneration {
    binding: FsviV2IdentityBinding,
    quality_binding: Option<FsviV2IdentityBinding>,
    metadata: GenerationManifest,
    components: ExactGenerationComponentsV1,
    vector: Arc<[u8]>,
    ann: Option<Arc<[u8]>>,
    lexical: Arc<[u8]>,
    metadata_bytes: Arc<[u8]>,
    retrieval: NativeBuildRetrieval,
    quality_retrieval: Option<NativeBuildRetrieval>,
    max_image_bytes: usize,
}

impl NativeIndexBuilder {
    /// Build a single vector tier and its complete immutable Quill cohort.
    ///
    /// `tier` labels the explicitly supplied primary provider as fast or quality;
    /// a quality-only generation does not load or invent a fast provider. A
    /// second provider and an unrepresentable source range are refused before
    /// inference or file creation. An empty cohort retains complete identity-bound
    /// vector and lexical images without performing document inference. The source
    /// range is supplied by the caller that owns the source cut, not inferred from IDs.
    ///
    /// The selected retrieval policy is preserved: HNSW includes a verified raw
    /// graph image and its full owner-bound receipt in the metadata image.
    /// "Exact" names the complete component authority contract, not a requirement
    /// to use exhaustive vector search.
    ///
    /// The encoded-image budget includes every component and reserves the
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
        if self.quality.is_some() {
            return Err(invalid(
                "builder.generation.plan",
                "unsupported",
                "authority generations require exactly one vector provider",
            ));
        }
        Box::pin(self.build_composite_generation(cx, tier, source, max_image_bytes)).await
    }

    /// Build independently identified fast and quality tiers over one complete cohort.
    ///
    /// Both providers must be explicitly supplied. Each retrieval plan may select
    /// exact search or native HNSW independently. Their dimensions may differ.
    /// Each FSVI image retains its own
    /// identity and receipt inside one canonical authority Vector component;
    /// Quill and metadata cover the same complete source checkpoint. Partial
    /// coverage and shards are not inferred. Configured graphs are independently
    /// bound to their own FSVI owners and packed into one authority ANN component.
    ///
    /// The encoded-image budget and caller-owned execution policy are the same
    /// as [`Self::build_exact_generation`]. No authority is published here.
    ///
    /// # Errors
    /// Refuses missing quality before inference or work-file
    /// creation, and propagates complete component, source, budget and cancellation errors.
    pub async fn build_progressive_generation(
        self,
        cx: &Cx,
        source: CommitRange,
        max_image_bytes: usize,
    ) -> SearchResult<NativeBuiltExactGeneration> {
        checkpoint(cx, "native_ann.builder.progressive_generation_start")?;
        if self.quality.is_none() {
            return Err(invalid(
                "builder.generation.plan",
                "unsupported",
                "complete progressive authority generations require independently supplied fast and quality providers",
            ));
        }
        Box::pin(self.build_composite_generation(
            cx,
            EmbedderTierTag::Fast,
            source,
            max_image_bytes,
        ))
        .await
    }

    async fn build_composite_generation(
        self,
        cx: &Cx,
        tier: EmbedderTierTag,
        source: CommitRange,
        max_image_bytes: usize,
    ) -> SearchResult<NativeBuiltExactGeneration> {
        if source
            .high
            .checked_sub(source.low)
            .and_then(|span| span.checked_add(1))
            .is_none()
        {
            return Err(invalid(
                "builder.generation.source",
                "unrepresentable",
                "a representable inclusive commit range is required",
            ));
        }
        let component_budget =
            remaining_budget(max_image_bytes, GENERATION_ACTIVATION_MANIFEST_MAX_BYTES_V2)?;
        let started = timestamp_millis()?;
        let retrieval = self.fast.retrieval;
        let quality_retrieval = self.quality.as_ref().map(|quality| quality.retrieval);
        let vectors = Box::pin(self.build(cx)).await?;
        let (vector, owner) = capture_vector(cx, vectors.fast(), component_budget)?;
        let source_checkpoint = SourceCheckpointV1::derive(&source);
        let cohort = if let Some(quality) = vectors.quality() {
            let remaining = remaining_budget(
                remaining_budget(component_budget, vector.len())?,
                VECTOR_COHORT_HEADER_BYTES_V1,
            )?;
            let (quality_bytes, _quality_owner) = capture_vector(cx, quality, remaining)?;
            Some(ExactVectorCohortV1::build(
                cx,
                (vectors.fast().binding(), Arc::clone(&vector)),
                (quality.binding(), quality_bytes),
                source_checkpoint,
                component_budget,
            )?)
        } else {
            None
        };
        let vector = cohort.as_ref().map_or(vector, ExactVectorCohortV1::bytes);
        let after_vector = remaining_budget(component_budget, vector.len())?;
        let has_graph = matches!(retrieval, NativeBuildRetrieval::Hnsw { .. })
            || quality_retrieval
                .is_some_and(|policy| matches!(policy, NativeBuildRetrieval::Hnsw { .. }));
        let graph_budget = if cohort.is_some() && has_graph {
            remaining_budget(after_vector, ANN_COHORT_HEADER_BYTES_V1)?
        } else {
            after_vector
        };
        let native_ann = capture_ann(cx, vectors.fast(), Arc::clone(&owner), graph_budget)?;
        let quality_ann = match (vectors.quality(), cohort.as_ref()) {
            (Some(quality), Some(cohort)) => capture_ann(
                cx,
                quality,
                Arc::clone(cohort.quality()),
                remaining_budget(
                    graph_budget,
                    native_ann.as_ref().map_or(0, |(bytes, _)| bytes.len()),
                )?,
            )?,
            _ => None,
        };
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
        let (ann, ann_receipt) = if let Some(cohort) = &cohort
            && has_graph
        {
            let packed = NativeAnnCohortV1::build(
                cx,
                cohort,
                native_ann
                    .as_ref()
                    .map(|(bytes, receipt)| (receipt, Arc::clone(bytes))),
                quality_ann
                    .as_ref()
                    .map(|(bytes, receipt)| (receipt, Arc::clone(bytes))),
                after_vector,
            )?;
            (Some(packed.bytes()), Some(packed.receipt().clone()))
        } else {
            let receipt = native_ann
                .as_ref()
                .map(|(_, receipt)| {
                    ann_component_receipt(
                        receipt,
                        ordered.iter().map(|document| document.id.as_str()),
                        source_checkpoint,
                    )
                    .map_err(|error| {
                        invalid("builder.generation.ann", "receipt", &error.to_string())
                    })
                })
                .transpose()?;
            (
                native_ann.as_ref().map(|(bytes, _)| Arc::clone(bytes)),
                receipt,
            )
        };
        let lexical_budget =
            remaining_budget(after_vector, ann.as_ref().map_or(0, |bytes| bytes.len()))?;
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
        let quality_binding = vectors.quality().map(|quality| quality.binding().clone());
        let mut metadata = build_metadata(
            &binding,
            tier,
            source,
            started,
            owner.witness().record_count,
            &vector_receipt,
            image.receipt(),
            native_ann.as_ref().map(|(_, receipt)| receipt),
        )?;
        if let (Some(cohort), Some(quality_binding)) = (&cohort, &quality_binding) {
            let quality = cohort.quality_receipt();
            metadata.vector_artifacts[0].path =
                format!("fast/{}", metadata.vector_artifacts[0].path);
            metadata.embedders.insert(
                "quality".to_owned(),
                quality_binding.frozen_identity().clone(),
            );
            metadata.vector_artifacts.push(VectorArtifact {
                path: format!(
                    "quality/{}",
                    component_object_name_v1(GenerationComponentRole::Vector, quality.bytes.sha256)
                ),
                size_bytes: quality.bytes.byte_len,
                checksum: lower_hex(&quality.bytes.sha256),
                vector_count: cohort.quality().witness().record_count,
                dimension: quality_binding.frozen_identity().identity.space.dimension,
                embedder_tier: EmbedderTierTag::Quality,
            });
            let physical_rows = owner
                .witness()
                .record_count
                .checked_add(cohort.quality().witness().record_count)
                .ok_or_else(image_budget_error)?;
            for invariant in &mut metadata.activation_invariants {
                if let InvariantKind::VectorCountConsistency { expected_total } =
                    &mut invariant.kind
                {
                    *expected_total = physical_rows;
                    "Every source document has a stored vector in each independently admitted tier"
                        .clone_into(&mut invariant.description);
                }
            }
            if let Some((_, receipt)) = &quality_ann {
                let encoded = serde_json::to_string(receipt).map_err(|error| {
                    invalid(
                        "builder.generation.ann",
                        "serialization",
                        &error.to_string(),
                    )
                })?;
                metadata
                    .metadata
                    .insert(NATIVE_QUALITY_ANN_RECEIPT_KEY.to_owned(), encoded);
            }
            metadata.manifest_hash = compute_manifest_hash(&metadata)?;
            require_valid(&validate_manifest(&metadata))?;
        }
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
            cohort
                .as_ref()
                .map_or(vector_receipt, |cohort| cohort.receipt().clone()),
            image.receipt().clone(),
            ann_receipt,
            metadata_receipt,
        )
        .map_err(|error| invalid("builder.generation.components", "join", &error.to_string()))?;
        checkpoint(cx, "native_ann.builder.generation_complete")?;
        Ok(NativeBuiltExactGeneration {
            binding,
            quality_binding,
            metadata,
            components,
            vector,
            ann,
            lexical,
            metadata_bytes,
            retrieval,
            quality_retrieval,
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

    /// Independently selected quality identity in a complete progressive generation.
    /// The primary binding is fast when this is present.
    #[must_use]
    pub const fn quality_binding(&self) -> Option<&FsviV2IdentityBinding> {
        self.quality_binding.as_ref()
    }

    /// Complete metadata derived from the admitted components.
    #[must_use]
    pub const fn metadata(&self) -> &GenerationManifest {
        &self.metadata
    }

    /// Actual primary-tier retrieval mode whose complete component images are retained.
    #[must_use]
    pub const fn retrieval_mode(&self) -> NativeRetrievalMode {
        match self.retrieval {
            NativeBuildRetrieval::Exact => NativeRetrievalMode::Exact {
                reason: super::super::NativeExactReason::Requested,
            },
            NativeBuildRetrieval::Hnsw { .. } => NativeRetrievalMode::Ann,
        }
    }

    /// Primary-tier construction/search parameters in the authenticated generation.
    #[must_use]
    pub const fn retrieval_configuration(&self) -> NativeBuildRetrieval {
        self.retrieval
    }

    /// Independent quality-tier policy in a complete fast/quality generation.
    /// A single-tier generation returns `None`, including quality-only topology.
    #[must_use]
    pub const fn quality_retrieval_configuration(&self) -> Option<NativeBuildRetrieval> {
        self.quality_retrieval
    }

    /// Share immutable component bytes without reopening the build directory.
    /// The ANN role is present only when HNSW was explicitly selected.
    #[must_use]
    pub fn component_bytes(&self, role: GenerationComponentRole) -> Option<Arc<[u8]>> {
        match role {
            GenerationComponentRole::Vector => Some(Arc::clone(&self.vector)),
            GenerationComponentRole::Lexical => Some(Arc::clone(&self.lexical)),
            GenerationComponentRole::Metadata => Some(Arc::clone(&self.metadata_bytes)),
            GenerationComponentRole::Ann => self.ann.as_ref().map(Arc::clone),
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
        self.admit_candidate(cx, candidate)?;
        Ok(())
    }

    /// Prepare a complete source scope before staging or publishing a candidate.
    ///
    /// The actual retained vector, lexical, metadata and optional ANN images are
    /// admitted against this exact candidate using the production reader checks.
    /// Its full stored source must fit `max_source_bytes` before any predicate is
    /// called. The budget and source semantics match
    /// [`crate::native_ann::NativeGenerationSnapshot::source_documents`].
    /// No model runs, pathname is reopened, or authority is changed. Predicates
    /// run once per source document and are not retained. After publication,
    /// [`PreparedNativeGenerationScope::bind`] checks and attaches the exact
    /// admitted result without repeating source work or observing cancellation.
    ///
    /// # Errors
    /// Refuses candidate/component drift, invalid stored source, budget
    /// exhaustion, predicate errors and cancellation without a partial scope.
    /// Predicate panics are not intercepted.
    pub fn prepare_scope(
        &self,
        cx: &Cx,
        candidate: &GenerationCandidateV1,
        max_source_bytes: usize,
        accept: impl FnMut(&IndexableDocument) -> SearchResult<bool>,
    ) -> SearchResult<PreparedNativeGenerationScope> {
        let readers = self.admit_candidate(cx, candidate)?;
        let source = readers.source_documents(cx, max_source_bytes)?;
        PreparedNativeGenerationScope::from_source(
            cx,
            candidate.authority(),
            candidate.manifest().components(),
            self.binding.clone(),
            self.metadata.vector_artifacts[0].embedder_tier,
            source,
            accept,
        )
    }

    fn admit_candidate(
        &self,
        cx: &Cx,
        candidate: &GenerationCandidateV1,
    ) -> SearchResult<super::super::generation::CompositeReaders> {
        if let Some(quality_binding) = &self.quality_binding {
            return super::super::generation::admit_progressive_components(
                cx,
                candidate.manifest(),
                &self.binding,
                quality_binding,
                DEFAULT_SCHEMA,
                QuillConfig::default(),
                [
                    Arc::clone(&self.vector),
                    Arc::clone(&self.lexical),
                    Arc::clone(&self.metadata_bytes),
                ],
                self.ann.as_ref().map(Arc::clone),
                self.max_image_bytes,
            );
        }
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
            self.ann.as_ref().map(Arc::clone),
            self.max_image_bytes,
        )
    }
}

fn capture_vector(
    cx: &Cx,
    tier: &NativeBuiltTier,
    budget: usize,
) -> SearchResult<(Arc<[u8]>, Arc<ValidatedFsviBytes>)> {
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
    Ok((bytes, Arc::new(owner)))
}

fn capture_ann(
    cx: &Cx,
    tier: &NativeBuiltTier,
    owner: Arc<ValidatedFsviBytes>,
    budget: usize,
) -> SearchResult<Option<(Arc<[u8]>, NativeHnswGenerationReceiptV2)>> {
    let (path, receipt) = match (tier.graph_path(), tier.graph_receipt.as_ref()) {
        (None, None) => return Ok(None),
        (Some(path), Some(receipt)) => (path, receipt),
        _ => {
            return Err(invalid(
                "builder.generation.ann",
                "missing",
                "the completed native graph requires its retained owner-bound receipt",
            ));
        }
    };
    let len = usize::try_from(receipt.graph_byte_len).map_err(|_| image_budget_error())?;
    if len > budget {
        return Err(image_budget_error());
    }
    checkpoint(cx, "native_ann.builder.generation_ann_open")?;
    let mut file = File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() != receipt.graph_byte_len {
        return Err(invalid(
            "builder.generation.ann",
            "changed",
            "the completed graph file differs from its retained build receipt",
        ));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|_| image_budget_error())?;
    let mut chunk = [0_u8; 16 * 1024];
    while bytes.len() < len {
        checkpoint(cx, "native_ann.builder.generation_ann_read")?;
        let count = chunk.len().min(len - bytes.len());
        file.read_exact(&mut chunk[..count])?;
        bytes.extend_from_slice(&chunk[..count]);
    }
    if file.read(&mut chunk[..1])? != 0 {
        return Err(image_budget_error());
    }
    checkpoint(cx, "native_ann.builder.generation_ann_admit")?;
    let bytes: Arc<[u8]> = bytes.into();
    ValidatedNativeHnsw::from_bytes(owner, Arc::clone(&bytes), receipt)
        .map_err(|error| invalid("builder.generation.ann", "invalid", &error.to_string()))?;
    checkpoint(cx, "native_ann.builder.generation_ann_complete")?;
    Ok(Some((bytes, receipt.clone())))
}

#[allow(clippy::too_many_arguments)]
fn build_metadata(
    binding: &FsviV2IdentityBinding,
    tier: EmbedderTierTag,
    source: CommitRange,
    started: u64,
    record_count: u64,
    vector: &ExactComponentReceiptV1,
    lexical: &ExactComponentReceiptV1,
    ann: Option<&NativeHnswGenerationReceiptV2>,
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
    if let Some(receipt) = ann {
        let encoded = serde_json::to_string(receipt).map_err(|error| {
            invalid(
                "builder.generation.ann",
                "serialization",
                &error.to_string(),
            )
        })?;
        metadata
            .metadata
            .insert(NATIVE_ANN_RECEIPT_KEY.to_owned(), encoded);
    }
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
