//! Complete replacement components derived from authenticated retained vectors.

use std::path::Path;
use std::sync::Arc;

use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, CanonicalDocsetV1, ExactComponentReceiptV1,
    ExactGenerationComponentsV1, GENERATION_ACTIVATION_MANIFEST_MAX_BYTES_V2,
    GenerationComponentRole, SourceCheckpointV1, compute_manifest_hash, require_valid,
    validate_manifest,
};
use frankensearch_index::ValidatedFsviBytes;
use frankensearch_index::ann_cohort::{ANN_COHORT_HEADER_BYTES_V1, NativeAnnCohortV1};
use frankensearch_index::exact_component_adapters::{
    ann_component_receipt, vector_component_receipt,
};
use frankensearch_index::generation_root::generation_reader::component_object_name_v1;
use frankensearch_index::native_hnsw::{NativeHnswGenerationReceiptV2, ValidatedNativeHnsw};
use frankensearch_index::vector_cohort::ExactVectorCohortV1;

use super::{
    NativeBuildRetrieval, NativeBuiltExactGeneration, capture_ann_image, checkpoint,
    image_budget_error, invalid, lower_hex, remaining_budget, timestamp_millis,
};
use crate::native_ann::NativeGenerationVectorSource;
use crate::native_ann::generation::{NATIVE_ANN_RECEIPT_KEY, NATIVE_QUALITY_ANN_RECEIPT_KEY};
use crate::{Cx, SearchResult};

impl NativeGenerationVectorSource {
    /// Rebuild a complete generation from admitted vectors without model inference.
    ///
    /// Every vector's stored bits, flags and physical position are preserved;
    /// only its canonical generation header changes. The complete Quill image
    /// is reused byte-for-byte, including titles and document metadata. Each
    /// tier requires an explicit exact/HNSW policy. A full generation requires
    /// its independent quality policy, while a single tier forbids one.
    ///
    /// Graphs are built against the rebound owners in a fresh work directory.
    /// Choose fresh HNSW seeds when repairing a graph: recreating the selected
    /// ANN receipt is refused, so a corrupt content-addressed object can never
    /// be overwritten or adopted. All old images remain untouched. The returned
    /// components still require ordinary candidate admission and expected-pair
    /// publication before they become the selected generation.
    ///
    /// Work is synchronous on the caller's execution lane, with cancellation
    /// checkpoints between image, graph and publication-preparation stages.
    /// The image budget bounds encoded components, not parsed graph memory or
    /// the work directory. An interrupted build can leave unselected work files.
    ///
    /// # Errors
    /// Refuses non-newer generations, inconsistent tier policies, invalid HNSW
    /// parameters, existing work directories, repeated ANN receipts, image
    /// budget overflow, graph/receipt failures, I/O errors and cancellation.
    #[allow(clippy::too_many_arguments)]
    pub fn rebuild(
        &self,
        cx: &Cx,
        directory: impl AsRef<Path>,
        generation: ArtifactGenerationIdentityV1,
        primary_policy: NativeBuildRetrieval,
        quality_policy: Option<NativeBuildRetrieval>,
        max_image_bytes: usize,
    ) -> SearchResult<NativeBuiltExactGeneration> {
        checkpoint(cx, "native_ann.vector_recovery.build")?;
        generation.validate()?;
        let source = self.source();
        if generation.sequence <= source.binding().generation().sequence {
            return Err(invalid(
                "vector_recovery.generation",
                "not-newer",
                "retained-vector repair requires a strictly newer artifact generation",
            ));
        }
        if quality_policy.is_some() != self.quality_owner().is_some() {
            return Err(invalid(
                "vector_recovery.topology",
                "policy-mismatch",
                "each retained tier requires exactly one explicit retrieval policy",
            ));
        }
        for policy in std::iter::once(primary_policy).chain(quality_policy) {
            if let NativeBuildRetrieval::Hnsw { params, .. } = policy {
                params.validate()?;
            }
        }
        let component_budget =
            remaining_budget(max_image_bytes, GENERATION_ACTIVATION_MANIFEST_MAX_BYTES_V2)?;
        let lexical = source.retained_source().lexical_bytes();
        let vector_budget = remaining_budget(component_budget, lexical.len())?;
        let original_vector = self.retained_vector_source().vector_bytes();
        if original_vector.len() > vector_budget {
            return Err(image_budget_error());
        }
        let started = timestamp_millis()?;
        checkpoint(cx, "native_ann.vector_recovery.create")?;
        // An existing leaf, including a symlink, is never adopted or truncated.
        std::fs::create_dir(directory.as_ref())?;
        let directory = std::fs::canonicalize(directory.as_ref())?;
        let primary = Arc::new(self.primary_owner().rebind_generation(
            cx,
            generation,
            vector_budget,
        )?);
        let quality = self
            .quality_owner()
            .map(|owner| {
                owner
                    .rebind_generation(cx, generation, vector_budget)
                    .map(Arc::new)
            })
            .transpose()?;
        let checkpoint_value = SourceCheckpointV1::derive(&source.metadata().commit_range);
        let cohort = quality
            .as_ref()
            .map(|quality| {
                ExactVectorCohortV1::build(
                    cx,
                    (primary.identity_binding(), primary.owned_bytes()),
                    (quality.identity_binding(), quality.owned_bytes()),
                    checkpoint_value,
                    vector_budget,
                )
            })
            .transpose()?;
        let vector = cohort
            .as_ref()
            .map_or_else(|| primary.owned_bytes(), ExactVectorCohortV1::bytes);
        let mut identifiers = Vec::new();
        for physical in 0..primary.record_count() {
            checkpoint(cx, "native_ann.vector_recovery.document")?;
            let row = primary.row(physical)?;
            if row.flags().is_live() {
                identifiers.push(row.doc_id().to_owned());
            }
        }
        let primary_receipt = vector_component_receipt(
            primary.witness(),
            identifiers.iter().map(String::as_str),
            checkpoint_value,
        )
        .map_err(|_| {
            invalid(
                "vector_recovery.vector",
                "invalid",
                "invalid live document set",
            )
        })?;
        let vector_receipt = cohort.as_ref().map_or_else(
            || primary_receipt.clone(),
            |cohort| cohort.receipt().clone(),
        );
        let selected = source.retained_source().manifest();
        if vector_receipt.docset_digest != selected.document_set_sha256()
            || vector_receipt.live_document_count != source.metadata().total_documents
        {
            return Err(invalid(
                "vector_recovery.source",
                "changed",
                "rebound vectors must preserve the complete authenticated live cohort",
            ));
        }
        let has_graph = matches!(primary_policy, NativeBuildRetrieval::Hnsw { .. })
            || quality_policy
                .is_some_and(|policy| matches!(policy, NativeBuildRetrieval::Hnsw { .. }));
        let ann_budget = remaining_budget(vector_budget, vector.len())?;
        let graph_budget = if cohort.is_some() && has_graph {
            remaining_budget(ann_budget, ANN_COHORT_HEADER_BYTES_V1)?
        } else {
            ann_budget
        };
        let primary_graph = build_graph(
            cx,
            Arc::clone(&primary),
            primary_policy,
            &directory.join("primary.fshnsw"),
            graph_budget,
        )?;
        let quality_graph = quality
            .as_ref()
            .zip(quality_policy)
            .map(|(owner, policy)| {
                build_graph(
                    cx,
                    Arc::clone(owner),
                    policy,
                    &directory.join("quality.fshnsw"),
                    remaining_budget(
                        graph_budget,
                        primary_graph.as_ref().map_or(0, |(bytes, _)| bytes.len()),
                    )?,
                )
            })
            .transpose()?
            .flatten();
        let (ann, ann_receipt) = if let Some(cohort) = &cohort
            && has_graph
        {
            let graphs = NativeAnnCohortV1::build(
                cx,
                cohort,
                primary_graph
                    .as_ref()
                    .map(|(bytes, receipt)| (receipt, Arc::clone(bytes))),
                quality_graph
                    .as_ref()
                    .map(|(bytes, receipt)| (receipt, Arc::clone(bytes))),
                ann_budget,
            )?;
            (Some(graphs.bytes()), Some(graphs.receipt().clone()))
        } else {
            let receipt = primary_graph
                .as_ref()
                .map(|(_, receipt)| {
                    ann_component_receipt(
                        receipt,
                        identifiers.iter().map(String::as_str),
                        checkpoint_value,
                    )
                    .map_err(|_| invalid("vector_recovery.ann", "invalid", "invalid graph receipt"))
                })
                .transpose()?;
            (
                primary_graph.as_ref().map(|(bytes, _)| Arc::clone(bytes)),
                receipt,
            )
        };
        if let Some(receipt) = &ann_receipt
            && Some(receipt.bytes) == selected.components().ann
        {
            return Err(invalid(
                "vector_recovery.ann",
                "same-retained-object",
                "choose fresh HNSW seeds so repair cannot reuse the selected ANN object address",
            ));
        }
        let lexical_receipt = ExactComponentReceiptV1 {
            role: GenerationComponentRole::Lexical,
            bytes: selected.components().lexical,
            docset_digest: vector_receipt.docset_digest,
            live_document_count: vector_receipt.live_document_count,
            source_checkpoint: checkpoint_value.to_bytes(),
        };
        let mut metadata = source.metadata().clone();
        metadata.generation_id = generation.fingerprint();
        metadata.build_started_at = started;
        metadata.build_completed_at = timestamp_millis()?;
        for (position, (owner, receipt)) in std::iter::once((primary.as_ref(), &primary_receipt))
            .chain(
                cohort
                    .as_ref()
                    .map(|cohort| (cohort.quality().as_ref(), cohort.quality_receipt())),
            )
            .enumerate()
        {
            let artifact = &mut metadata.vector_artifacts[position];
            let name =
                component_object_name_v1(GenerationComponentRole::Vector, receipt.bytes.sha256);
            artifact.path = if cohort.is_some() {
                format!("{}/{name}", if position == 0 { "fast" } else { "quality" })
            } else {
                name
            };
            artifact.size_bytes = receipt.bytes.byte_len;
            artifact.checksum = lower_hex(&receipt.bytes.sha256);
            artifact.vector_count = owner.witness().record_count;
        }
        metadata.metadata.remove(NATIVE_ANN_RECEIPT_KEY);
        metadata.metadata.remove(NATIVE_QUALITY_ANN_RECEIPT_KEY);
        for (key, graph) in [
            (NATIVE_ANN_RECEIPT_KEY, primary_graph.as_ref()),
            (NATIVE_QUALITY_ANN_RECEIPT_KEY, quality_graph.as_ref()),
        ] {
            if let Some((_, receipt)) = graph {
                let encoded = serde_json::to_string(receipt).map_err(|_| {
                    invalid(
                        "vector_recovery.metadata",
                        "serialization",
                        "graph receipt encoding failed",
                    )
                })?;
                metadata.metadata.insert(key.to_owned(), encoded);
            }
        }
        metadata.manifest_hash = compute_manifest_hash(&metadata)?;
        require_valid(&validate_manifest(&metadata))?;
        let metadata_bytes: Arc<[u8]> = serde_json::to_vec(&metadata)
            .map_err(|_| {
                invalid(
                    "vector_recovery.metadata",
                    "serialization",
                    "metadata encoding failed",
                )
            })?
            .into();
        let metadata_budget =
            remaining_budget(ann_budget, ann.as_ref().map_or(0, |bytes| bytes.len()))?;
        if metadata_bytes.len() > metadata_budget {
            return Err(image_budget_error());
        }
        let documents =
            CanonicalDocsetV1::from_ordered_live_documents(identifiers).map_err(|_| {
                invalid(
                    "vector_recovery.metadata",
                    "invalid",
                    "invalid live document set",
                )
            })?;
        let metadata_receipt =
            ExactComponentReceiptV1::for_metadata_manifest(&metadata, &metadata_bytes, &documents)
                .map_err(|_| {
                    invalid(
                        "vector_recovery.metadata",
                        "invalid",
                        "metadata receipt admission failed",
                    )
                })?;
        let components = ExactGenerationComponentsV1::admit(
            vector_receipt,
            lexical_receipt,
            ann_receipt,
            metadata_receipt,
        )
        .map_err(|_| {
            invalid(
                "vector_recovery.components",
                "mismatch",
                "incoherent repair generation",
            )
        })?;
        checkpoint(cx, "native_ann.vector_recovery.complete")?;
        Ok(NativeBuiltExactGeneration {
            binding: primary.identity_binding().clone(),
            quality_binding: quality
                .as_ref()
                .map(|owner| owner.identity_binding().clone()),
            metadata,
            components,
            vector,
            ann,
            lexical,
            metadata_bytes,
            retrieval: primary_policy,
            quality_retrieval: quality_policy,
            max_image_bytes,
        })
    }
}

fn build_graph(
    cx: &Cx,
    owner: Arc<ValidatedFsviBytes>,
    policy: NativeBuildRetrieval,
    path: &Path,
    budget: usize,
) -> SearchResult<Option<(Arc<[u8]>, NativeHnswGenerationReceiptV2)>> {
    let NativeBuildRetrieval::Hnsw { params, seed } = policy else {
        return Ok(None);
    };
    checkpoint(cx, "native_ann.vector_recovery.graph")?;
    let graph = ValidatedNativeHnsw::build(cx, Arc::clone(&owner), params, seed)?;
    checkpoint(cx, "native_ann.vector_recovery.graph_save")?;
    let receipt = graph.save(path)?;
    checkpoint(cx, "native_ann.vector_recovery.graph_saved")?;
    capture_ann_image(cx, path, &receipt, owner, budget).map(Some)
}
