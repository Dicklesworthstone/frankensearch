//! Rebuild damaged derived images from one authenticated retained source cohort.
//!
//! Source admission never produces a searchable partial snapshot. This explicit
//! operation keeps the selected physical authority pair through model loading
//! and rebuilding, then publishes a fully admitted Repair successor. Explicit
//! retained-vector repair skips models and source export, preserving stored
//! vector bits and Quill bytes. Either route can build independently seeded
//! graphs; reusing the selected ANN object address is refused before publication.

use frankensearch::native_ann::{
    NativeGenerationQualityCompletionSource, NativeGenerationSource, NativeGenerationTopology,
    NativeGenerationVectorSource,
};
use frankensearch_core::generation::{GenerationComponentRole, GenerationManifest};
use frankensearch_index::generation_root::generation_reader::{
    OpenedGenerationSourceV1, SourceOpenOutcomeV1, VectorSourceOpenOutcomeV1,
};
use frankensearch_index::vector_cohort::VectorCohortCoverage;
use sha2::{Digest as _, Sha256};

use super::{
    Arc, BlockingPoolHandle, Cx, Embedder, EmbedderTierTag, FrozenEmbeddingIdentityBundleV1,
    FsviV2IdentityBinding, GenerationAuthorityActionV1, MAX_RECORD_BYTES, Models,
    NativeBuildRetrieval, NativeBuiltExactGeneration, NativeIndexBuilder, Options, PROFILE,
    PreparedPublication, PublicationOptions, QualifiedGenerationRoot, QueryMode, Receipt,
    ReceiptQueryMode, Result, SCHEMA, Write, WriterIdentityV1, admit_root, annotate_retrieval, bad,
    checkpoint, emit, execution_context, load_authority_models, new_generation, new_path,
    publication_error, rebuild_identity, rebuild_precision, separate_destinations,
    stage_prepared_and_publish, validate_retained_documents,
};

#[derive(Debug, Default)]
pub(super) struct RetainedVectorOptions {
    pub(super) reuse_vectors: bool,
    pub(super) primary_seed: Option<u64>,
    pub(super) quality_seed: Option<u64>,
}

enum SourceRepair {
    Preserve(NativeGenerationSource),
    CompleteQuality(NativeGenerationQualityCompletionSource),
}

impl SourceRepair {
    fn facts(&self, cx: &Cx) -> Result<RepairSourceFacts<'_>> {
        match self {
            Self::Preserve(source) => RepairSourceFacts::preserving(cx, source),
            Self::CompleteQuality(source) => Ok(RepairSourceFacts {
                retained: source.retained_source(),
                metadata: source.metadata(),
                primary: source.binding(),
                quality: Some(source.quality_binding()),
                tier: EmbedderTierTag::Fast,
                // This is the explicitly authorized successor coverage, never
                // a claim that the damaged predecessor's subset was admitted.
                coverage: VectorCohortCoverage::Complete,
                quality_document_ids: None,
            }),
        }
    }

    fn source_documents(
        &self,
        cx: &Cx,
        max_source_bytes: usize,
    ) -> Result<Vec<frankensearch::IndexableDocument>> {
        Ok(match self {
            Self::Preserve(source) => source.source_documents(cx, max_source_bytes)?,
            Self::CompleteQuality(source) => source.source_documents(cx, max_source_bytes)?,
        })
    }
}

struct RepairSourceFacts<'a> {
    retained: &'a Arc<OpenedGenerationSourceV1>,
    metadata: &'a GenerationManifest,
    primary: &'a FsviV2IdentityBinding,
    quality: Option<&'a FsviV2IdentityBinding>,
    tier: EmbedderTierTag,
    coverage: VectorCohortCoverage,
    quality_document_ids: Option<Vec<String>>,
}

impl<'a> RepairSourceFacts<'a> {
    fn preserving(cx: &Cx, source: &'a NativeGenerationSource) -> Result<Self> {
        let (coverage, quality_document_ids) = match source.topology() {
            NativeGenerationTopology::PartialQuality { .. } => (
                VectorCohortCoverage::PartialQuality,
                Some(source.partial_quality_document_ids(cx)?),
            ),
            _ => (VectorCohortCoverage::Complete, None),
        };
        Ok(Self {
            retained: source.retained_source(),
            metadata: source.metadata(),
            primary: source.binding(),
            quality: source.quality_binding(),
            tier: source.tier(),
            coverage,
            quality_document_ids,
        })
    }
}

#[derive(PartialEq, Eq)]
struct RepairIdentities {
    primary: FrozenEmbeddingIdentityBundleV1,
    quality: Option<FrozenEmbeddingIdentityBundleV1>,
}

pub(super) fn validate_options(options: &Options) -> Result<()> {
    let vectors = &options.repair_vectors;
    if vectors.reuse_vectors
        && (options.models.is_some()
            || options.quality.backend.is_some()
            || options.quality.directory.is_some()
            || options.complete_quality
            || options.migrate_model)
    {
        return Err(bad(
            "--reuse-vectors cannot be combined with model options, --complete-quality, or --migrate-model",
        ));
    }
    match options.retrieval {
        Some(NativeBuildRetrieval::Exact)
            if vectors.primary_seed.is_none() && vectors.quality_seed.is_none() =>
        {
            Ok(())
        }
        Some(NativeBuildRetrieval::Hnsw { .. }) if vectors.primary_seed.is_some() => Ok(()),
        _ => Err(bad(
            "authority-repair requires explicit --retrieval exact without seeds or --retrieval hnsw with --hnsw-seed",
        )),
    }
}

fn vector_policies(
    options: &Options,
    selected: &Receipt,
) -> Result<(NativeBuildRetrieval, Option<NativeBuildRetrieval>)> {
    if options.repair_vectors.quality_seed.is_some() && selected.quality_identity.is_none() {
        return Err(bad("--quality-hnsw-seed requires a two-tier selection"));
    }
    let primary = match options.retrieval {
        Some(NativeBuildRetrieval::Exact) => NativeBuildRetrieval::Exact,
        Some(NativeBuildRetrieval::Hnsw { params, .. }) => NativeBuildRetrieval::Hnsw {
            params,
            seed: options
                .repair_vectors
                .primary_seed
                .ok_or_else(|| bad("missing --hnsw-seed"))?,
        },
        None => return Err(bad("missing explicit repair retrieval policy")),
    };
    let quality = selected.quality_identity.as_ref().map(|_| match primary {
        NativeBuildRetrieval::Exact => NativeBuildRetrieval::Exact,
        NativeBuildRetrieval::Hnsw { params, seed } => NativeBuildRetrieval::Hnsw {
            params,
            seed: options.repair_vectors.quality_seed.unwrap_or(seed),
        },
    });
    Ok((primary, quality))
}

pub(super) async fn run(
    cx: &Cx,
    options: &Options,
    output: &mut impl Write,
    pool: Option<BlockingPoolHandle>,
) -> Result<()> {
    let context = execution_context(cx, pool)?;
    Box::pin(with_loader(&context, options, output, |mode| {
        load_authority_models(&context, options, mode, context.blocking_pool_handle())
    }))
    .await
}

async fn with_loader<F>(
    cx: &Cx,
    options: &Options,
    output: &mut impl Write,
    load_models: F,
) -> Result<()>
where
    F: FnOnce(QueryMode) -> Result<Models>,
{
    checkpoint(cx)?;
    validate_options(options)?;
    let selected = Receipt::read(
        options
            .from_receipt
            .as_deref()
            .ok_or_else(|| bad("missing repair source receipt"))?,
    )?;
    if options.complete_quality && selected.quality_documents.is_none() {
        return Err(bad(
            "--complete-quality requires a partial-quality repair source",
        ));
    }
    options.quality.validate_usage(
        selected.tier == EmbedderTierTag::Quality || selected.quality_identity.is_some(),
    )?;
    let (primary, quality) = vector_policies(options, &selected)?;
    let work_path = new_path(
        options
            .work
            .as_deref()
            .ok_or_else(|| bad("missing work directory"))?,
    )?;
    let receipt_path = new_path(&options.receipt)?;
    separate_destinations(&[&selected.root, &work_path, &receipt_path])?;
    let root = admit_root(&selected.root)?;
    let sequence = selected
        .generation
        .sequence
        .checked_add(1)
        .ok_or_else(|| bad("the artifact generation sequence is exhausted"))?;
    let (built, prepared, model_migrated) = if options.repair_vectors.reuse_vectors {
        // This branch never invokes the loader or exports the retained corpus.
        // Actual source/vector bytes, not the selected graph, authorize repair.
        let source = open_vector_source(cx, &root, &selected, options.max_image_bytes)?;
        source.source().authority().next_sequence()?;
        let built = source.rebuild(
            cx,
            &work_path,
            new_generation(sequence)?,
            primary,
            quality,
            options.max_image_bytes,
        );
        checkpoint(cx)?;
        let built = built?;
        let prepared = prepare_repair(
            cx,
            source.source(),
            &built,
            options.max_image_bytes,
            primary,
            quality,
            true,
        )?;
        (built, prepared, false)
    } else {
        // No complete snapshot open: a selected vector or graph may be damaged.
        // The source capability is not queryable and never follows a newer head.
        let source = if options.complete_quality {
            SourceRepair::CompleteQuality(open_quality_completion_source(
                cx,
                &root,
                &selected,
                options.max_image_bytes,
            )?)
        } else {
            SourceRepair::Preserve(open_source(cx, &root, &selected, options.max_image_bytes)?)
        };
        let facts = source.facts(cx)?;
        facts.retained.head().authority.next_sequence()?;
        let documents = source.source_documents(cx, options.max_source_bytes)?;
        validate_retained_documents(cx, &documents)?;
        let loaded = load_models(if facts.quality.is_some() {
            QueryMode::Full
        } else {
            selected.query_mode(None)?
        });
        checkpoint(cx)?;
        let loaded = loaded?;
        let identities = admit_models(&facts, &loaded, options.migrate_model);
        checkpoint(cx)?;
        let identities = identities?;

        // Move the complete authenticated documents directly into the builder.
        // Providers receive only content strings, never editable source records;
        // the builder checks its stored Quill fields against these owned records.
        let mut builder = NativeIndexBuilder::new(
            &work_path,
            new_generation(sequence)?,
            Arc::clone(&loaded.fast),
        )?
        .with_batch_size(options.batch_size)?
        .with_max_batch_input_bytes(MAX_RECORD_BYTES)?
        .with_fast_storage(rebuild_precision(&selected.identity)?, primary)
        .add_documents(documents);
        if let (Some(identity), Some(provider)) = (&selected.quality_identity, &loaded.quality) {
            builder = builder
                .with_quality_embedder(Arc::clone(provider))?
                .with_quality_storage(
                    rebuild_precision(identity)?,
                    quality.ok_or_else(|| bad("missing quality repair retrieval policy"))?,
                )?;
        }
        let source_cut = facts.metadata.commit_range.clone();
        let built = if let Some(quality_ids) = &facts.quality_document_ids {
            Box::pin(builder.build_partial_quality_generation(
                cx,
                source_cut,
                quality_ids.iter().cloned(),
                options.max_image_bytes,
            ))
            .await
        } else if selected.quality_identity.is_some() {
            Box::pin(builder.build_progressive_generation(cx, source_cut, options.max_image_bytes))
                .await
        } else {
            Box::pin(builder.build_exact_generation(
                cx,
                selected.tier,
                source_cut,
                options.max_image_bytes,
            ))
            .await
        };
        checkpoint(cx)?;
        let built = built?;
        let after = admit_models(&facts, &loaded, options.migrate_model);
        checkpoint(cx)?;
        if after? != identities {
            return Err(bad("the repair providers changed after identity admission"));
        }
        let prepared = prepare_repair_with_identities(
            cx,
            &facts,
            &built,
            options.max_image_bytes,
            (primary, quality),
            false,
            &identities,
        )?;
        let model_migrated = identities.primary != *facts.primary.frozen_identity()
            || identities.quality.as_ref()
                != facts.quality.map(FsviV2IdentityBinding::frozen_identity);
        (built, prepared, model_migrated)
    };
    let published = stage_prepared_and_publish(
        cx,
        &root,
        &selected.root,
        &receipt_path,
        &built,
        prepared,
        PublicationOptions::unscoped(options.max_image_bytes),
    )?;
    let receipt = published.receipt;
    let mut report = serde_json::json!({
        "schema": SCHEMA, "ok": true, "event": "repaired",
        "publication": "committed", "serving": "activated",
        "receipt": receipt_path,
        "source": if options.repair_vectors.reuse_vectors { "authenticated_retained_vectors" } else { "authenticated_retained_source" },
        "vectors_reused": options.repair_vectors.reuse_vectors,
        "replaces": selected.authority, "model_migrated": model_migrated,
        "quality_completed": options.complete_quality,
        "selection": receipt,
    });
    annotate_retrieval(
        &mut report,
        receipt.tier,
        built.retrieval_configuration(),
        built.quality_retrieval_configuration(),
        None,
    );
    emit(output, &report).map_err(|_| {
        publication_error(
            &receipt,
            "committed",
            "activated",
            "result_delivery_failed",
            None,
        )
    })
}

fn open_vector_source(
    cx: &Cx,
    root: &QualifiedGenerationRoot,
    selected: &Receipt,
    max_image_bytes: usize,
) -> Result<NativeGenerationVectorSource> {
    selected.validate()?;
    checkpoint(cx)?;
    let budget = u64::try_from(max_image_bytes)?.min(selected.image_bytes);
    let opened = root.open_expected_generation_vector_source_for_recovery_with_budget(
        selected.root_id,
        PROFILE,
        None,
        selected.authority()?,
        selected.generation,
        budget,
    );
    checkpoint(cx)?;
    let opened = match opened? {
        VectorSourceOpenOutcomeV1::Opened(opened) => opened,
        VectorSourceOpenOutcomeV1::Refused(reason) => {
            return Err(bad(&format!(
                "authority repair vector admission refused: {reason:?}"
            )));
        }
    };
    if opened.source().root_id() != selected.root_id || opened.source().profile() != PROFILE {
        return Err(bad(
            "the retained vectors differ from the selected root or profile",
        ));
    }
    let primary = selected.binding()?;
    let quality = selected.quality_binding()?;
    let source = match selected.topology() {
        NativeGenerationTopology::PartialQuality { .. } => {
            NativeGenerationVectorSource::admit_partial_quality(
                cx,
                opened,
                &primary,
                quality
                    .as_ref()
                    .ok_or_else(|| bad("missing partial quality identity"))?,
                usize::try_from(budget)?,
            )
        }
        _ => NativeGenerationVectorSource::admit(
            cx,
            opened,
            &primary,
            quality.as_ref(),
            selected.tier,
            usize::try_from(budget)?,
        ),
    };
    checkpoint(cx)?;
    let source = source?;
    if source.source().topology() != selected.topology()
        || source.source().metadata().total_documents != selected.documents
    {
        return Err(bad(
            "the retained vectors differ from the selected topology or document count",
        ));
    }
    Ok(source)
}

/// Admit exactly the components that prove this receipt's selected source.
/// Authenticated partial membership permits source rescue without vector or
/// graph images. Older partial metadata still requires its healthy retained
/// quality rows; no other source admission failure authorizes that fallback.
pub(super) fn export_documents(
    cx: &Cx,
    root: &QualifiedGenerationRoot,
    selected: &Receipt,
    max_image_bytes: usize,
    max_source_bytes: usize,
) -> Result<Vec<frankensearch::IndexableDocument>> {
    match open_source(cx, root, selected, max_image_bytes) {
        Ok(source) => Ok(source.source_documents(cx, max_source_bytes)?),
        Err(error)
            if selected.quality_documents.is_some()
                && matches!(
                    error.downcast_ref::<frankensearch::SearchError>(),
                    Some(frankensearch::SearchError::InvalidConfig { field, value, .. })
                        if field == "native_ann.generation.source-recovery"
                            && value == "quality-membership-required"
                ) =>
        {
            // A missing historical declaration does not establish coverage.
            // Reopen only the same expected authority and independently prove
            // its subset from healthy vectors; never follow a newer selection.
            let vectors = open_vector_source(cx, root, selected, max_image_bytes)?;
            Ok(vectors.source().source_documents(cx, max_source_bytes)?)
        }
        Err(error) => Err(error),
    }
}

fn open_source_capability(
    cx: &Cx,
    root: &QualifiedGenerationRoot,
    selected: &Receipt,
    max_image_bytes: usize,
) -> Result<(Arc<OpenedGenerationSourceV1>, usize)> {
    selected.validate()?;
    checkpoint(cx)?;
    let budget = u64::try_from(max_image_bytes)?.min(selected.image_bytes);
    let opened = root.open_expected_generation_source_for_recovery_with_budget(
        selected.root_id,
        PROFILE,
        None,
        selected.authority()?,
        selected.generation,
        budget,
    );
    checkpoint(cx)?;
    let opened = match opened? {
        SourceOpenOutcomeV1::Opened(opened) => opened,
        SourceOpenOutcomeV1::Refused(reason) => {
            return Err(bad(&format!(
                "authority repair source admission refused: {reason:?}"
            )));
        }
    };
    if opened.root_id() != selected.root_id || opened.profile() != PROFILE {
        return Err(bad(
            "the repair source differs from the selected root or profile",
        ));
    }
    Ok((opened, usize::try_from(budget)?))
}

fn open_source(
    cx: &Cx,
    root: &QualifiedGenerationRoot,
    selected: &Receipt,
    max_image_bytes: usize,
) -> Result<NativeGenerationSource> {
    let (opened, budget) = open_source_capability(cx, root, selected, max_image_bytes)?;
    let primary = selected.binding()?;
    let quality = selected.quality_binding()?;
    let source = if selected.quality_documents.is_some() {
        NativeGenerationSource::admit_partial_quality(
            cx,
            opened,
            &primary,
            quality
                .as_ref()
                .ok_or_else(|| bad("missing partial quality identity"))?,
            budget,
        )
    } else {
        NativeGenerationSource::admit(
            cx,
            opened,
            &primary,
            quality.as_ref(),
            selected.tier,
            budget,
        )
    };
    checkpoint(cx)?;
    let source = source?;
    if source.topology() != selected.topology()
        || source.metadata().total_documents != selected.documents
    {
        return Err(bad(
            "the repair source differs from the selected topology or document count",
        ));
    }
    Ok(source)
}

fn open_quality_completion_source(
    cx: &Cx,
    root: &QualifiedGenerationRoot,
    selected: &Receipt,
    max_image_bytes: usize,
) -> Result<NativeGenerationQualityCompletionSource> {
    if selected.quality_documents.is_none() || selected.tier != EmbedderTierTag::Fast {
        return Err(bad(
            "quality-completing repair requires a partial-quality selection",
        ));
    }
    let (opened, budget) = open_source_capability(cx, root, selected, max_image_bytes)?;
    let quality = selected
        .quality_binding()?
        .ok_or_else(|| bad("partial-quality repair requires its independent quality identity"))?;
    let source = NativeGenerationQualityCompletionSource::admit(
        cx,
        opened,
        &selected.binding()?,
        &quality,
        budget,
    );
    checkpoint(cx)?;
    let source = source?;
    if source.metadata().total_documents != selected.documents {
        return Err(bad(
            "the repair source differs from the selected document count",
        ));
    }
    Ok(source)
}

fn admit_model(
    expected: &FrozenEmbeddingIdentityBundleV1,
    model: &dyn Embedder,
    migrate_model: bool,
) -> Result<FrozenEmbeddingIdentityBundleV1> {
    // Normalize only the persisted encoding, exactly as the source builder
    // does. Retained F16 remains F16 and prepared input is never reinterpreted.
    let identity = rebuild_identity(expected, model, true, true)?;
    if !migrate_model && &identity != expected {
        return Err(bad(
            "authority-repair requires the selected complete producer identity",
        ));
    }
    rebuild_precision(expected)?;
    Ok(identity)
}

fn admit_models(
    source: &RepairSourceFacts<'_>,
    models: &Models,
    migrate_model: bool,
) -> Result<RepairIdentities> {
    let primary = admit_model(
        source.primary.frozen_identity(),
        models.fast.as_ref(),
        migrate_model,
    )?;
    let quality = match (source.quality, &models.quality) {
        (Some(binding), Some(model)) => Some(admit_model(
            binding.frozen_identity(),
            model.as_ref(),
            migrate_model,
        )?),
        (None, None) => None,
        _ => {
            return Err(bad(
                "the repair providers do not match the selected topology",
            ));
        }
    };
    Ok(RepairIdentities { primary, quality })
}

// Only this explicit repair route can turn a nonqueryable source capability
// into publication intent. Ordinary rebuild/activation still require complete
// admitted snapshots and never bypass damaged derived-image admission.
fn prepare_repair(
    cx: &Cx,
    source: &NativeGenerationSource,
    built: &NativeBuiltExactGeneration,
    max_image_bytes: usize,
    primary_policy: NativeBuildRetrieval,
    quality_policy: Option<NativeBuildRetrieval>,
    reuse_vectors: bool,
) -> Result<PreparedPublication> {
    let facts = RepairSourceFacts::preserving(cx, source)?;
    let identities = RepairIdentities {
        primary: source.binding().frozen_identity().clone(),
        quality: source
            .quality_binding()
            .map(|binding| binding.frozen_identity().clone()),
    };
    prepare_repair_with_identities(
        cx,
        &facts,
        built,
        max_image_bytes,
        (primary_policy, quality_policy),
        reuse_vectors,
        &identities,
    )
}

fn prepare_repair_with_identities(
    cx: &Cx,
    source: &RepairSourceFacts<'_>,
    built: &NativeBuiltExactGeneration,
    max_image_bytes: usize,
    policies: (NativeBuildRetrieval, Option<NativeBuildRetrieval>),
    reuse_vectors: bool,
    identities: &RepairIdentities,
) -> Result<PreparedPublication> {
    let retained = source.retained;
    let (primary_policy, quality_policy) = policies;
    if built.binding().generation().sequence <= source.primary.generation().sequence
        || built.binding().frozen_identity() != &identities.primary
        || built
            .quality_binding()
            .map(FsviV2IdentityBinding::frozen_identity)
            != identities.quality.as_ref()
        || built.metadata().vector_artifacts[0].embedder_tier != source.tier
        || built.metadata().commit_range != source.metadata.commit_range
        || built.metadata().total_documents != source.metadata.total_documents
        || built.coverage() != source.coverage
        || built.retrieval_configuration() != primary_policy
        || built.quality_retrieval_configuration() != quality_policy
    {
        return Err(bad(
            "repair must preserve source and authorized coverage and producer identities in a newer generation with the requested retrieval policies",
        ));
    }
    let writer = WriterIdentityV1::for_current_process()?;
    let object_id = new_generation(1)?.nonce;
    let mut fence = Sha256::new();
    fence.update(b"frankensearch.native.authority-repair.v1");
    fence.update(writer.writer_id);
    fence.update(writer.nonce);
    fence.update(object_id);
    let candidate = built.candidate(
        object_id,
        Some(retained.head().authority),
        GenerationAuthorityActionV1::Repair,
        fence.finalize().into(),
    )?;
    if candidate.manifest().document_set_sha256() != retained.manifest().document_set_sha256()
        || candidate.manifest().source_checkpoint_sha256()
            != retained.manifest().source_checkpoint_sha256()
    {
        return Err(bad(
            "repair must preserve the complete authenticated document set and source checkpoint",
        ));
    }
    if let Some(quality_ids) = &source.quality_document_ids {
        let preview = built.preview_candidate(cx, &candidate)?;
        if preview.partial_quality_document_ids(cx)? != *quality_ids {
            return Err(bad(
                "repair must preserve the exact authenticated quality subset",
            ));
        }
    }
    if candidate
        .manifest()
        .components()
        .ann
        .is_some_and(|ann| Some(ann) == retained.manifest().components().ann)
    {
        return Err(bad("repair must use a new ANN object address"));
    }
    if reuse_vectors
        && built.component_bytes(GenerationComponentRole::Lexical) != Some(retained.lexical_bytes())
    {
        return Err(bad(
            "retained-vector repair must preserve exact Quill bytes",
        ));
    }
    if candidate.owned_image_bytes() > u64::try_from(max_image_bytes)? {
        return Err(bad(
            "complete repair generation exceeds the encoded-image budget",
        ));
    }
    Ok(PreparedPublication {
        candidate,
        root_id: retained.root_id(),
        expected: retained.authority_pair(),
        writer,
    })
}

#[cfg(test)]
#[path = "authority_repair_tests.rs"]
mod tests;
