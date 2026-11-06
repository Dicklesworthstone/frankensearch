//! Rebuild damaged derived images from one authenticated retained source cohort.
//!
//! Source admission never produces a searchable partial snapshot. This explicit
//! operation keeps the selected physical authority pair through model loading
//! and rebuilding, then publishes a fully admitted Repair successor. Explicit
//! retained-vector repair skips models and source export, preserves stored
//! vector bits and Quill bytes, and can build fresh independently seeded graphs.
//! Reusing the selected ANN object address is refused before publication.

use frankensearch::native_ann::{NativeGenerationSource, NativeGenerationVectorSource};
use frankensearch_core::generation::GenerationComponentRole;
use frankensearch_index::generation_root::generation_reader::{
    SourceOpenOutcomeV1, VectorSourceOpenOutcomeV1,
};
use sha2::{Digest as _, Sha256};

use super::{
    Arc, BlockingPoolHandle, Cx, Embedder, EmbedderTierTag, FrozenEmbeddingIdentityBundleV1,
    FsviV2IdentityBinding, GenerationAuthorityActionV1, MAX_RECORD_BYTES, Models,
    NativeBuildRetrieval, NativeBuiltExactGeneration, NativeIndexBuilder, Options, PROFILE,
    PreparedPublication, PublicationOptions, QualifiedGenerationRoot, QueryMode, Receipt, Result,
    SCHEMA, Write, WriterIdentityV1, admit_root, annotate_retrieval, bad, checkpoint, emit,
    execution_context, load_authority_models, new_generation, new_path, publication_error,
    rebuild_identity, rebuild_precision, separate_destinations, stage_prepared_and_publish,
    validate_retained_documents,
};

#[derive(Debug, Default)]
pub(super) struct RetainedVectorOptions {
    pub(super) reuse_vectors: bool,
    pub(super) primary_seed: Option<u64>,
    pub(super) quality_seed: Option<u64>,
}

pub(super) fn validate_options(options: &Options) -> Result<()> {
    let vectors = &options.repair_vectors;
    if !vectors.reuse_vectors {
        if options.retrieval != Some(NativeBuildRetrieval::Exact)
            || vectors.primary_seed.is_some()
            || vectors.quality_seed.is_some()
        {
            return Err(bad(
                "authority-repair without --reuse-vectors requires explicit --retrieval exact and no graph seeds",
            ));
        }
        return Ok(());
    }
    if options.models.is_some()
        || options.quality.backend.is_some()
        || options.quality.directory.is_some()
    {
        return Err(bad(
            "--reuse-vectors cannot be combined with model directory or backend options",
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
            "--reuse-vectors requires explicit --retrieval exact without seeds or --retrieval hnsw with --hnsw-seed",
        )),
    }
}

fn vector_policies(
    options: &Options,
    selected: &Receipt,
) -> Result<(NativeBuildRetrieval, Option<NativeBuildRetrieval>)> {
    if options.repair_vectors.quality_seed.is_some() && selected.quality_identity.is_none() {
        return Err(bad(
            "--quality-hnsw-seed requires a complete two-tier selection",
        ));
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
    options.quality.validate_usage(
        selected.tier == EmbedderTierTag::Quality || selected.quality_identity.is_some(),
    )?;
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
    let (built, prepared) = if options.repair_vectors.reuse_vectors {
        // This branch never invokes the loader or exports the retained corpus.
        // Actual source/vector bytes, not the selected graph, authorize repair.
        let (primary, quality) = vector_policies(options, &selected)?;
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
            source.source(),
            &built,
            options.max_image_bytes,
            primary,
            quality,
            true,
        )?;
        (built, prepared)
    } else {
        // No complete snapshot open: a selected vector or graph may be damaged.
        // The source capability is not queryable and never follows a newer head.
        let source = open_source(cx, &root, &selected, options.max_image_bytes)?;
        source.authority().next_sequence()?;
        let documents = source.source_documents(cx, options.max_source_bytes)?;
        validate_retained_documents(cx, &documents)?;
        let loaded = load_models(selected.query_mode(None)?);
        checkpoint(cx)?;
        let loaded = loaded?;
        let identities = admit_models(&source, &loaded);
        checkpoint(cx)?;
        identities?;

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
        .with_fast_storage(
            rebuild_precision(&selected.identity)?,
            NativeBuildRetrieval::Exact,
        )
        .add_documents(documents);
        if let (Some(identity), Some(quality)) = (&selected.quality_identity, &loaded.quality) {
            builder = builder
                .with_quality_embedder(Arc::clone(quality))?
                .with_quality_storage(rebuild_precision(identity)?, NativeBuildRetrieval::Exact)?;
        }
        let source_cut = source.metadata().commit_range.clone();
        let built = if selected.quality_identity.is_some() {
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
        let identities = admit_models(&source, &loaded);
        checkpoint(cx)?;
        identities?;
        let prepared = prepare_repair(
            &source,
            &built,
            options.max_image_bytes,
            NativeBuildRetrieval::Exact,
            source
                .quality_binding()
                .map(|_| NativeBuildRetrieval::Exact),
            false,
        )?;
        (built, prepared)
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
        "replaces": selected.authority, "model_migrated": false,
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
    let source = NativeGenerationVectorSource::admit(
        cx,
        opened,
        &selected.binding()?,
        selected.quality_binding()?.as_ref(),
        selected.tier,
        usize::try_from(budget)?,
    );
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

fn open_source(
    cx: &Cx,
    root: &QualifiedGenerationRoot,
    selected: &Receipt,
    max_image_bytes: usize,
) -> Result<NativeGenerationSource> {
    selected.validate()?;
    checkpoint(cx)?;
    let budget = u64::try_from(max_image_bytes)?.min(selected.image_bytes);
    let opened = root.open_expected_generation_source_for_recovery_with_budget(
        selected.root_id,
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
    let source = NativeGenerationSource::admit(
        cx,
        opened,
        &selected.binding()?,
        selected.quality_binding()?.as_ref(),
        selected.tier,
        usize::try_from(budget)?,
    );
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

fn admit_model(expected: &FrozenEmbeddingIdentityBundleV1, model: &dyn Embedder) -> Result<()> {
    // Normalize only the persisted encoding, exactly as the source builder
    // does. Retained F16 remains F16; producer/input/space equality is strict.
    let identity = rebuild_identity(expected, model, true, true)?;
    if &identity != expected {
        return Err(bad(
            "authority-repair requires the selected complete producer identity",
        ));
    }
    rebuild_precision(expected)?;
    Ok(())
}

fn admit_models(source: &NativeGenerationSource, models: &Models) -> Result<()> {
    admit_model(source.binding().frozen_identity(), models.fast.as_ref())?;
    match (source.quality_binding(), &models.quality) {
        (Some(binding), Some(model)) => admit_model(binding.frozen_identity(), model.as_ref()),
        (None, None) => Ok(()),
        _ => Err(bad(
            "the repair providers do not match the selected topology",
        )),
    }
}

// Only this explicit repair route can turn a nonqueryable source capability
// into publication intent. Ordinary rebuild/activation still require complete
// admitted snapshots and never bypass damaged derived-image admission.
fn prepare_repair(
    source: &NativeGenerationSource,
    built: &NativeBuiltExactGeneration,
    max_image_bytes: usize,
    primary_policy: NativeBuildRetrieval,
    quality_policy: Option<NativeBuildRetrieval>,
    reuse_vectors: bool,
) -> Result<PreparedPublication> {
    let retained = source.retained_source();
    if built.binding().generation().sequence <= source.binding().generation().sequence
        || built.binding().frozen_identity() != source.binding().frozen_identity()
        || built
            .quality_binding()
            .map(FsviV2IdentityBinding::frozen_identity)
            != source
                .quality_binding()
                .map(FsviV2IdentityBinding::frozen_identity)
        || built.metadata().vector_artifacts[0].embedder_tier != source.tier()
        || built.metadata().commit_range != source.metadata().commit_range
        || built.metadata().total_documents != source.metadata().total_documents
        || built.retrieval_configuration() != primary_policy
        || built.quality_retrieval_configuration() != quality_policy
    {
        return Err(bad(
            "repair must preserve source, topology and complete producer identities in a newer generation with the requested retrieval policies",
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
        Some(source.authority()),
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
    if reuse_vectors
        && (built.component_bytes(GenerationComponentRole::Lexical)
            != Some(retained.lexical_bytes())
            || candidate
                .manifest()
                .components()
                .ann
                .is_some_and(|ann| Some(ann) == retained.manifest().components().ann))
    {
        return Err(bad(
            "retained-vector repair must preserve exact Quill bytes and use a new ANN object address",
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
