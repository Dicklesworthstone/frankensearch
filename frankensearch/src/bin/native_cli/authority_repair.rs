//! Rebuild damaged derived images from one authenticated retained source cohort.
//!
//! Source admission never produces a searchable partial snapshot. This explicit
//! operation keeps the selected physical authority pair through model loading
//! and rebuilding, then publishes a fully admitted Repair successor. Exact
//! retrieval is required so a corrupt content-addressed ANN name is never
//! overwritten or silently reused by a deterministic graph rebuild.

use frankensearch::native_ann::NativeGenerationSource;
use frankensearch_index::generation_root::generation_reader::SourceOpenOutcomeV1;
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
    if options.retrieval != Some(NativeBuildRetrieval::Exact) {
        return Err(bad("authority-repair requires explicit --retrieval exact"));
    }
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
    // No complete snapshot open: a selected vector or graph may be damaged.
    // The source capability is not queryable and never follows a newer head.
    let source = open_source(cx, &root, &selected, options.max_image_bytes)?;
    source.authority().next_sequence()?;
    let sequence = selected
        .generation
        .sequence
        .checked_add(1)
        .ok_or_else(|| bad("the artifact generation sequence is exhausted"))?;
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
    let prepared = prepare_repair(&source, &built, options.max_image_bytes)?;
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
        "receipt": receipt_path, "source": "authenticated_retained_source",
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
        || built.retrieval_configuration() != NativeBuildRetrieval::Exact
        || built.quality_retrieval_configuration()
            != source
                .quality_binding()
                .map(|_| NativeBuildRetrieval::Exact)
    {
        return Err(bad(
            "repair must preserve source, topology and complete producer identities in a newer exact generation",
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
