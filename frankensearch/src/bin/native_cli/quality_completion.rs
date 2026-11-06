//! Complete an explicitly selected fast-only cohort without repeating its fast
//! document inference. Only a completely built and sealed successor gets a new
//! selection receipt; the predecessor remains available throughout the build.

use frankensearch::SearchError;

use super::{
    BlockingPoolHandle, Cx, HnswParams, MAX_RECORD_BYTES, Models, NativeBuildPrecision,
    NativeBuildRetrieval, Options, Path, Result, SCHEMA, SELECTION_SCHEMA, Selection, Write, bad,
    emit, execution_context, fs, models, new_generation, new_path, save_selection, sharded, update,
};

fn checkpoint(cx: &Cx) -> Result<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: "native_cli.quality_completion".to_owned(),
        reason: error.to_string(),
    })?;
    if cx.is_cancel_requested() {
        return Err(SearchError::Cancelled {
            phase: "native_cli.quality_completion".to_owned(),
            reason: "caller cancelled quality completion".to_owned(),
        }
        .into());
    }
    Ok(())
}

pub async fn execute(
    cx: &Cx,
    options: &Options,
    output: &mut impl Write,
    pool: Option<BlockingPoolHandle>,
) -> Result<()> {
    let context = execution_context(cx, pool)?;
    let cx = &context;
    let pool = cx.blocking_pool_handle();
    execute_with_loader(cx, options, output, |root| {
        models::load(cx, root, true, &options.quality, pool)
    })
    .await
}

async fn execute_with_loader<F>(
    cx: &Cx,
    options: &Options,
    output: &mut impl Write,
    load: F,
) -> Result<()>
where
    F: FnOnce(Option<&Path>) -> Result<Models>,
{
    checkpoint(cx)?;
    let previous = Selection::read(&options.receipt)?;
    if previous.quality_producer.is_some() {
        return Err(bad(
            "complete-quality requires a fast-only selection; use rebuild to replace an existing quality model",
        ));
    }
    match (previous.schema.as_str(), options.shard_size) {
        (sharded::SELECTION_SCHEMA, Some(size)) => {
            sharded::validate_size(size, previous.documents)?;
        }
        (sharded::SELECTION_SCHEMA, None) => {
            return Err(bad(
                "sharded quality completion requires an explicit --shard-size",
            ));
        }
        (SELECTION_SCHEMA, None) => {}
        (SELECTION_SCHEMA, Some(_)) => {
            return Err(bad(
                "shard size cannot change an ordinary completion's layout",
            ));
        }
        _ => return Err(bad("unsupported quality-completion source layout")),
    }
    let sequence = previous
        .generation
        .sequence
        .checked_add(1)
        .ok_or_else(|| bad("generation sequence exhausted"))?;
    let old_root = fs::canonicalize(&previous.directory)?;
    let directory = new_path(
        options
            .directory
            .as_deref()
            .ok_or_else(|| bad("missing new index directory"))?,
    )?;
    let receipt = new_path(
        options
            .new_receipt
            .as_deref()
            .ok_or_else(|| bad("missing --new-receipt"))?,
    )?;
    if directory.starts_with(&old_root)
        || receipt.starts_with(&old_root)
        || directory.starts_with(&receipt)
        || receipt.starts_with(&directory)
    {
        return Err(bad(
            "new destinations must be outside the predecessor and must not overlap",
        ));
    }
    checkpoint(cx)?;
    let loaded = load(options.models.as_deref())?;
    checkpoint(cx)?;
    let quality = loaded.quality.ok_or_else(|| {
        bad("quality completion requires an independently supplied quality model")
    })?;
    // Admit the original fast owner using its original receipt. The new quality
    // model cannot satisfy, replace, or relabel the old producer identity.
    let index = sharded::Opened::open(
        cx,
        &previous,
        Models {
            fast: loaded.fast,
            quality: None,
        },
    )
    .await?;
    checkpoint(cx)?;
    update::validate_final_cohort(cx, index.borrow(), &update::Edits::new())?;
    let generation = new_generation(sequence)?;
    let retrieval = if options.exact {
        NativeBuildRetrieval::Exact
    } else {
        NativeBuildRetrieval::Hnsw {
            params: HnswParams::default(),
            seed: 42,
        }
    };
    let selection = match &index {
        sharded::Opened::Single(index) => {
            let successor = index
                .begin_quality_completion(cx, &directory, generation, quality)?
                .with_batch_size(options.batch_size)?
                .with_max_batch_input_bytes(MAX_RECORD_BYTES)?
                .with_quality_storage(NativeBuildPrecision::F32, retrieval)?
                .build_hybrid(cx)
                .await?;
            update::seal_selection(cx, (&successor).into())?
        }
        sharded::Opened::Sharded(index) => {
            let successor = index
                .begin_quality_completion(cx, &directory, generation, quality)?
                .with_batch_size(options.batch_size)?
                .with_max_batch_input_bytes(MAX_RECORD_BYTES)?
                .with_quality_storage(NativeBuildPrecision::F32, retrieval)?
                .build_sharded_hybrid(
                    cx,
                    options
                        .shard_size
                        .ok_or_else(|| bad("missing shard size"))?,
                )
                .await?;
            update::seal_selection(cx, (&successor).into())?
        }
    };
    checkpoint(cx)?;
    save_selection(&selection, &receipt)?;
    // Persistence is synchronous. If cancellation arrived during fsync, keep
    // the complete sealed receipt but do not report a successful command.
    checkpoint(cx)?;
    emit(
        output,
        &serde_json::json!({
            "schema": SCHEMA, "ok": true, "event": "quality_completed",
            "predecessor": previous.generation,
            "fast_reused_documents": previous.documents,
            "receipt": receipt, "selection": selection,
        }),
    )
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
#[path = "quality_completion_tests.rs"]
mod tests;
