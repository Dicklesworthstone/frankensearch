//! Opt-in source transactions using the live handle's original models and pin.
//! A completed candidate is sealed and its NEW receipt synced before process-
//! local installation. Neither persistence nor installation is ever rolled back
//! by a later error. This does not implement a durable CURRENT authority or GC.

use frankensearch::SearchError;
use frankensearch::native_ann::builder::live::NativeHybridSnapshot;

#[cfg(test)]
use super::NativeLiveHybridIndex;
use super::validate_id;
use crate::{
    ArtifactGenerationIdentityV1, Cx, Deserialize, Path, PathBuf, Result, SCHEMA, Selection, Write,
    bad, emit, fs, live, new_generation, new_path, query, save_selection, sharded, update,
};

pub(super) const MAX_MUTATIONS: usize = 1_000;
const BATCH_SIZE: usize = 16;

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Request {
    Update {
        id: Option<String>,
        expected_generation: ArtifactGenerationIdentityV1,
        index_dir: PathBuf,
        new_receipt: PathBuf,
        changes: Vec<update::Mutation>,
        shard_size: Option<usize>,
    },
}

struct Prepared {
    candidate: live::Candidate,
    selection: Selection,
    receipt: PathBuf,
    edited_ids: usize,
}

/// Track effects before a fallible operation, not only after it succeeds. An
/// incomplete receipt write may have exposed bytes even when sync returned Err.
struct Progress {
    stage: &'static str,
    previous: Option<ArtifactGenerationIdentityV1>,
    candidate: Option<ArtifactGenerationIdentityV1>,
    receipt: Option<PathBuf>,
    receipt_state: &'static str,
}

impl Default for Progress {
    fn default() -> Self {
        Self {
            stage: "admission",
            previous: None,
            candidate: None,
            receipt: None,
            receipt_state: "not_started",
        }
    }
}

fn checkpoint(cx: &Cx) -> Result<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: "native_cli.warm_update".to_owned(),
        reason: error.to_string(),
    })?;
    Ok(())
}

fn validate_destination(path: &Path) -> Result<()> {
    let text = path
        .to_str()
        .ok_or_else(|| bad("update destinations must be UTF-8"))?;
    if !path.is_absolute() || text.len() > 4096 || text.contains('\0') {
        return Err(bad(
            "update destinations must be absolute, NUL-free paths of at most 4096 bytes",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn prepare(
    base: &NativeHybridSnapshot,
    cx: &Cx,
    expected: ArtifactGenerationIdentityV1,
    directory: &Path,
    receipt: &Path,
    changes: Vec<update::Mutation>,
    progress: &mut Progress,
) -> Result<Prepared> {
    let base = live::Snapshot::Single(base.clone());
    prepare_with_partition(
        &base, cx, expected, directory, receipt, changes, None, progress,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn prepare_with_partition(
    base: &live::Snapshot,
    cx: &Cx,
    expected: ArtifactGenerationIdentityV1,
    directory: &Path,
    receipt: &Path,
    changes: Vec<update::Mutation>,
    shard_size: Option<usize>,
    progress: &mut Progress,
) -> Result<Prepared> {
    checkpoint(cx)?;
    expected.validate()?;
    progress.previous = Some(base.generation());
    if expected != base.generation() {
        return Err(bad(
            "expected_generation does not match the serving snapshot; no update started",
        ));
    }
    if changes.is_empty() || changes.len() > MAX_MUTATIONS {
        return Err(bad("an update must contain between 1 and 1000 mutations"));
    }
    update::validate_partition_policy(matches!(base, live::Snapshot::Sharded(_)), shard_size)?;
    // Validate every operation, including overwritten entries, before touching
    // any path or starting a model. Final-cohort limits also apply to growth.
    let edits = update::from_mutations(changes)?;
    let count = update::validate_final_cohort(cx, base.index(), &edits)?;
    if let Some(size) = shard_size {
        sharded::validate_size(size, count)?;
    }
    validate_destination(directory)?;
    validate_destination(receipt)?;
    let directory = new_path(directory)?;
    let receipt = new_path(receipt)?;
    let old_root = fs::canonicalize(update::source_directory(base.index()))?;
    if directory.starts_with(&old_root)
        || receipt.starts_with(&old_root)
        || directory.starts_with(&receipt)
        || receipt.starts_with(&directory)
    {
        return Err(bad(
            "update destinations must be outside the predecessor and must not overlap",
        ));
    }
    let sequence = expected
        .sequence
        .checked_add(1)
        .ok_or_else(|| bad("generation sequence exhausted"))?;
    let generation = new_generation(sequence)?;
    progress.candidate = Some(generation);
    progress.receipt = Some(receipt.clone());
    let edited_ids = edits.len();
    // No receipt is read and no model is loaded here. The original snapshot
    // supplies the only models, sources, precision and graph policy permitted.
    progress.stage = "building";
    let candidate = base
        .prepare_update(cx, &directory, generation, edits, BATCH_SIZE, shard_size)
        .await?;
    checkpoint(cx)?;
    progress.stage = "sealing";
    let selection = update::seal_selection(cx, candidate.index())?;
    checkpoint(cx)?;
    progress.stage = "sealed";
    Ok(Prepared {
        candidate,
        selection,
        receipt,
        edited_ids,
    })
}

async fn commit<'l>(
    prepared: &Prepared,
    live: impl Into<live::Live<'l>> + Send,
    cx: &Cx,
    progress: &mut Progress,
) -> Result<live::Snapshot> {
    let live = live.into();
    checkpoint(cx)?;
    progress.stage = "writing_receipt";
    progress.receipt_state = "uncertain";
    save_selection(&prepared.selection, &prepared.receipt)?;
    progress.receipt_state = "durable";
    progress.stage = "installing";
    // Installation has its own final cancellation checkpoint and exact-Arc
    // predecessor comparison. Saving a receipt does not authorize a stale edit.
    // Failure here leaves a complete saved candidate, never a claimed rollback.
    let installed = live.install(cx, &prepared.candidate).await?;
    progress.stage = "installed";
    // No checkpoint follows this visible effect, including before the response.
    Ok(installed)
}

pub(super) async fn execute<'l, W: Write>(
    live: impl Into<live::Live<'l>> + Send,
    cx: &Cx,
    request: Request,
    ordinal: u64,
    allowed: bool,
    output: &mut W,
) -> Result<()> {
    let live = live.into();
    let Request::Update {
        id,
        expected_generation,
        index_dir,
        new_receipt,
        changes,
        shard_size,
    } = request;
    let mut progress = Progress::default();
    let result: Result<serde_json::Value> = async {
        checkpoint(cx)?;
        validate_id(id.as_deref())?;
        if !allowed {
            return Err(bad("updates are disabled; start serve with --allow-updates for a trusted writing controller"));
        }
        let base = live.snapshot(cx).await?;
        let prepared = match &base {
            live::Snapshot::Single(base) => {
                update::validate_partition_policy(false, shard_size)?;
                prepare(
                    base, cx, expected_generation, &index_dir, &new_receipt, changes, &mut progress,
                ).await?
            }
            live::Snapshot::Sharded(_) => {
                prepare_with_partition(
                    &base, cx, expected_generation, &index_dir, &new_receipt, changes,
                    shard_size, &mut progress,
                ).await?
            }
        };
        let installed = commit(&prepared, live, cx, &mut progress).await?;
        let mut payload = serde_json::json!({
            "ok": true, "status": "complete", "generation": installed.generation(),
            "previous_generation": base.generation(), "selection_changed": true,
            "receipt": prepared.receipt, "receipt_state": "durable",
            "selection": prepared.selection, "edited_ids": prepared.edited_ids,
            "stage": "installed",
        });
        installed.index().annotate(&mut payload);
        Ok(payload)
    }.await;
    let mut payload = match result {
        Ok(payload) => payload,
        Err(error) => {
            let mut payload = query::failure(error.as_ref());
            payload["ok"] = serde_json::json!(false);
            payload["stage"] = serde_json::json!(progress.stage);
            payload["previous_generation"] = serde_json::json!(progress.previous);
            payload["candidate_generation"] = serde_json::json!(progress.candidate);
            payload["receipt"] = serde_json::json!(progress.receipt);
            payload["receipt_state"] = serde_json::json!(progress.receipt_state);
            payload["saved_not_installed"] = serde_json::json!(progress.receipt_state == "durable");
            // Describes this request's effects, not a claim that no other owner
            // changed the shared live handle while it was preparing the update.
            payload["selection_changed"] = serde_json::json!(false);
            payload
        }
    };
    payload["schema"] = serde_json::json!(SCHEMA);
    payload["event"] = serde_json::json!("terminal");
    payload["operation"] = serde_json::json!("update");
    payload["request"] = serde_json::json!(ordinal);
    payload["id"] = serde_json::json!(id.as_deref().filter(|id| validate_id(Some(id)).is_ok()));
    payload["seq"] = serde_json::json!(0);
    payload["partial_results"] = serde_json::json!(false);
    payload["activation_scope"] = serde_json::json!("process_local");
    // Never append a failure frame after a possibly partial success frame.
    // A lost acknowledgement is reconciled through status plus the named receipt.
    emit(output, &payload)
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
#[path = "warm_update_tests.rs"]
mod tests;

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
#[path = "warm_update_sharded_tests.rs"]
mod sharded_tests;
