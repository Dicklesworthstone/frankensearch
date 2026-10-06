//! Explicit serving activation, not filesystem publication or receipt discovery.
//!
//! The controller supplying stdin is trusted to select receipts only when the
//! owner starts `serve --allow-activation`. No path is read on a refused request.
//! Existing native live admission owns producer joins and atomic installation.

use super::validate_id;
use crate::{
    ArtifactGenerationIdentityV1, Cx, Deserialize, Path, PathBuf, Result, SCHEMA, Selection, Write,
    bad, emit, live,
};

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Request {
    Activate {
        id: Option<String>,
        receipt: PathBuf,
        expected_generation: ArtifactGenerationIdentityV1,
    },
    Status {
        id: Option<String>,
    },
}

impl Request {
    fn id(&self) -> Option<&str> {
        match self {
            Self::Activate { id, .. } | Self::Status { id } => id.as_deref(),
        }
    }

    fn operation(&self) -> &'static str {
        match self {
            Self::Activate { .. } => "activate",
            Self::Status { .. } => "status",
        }
    }
}

async fn install(
    live: live::Live<'_>,
    base: &live::Snapshot,
    cx: &Cx,
    receipt: &Path,
    expected: ArtifactGenerationIdentityV1,
) -> Result<live::Snapshot> {
    cx.checkpoint()
        .map_err(|_| bad("native activation cancelled"))?;
    expected.validate()?;
    if expected != base.generation() {
        return Err(bad(
            "expected_generation does not match the serving snapshot; no activation performed",
        ));
    }
    let path = receipt
        .to_str()
        .ok_or_else(|| bad("receipt path must be UTF-8"))?;
    if !receipt.is_absolute() || path.len() > 4096 || path.contains('\0') {
        return Err(bad(
            "activation requires an absolute NUL-free receipt path of at most 4096 bytes",
        ));
    }
    // Read exactly the controller's receipt. Never search for the highest
    // sequence, recalculate a missing digest or reinterpret a rejected cohort.
    let selection = Selection::read(receipt)?;
    if selection.generation.sequence <= base.generation().sequence {
        return Err(bad(
            "activation requires a strictly newer generation; rollback and same-generation replay are refused",
        ));
    }
    let (fast_producer, quality_producer) = base.index().producers()?;
    if fast_producer != selection.fast_producer || quality_producer != selection.quality_producer {
        return Err(bad(
            "activation must preserve the retained producers and quality-tier presence",
        ));
    }
    let candidate = base.prepare_selected(cx, &selection).await?;
    // install checks the *same predecessor Arc* under the cancel-aware write
    // lock. A concurrently prepared successor cannot win by sequence alone.
    // There is no cancellation point after successful installation.
    live.install(cx, &candidate).await
}

pub(super) async fn execute<W: Write>(
    live: live::Live<'_>,
    cx: &Cx,
    request: &Request,
    ordinal: u64,
    allow_activation: bool,
    output: &mut W,
) -> Result<()> {
    let base = live.snapshot(cx).await?;
    let outcome = async {
        validate_id(request.id())?;
        match request {
            Request::Status { .. } => Ok(base.clone()),
            Request::Activate { receipt, expected_generation, .. } => {
                if !allow_activation {
                    return Err(bad("activation is disabled; start serve with --allow-activation for a trusted controller"));
                }
                install(live, &base, cx, receipt, *expected_generation).await
            }
        }
    }.await;
    let payload = match outcome {
        Ok(snapshot) => {
            let index = snapshot.index();
            let mut payload = serde_json::json!({
                "schema": SCHEMA, "event": "terminal", "operation": request.operation(),
                "ok": true, "status": "complete", "request": ordinal,
                "id": request.id(), "seq": 0, "partial_results": false,
                "generation": snapshot.generation(), "previous_generation": base.generation(),
                "selection_changed": matches!(request, Request::Activate { .. }),
                "activation_scope": "process_local", "activation_enabled": allow_activation,
                "documents": index.document_count(), "quality": index.has_quality(),
                "fast_native_hnsw": index.all_native_hnsw(false),
                "quality_native_hnsw": index.all_native_hnsw(true),
            });
            index.annotate(&mut payload);
            payload
        }
        Err(error) => serde_json::json!({
            "schema": SCHEMA, "event": "terminal", "operation": request.operation(),
            "ok": false, "status": "failed", "request": ordinal,
            "id": request.id().filter(|id| validate_id(Some(id)).is_ok()),
            "seq": 0, "partial_results": false, "generation": base.generation(),
            "selection_changed": false, "activation_scope": "process_local",
            "error": error.to_string(),
        }),
    };
    // Failed delivery after installation must stop serving, not roll back or
    // emit a contradictory "activation failed" frame. A status request on a
    // surviving externally owned live handle can resolve a lost acknowledgement.
    emit(output, &payload)
}
