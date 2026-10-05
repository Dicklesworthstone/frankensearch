//! Explicit serving activation, not filesystem publication or receipt discovery.
//!
//! The controller supplying stdin is trusted to select receipts only when the
//! owner starts `serve --allow-activation`. No path is read on a refused request.
//! Existing native live admission owns producer joins and atomic installation.

use frankensearch::native_ann::builder::NativeHybridReopenLimits;
use frankensearch::native_ann::builder::live::NativeHybridSnapshot;

use super::{NativeLiveHybridIndex, validate_id};
use crate::{
    ArtifactGenerationIdentityV1, Cx, Deserialize, GenerationComponentReceiptV1, MAX_DOCUMENTS,
    Path, PathBuf, Result, SCHEMA, Selection, Write, bad, emit,
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
    live: &NativeLiveHybridIndex,
    base: &NativeHybridSnapshot,
    cx: &Cx,
    receipt: &Path,
    expected: ArtifactGenerationIdentityV1,
) -> Result<NativeHybridSnapshot> {
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
    let vectors = base.index().vectors();
    if vectors.fast().embedder().identity()?.fingerprint() != selection.fast_producer
        || vectors
            .quality()
            .map(|tier| {
                tier.embedder()
                    .identity()
                    .map(|identity| identity.fingerprint())
            })
            .transpose()?
            != selection.quality_producer
    {
        return Err(bad(
            "activation must preserve the retained producers and quality-tier presence",
        ));
    }
    let snapshot_receipt = GenerationComponentReceiptV1 {
        byte_len: selection.snapshot.byte_len,
        sha256: selection.snapshot.sha256,
    };
    let mut limits = NativeHybridReopenLimits::default();
    limits.vectors.max_documents = MAX_DOCUMENTS;
    let candidate = base
        .prepare_selected(cx, &selection.directory, &snapshot_receipt, limits)
        .await?;
    let admitted = candidate.index().vectors();
    if admitted.documents().len() != selection.documents
        || admitted.fast().index().owner_witness().generation != selection.generation
    {
        return Err(bad(
            "admitted successor differs from the trusted receipt; serving selection unchanged",
        ));
    }
    // install checks the *same predecessor Arc* under the cancel-aware write
    // lock. A concurrently prepared successor cannot win by sequence alone.
    // There is no cancellation point after successful installation.
    Ok(live.install(cx, &candidate).await?)
}

pub(super) async fn execute<W: Write>(
    live: &NativeLiveHybridIndex,
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
            let vectors = snapshot.index().vectors();
            serde_json::json!({
                "schema": SCHEMA, "event": "terminal", "operation": request.operation(),
                "ok": true, "status": "complete", "request": ordinal,
                "id": request.id(), "seq": 0, "partial_results": false,
                "generation": snapshot.generation(), "previous_generation": base.generation(),
                "selection_changed": matches!(request, Request::Activate { .. }),
                "activation_scope": "process_local", "activation_enabled": allow_activation,
                "documents": vectors.documents().len(), "quality": vectors.quality().is_some(),
                "fast_native_hnsw": vectors.fast().graph_path().is_some(),
                "quality_native_hnsw": vectors.quality().is_some_and(|tier| tier.graph_path().is_some()),
            })
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
