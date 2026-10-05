//! Warm stdio serving with explicit whole-cohort activation between requests.
//! Every phase is flushed before the next phase is polled. No result cache,
//! model reload, independent tier refresh, or detached request task is involved.

use frankensearch::native_ann::{NativePhaseCandidates, NativeSearchPhase};
pub(super) use frankensearch::native_ann::builder::live::NativeLiveHybridIndex;

use super::*;

#[path = "activation.rs"]
mod activation;

const MAX_REQUEST_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ActivationPermission {
    Disabled,
    Enabled,
}

// Unknown control fields cannot fall through into an unrestricted search.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Message {
    Search(Request),
    Control(activation::Request),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    pub(super) id: Option<String>,
    pub(super) query: String,
    pub(super) mode: Option<Mode>,
    pub(super) limit: Option<usize>,
    pub(super) filter: Option<filter::Filter>,
}

impl Request {
    fn validate(&self) -> Result<()> {
        validate_query(&self.query)?;
        validate_id(self.id.as_deref())?;
        if let Some(filter) = &self.filter {
            filter.validate()?;
        }
        if self.limit.is_some_and(|limit| limit == 0 || limit > 1_000) {
            return Err(bad("request limit must be between 1 and 1000"));
        }
        Ok(())
    }
}

fn validate_id(id: Option<&str>) -> Result<()> {
    if id.is_some_and(|id| id.len() > 256 || id.contains('\0')) {
        return Err(bad("request id must be NUL-free and at most 256 bytes"));
    }
    Ok(())
}

// A failed output write may already have exposed part of a frame. It must end
// the session, not be disguised as a recoverable query error followed by JSON.
enum Failure {
    Query(Box<dyn Error + Send + Sync>),
    Delivery(Box<dyn Error + Send + Sync>),
}

struct Frames<'a, W> {
    output: &'a mut W,
    request: u64,
    id: Option<&'a str>,
    generation: ArtifactGenerationIdentityV1,
    seq: u64,
    partial: bool,
}

impl<W: Write> Frames<'_, W> {
    fn send(&mut self, mut payload: serde_json::Value) -> Result<()> {
        payload["schema"] = serde_json::json!(SCHEMA);
        payload["request"] = serde_json::json!(self.request);
        payload["id"] = serde_json::json!(self.id);
        payload["seq"] = serde_json::json!(self.seq);
        payload["generation"] = serde_json::json!(self.generation);
        emit(self.output, &payload)?;
        self.seq += 1;
        Ok(())
    }
}

async fn phases<W: Write>(
    index: &NativeBuiltHybridIndex,
    cx: &Cx,
    request: &Request,
    mode: Mode,
    limit: usize,
    frames: &mut Frames<'_, W>,
    filters: filter::Filters<'_>,
) -> std::result::Result<bool, Failure> {
    let scoped = filter::Query::prepare(index, cx, filters).map_err(Failure::Query)?;
    let mut started = serde_json::json!({
        "event": "started", "ok": true, "mode": mode, "limit": limit,
    });
    scoped.annotate(&mut started);
    frames.send(started).map_err(Failure::Delivery)?;
    if mode != Mode::Full {
        let payload = scoped.search(cx, &request.query, mode, limit)
            .await
            .map_err(Failure::Query)?;
        frames.send(payload).map_err(Failure::Delivery)?;
        frames.partial = true;
        return Ok(false);
    }
    let mut stream = scoped
        .progressive(cx, &request.query, limit)
        .map_err(|error| Failure::Query(error.into()))?;
    let mut degraded = false;
    while let Some(phase) = stream
        .next_phase()
        .await
        .map_err(|error| Failure::Query(error.into()))?
    {
        let mut payload = match phase {
            NativeSearchPhase::Initial { results, candidates } => {
                result_frame("initial", results, candidates)
            }
            NativeSearchPhase::Refined { results, candidates } => {
                result_frame("refined", results, candidates)
            }
            NativeSearchPhase::Reranked { results, candidates, evaluated } => {
                let mut frame = result_frame("reranked", results, candidates);
                frame["evaluated"] = serde_json::json!(evaluated);
                frame
            }
            NativeSearchPhase::RefinementFailed { initial_results, error } => {
                degraded = true;
                serde_json::json!({
                    "event": "results", "ok": false, "phase": "refinement_failed",
                    "results": initial_results, "error": error.to_string(),
                })
            }
            NativeSearchPhase::RerankFailed { previous_results, error } => {
                degraded = true;
                serde_json::json!({
                    "event": "results", "ok": false, "phase": "rerank_failed",
                    "results": previous_results, "error": error.to_string(),
                })
            }
        };
        scoped.annotate(&mut payload);
        frames.send(payload).map_err(Failure::Delivery)?;
        frames.partial = true;
    }
    Ok(degraded)
}

fn result_frame(
    phase: &str,
    results: Vec<frankensearch::ScoredResult>,
    candidates: NativePhaseCandidates,
) -> serde_json::Value {
    serde_json::json!({
        "event": "results", "ok": true, "phase": phase, "results": results,
        "candidates": {
            "fast": candidates.fast, "quality": candidates.quality, "lexical": candidates.lexical,
        },
    })
}

/// Returns false for a failed/degraded request that was fully reported. Output
/// failure returns Err and must stop the process, even when more input exists.
pub(super) async fn stream_one<W: Write>(
    index: &NativeBuiltHybridIndex,
    cx: &Cx,
    request: &Request,
    ordinal: u64,
    defaults: (Mode, usize),
    output: &mut W,
    base_filter: Option<&filter::Filter>,
) -> Result<bool> {
    let mut frames = Frames {
        output,
        request: ordinal,
        id: request.id.as_deref(),
        generation: index.vectors().fast().index().owner_witness().generation,
        seq: 0,
        partial: false,
    };
    let mode = request.mode.unwrap_or(defaults.0);
    let limit = request.limit.unwrap_or(defaults.1);
    let result = match request.validate() {
        Ok(()) => phases(index, cx, request, mode, limit, &mut frames,
            [base_filter, request.filter.as_ref()]).await,
        Err(error) => Err(Failure::Query(error)),
    };
    match result {
        Ok(degraded) => {
            frames.send(serde_json::json!({
                "event": "terminal", "ok": !degraded,
                "status": if degraded { "degraded" } else { "complete" },
                "partial_results": degraded && frames.partial,
            }))?;
            Ok(!degraded)
        }
        Err(Failure::Query(error)) => {
            frames.send(serde_json::json!({
                "event": "terminal", "ok": false, "status": "failed",
                "partial_results": frames.partial, "error": error.to_string(),
            }))?;
            Ok(false)
        }
        Err(Failure::Delivery(error)) => Err(error),
    }
}

pub(super) async fn run<R: BufRead, W: Write>(
    live: &NativeLiveHybridIndex,
    cx: &Cx,
    input: &mut R,
    output: &mut W,
    defaults: (Mode, usize),
    allow_activation: bool,
    base_filter: Option<&filter::Filter>,
) -> Result<()> {
    cx.checkpoint().map_err(|_| bad("native serving cancelled"))?;
    if let Some(filter) = base_filter {
        filter.validate()?;
    }
    let initial = live.snapshot(cx).await?;
    let index = initial.index();
    emit(output, &serde_json::json!({
        "schema": SCHEMA, "event": "ready", "ok": true,
        "generation": index.vectors().fast().index().owner_witness().generation,
        "documents": index.vectors().documents().len(),
        "quality": index.vectors().quality().is_some(),
        "fast_native_hnsw": index.vectors().fast().graph_path().is_some(),
        "activation_enabled": allow_activation,
        "default_filter_applied": base_filter.is_some(),
    }))?;
    drop(initial);
    let mut line = Vec::new();
    let mut ordinal = 0_u64;
    loop {
        // Standard input is a blocking read on the owning command lane. EOF
        // exits normally. No preemptible idle read or detached input task is claimed.
        cx.checkpoint().map_err(|_| bad("native serving cancelled"))?;
        line.clear();
        let count = (&mut *input)
            .take(MAX_REQUEST_BYTES as u64 + 1)
            .read_until(b'\n', &mut line)?;
        cx.checkpoint().map_err(|_| bad("native serving cancelled"))?;
        if count == 0 {
            return Ok(());
        }
        if line.len() > MAX_REQUEST_BYTES {
            return Err(bad("serve request exceeds 1 MiB; session stopped without draining input"));
        }
        let raw = line.trim_ascii();
        if raw.is_empty() {
            continue;
        }
        if raw == b"quit" || raw == b"exit" {
            return Ok(());
        }
        ordinal = ordinal.checked_add(1).ok_or_else(|| bad("request ordinal exhausted"))?;
        let message: Message = match serde_json::from_slice(raw) {
            Ok(request) => request,
            Err(error) => {
                let snapshot = live.snapshot(cx).await?;
                emit(output, &serde_json::json!({
                    "schema": SCHEMA, "event": "terminal", "ok": false, "status": "failed",
                    "request": ordinal, "id": null, "seq": 0, "partial_results": false,
                    "generation": snapshot.generation(),
                    "error": format!("invalid request ({:?} at column {})", error.classify(), error.column()),
                }))?;
                continue;
            }
        };
        match message {
            Message::Search(request) => {
                // Pin once, through Initial, quality, hydration and terminal
                // delivery. Never resolve the serving selection between phases.
                let snapshot = live.snapshot(cx).await?;
                let _complete =
                    stream_one(snapshot.index(), cx, &request, ordinal, defaults, output, base_filter).await?;
            }
            Message::Control(request) => {
                activation::execute(live, cx, &request, ordinal, allow_activation, output)
                    .await?;
            }
        }
    }
}
