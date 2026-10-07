//! Explicit keyword-only search over authenticated native source/Quill snapshots.
//! This module never loads a model, opens a vector, selects a new head or catches
//! a failed semantic open as permission to downgrade the requested search mode.

use std::collections::BTreeSet;

use frankensearch::native_ann::builder::NativeHybridReopenLimits;
use frankensearch::native_ann::builder::sharded::{
    NativeBuiltShardedHybridIndex, NativeShardedHybridReopenLimits,
};
use frankensearch::{LexicalRead, SearchError};
use frankensearch_quill::QuillSearchIndex;

use super::{
    ArtifactGenerationIdentityV1, BufRead, Cx, Deserialize, GenerationComponentReceiptV1,
    IndexableDocument, MAX_DOCUMENTS, MAX_INPUT_BYTES, MAX_RECORD_BYTES, NativeBuiltHybridIndex,
    Options, Result, SCHEMA, SELECTION_SCHEMA, Selection, Write, bad, emit, filter, query, serve,
    sharded, validate_query,
};

struct Opened {
    generation: ArtifactGenerationIdentityV1,
    reader: QuillSearchIndex,
    documents: Vec<IndexableDocument>,
    layout: &'static str,
}

fn checkpoint(cx: &Cx) -> Result<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: "native_cli.lexical".to_owned(),
        reason: error.to_string(),
    })?;
    Ok(())
}

impl Opened {
    async fn open(cx: &Cx, selection: &Selection) -> Result<Self> {
        checkpoint(cx)?;
        let expected = GenerationComponentReceiptV1 {
            byte_len: selection.snapshot.byte_len,
            sha256: selection.snapshot.sha256,
        };
        let (generation, reader, documents, layout) = match selection.schema.as_str() {
            SELECTION_SCHEMA => {
                let mut limits = NativeHybridReopenLimits::default();
                limits.vectors.max_documents = MAX_DOCUMENTS;
                limits.vectors.max_source_bytes = MAX_INPUT_BYTES as u64;
                limits.vectors.max_document_bytes = MAX_RECORD_BYTES as u64;
                let (generation, reader, documents) =
                    NativeBuiltHybridIndex::open_selected_lexical(
                        cx,
                        &selection.directory,
                        &expected,
                        limits,
                    )
                    .await?;
                (generation, reader, documents, "single")
            }
            sharded::SELECTION_SCHEMA => {
                let mut limits = NativeShardedHybridReopenLimits::default();
                limits.vectors.max_documents = MAX_DOCUMENTS;
                limits.vectors.max_artifact_bytes = MAX_INPUT_BYTES as u64;
                limits.vectors.partition.max_documents = MAX_DOCUMENTS;
                limits.vectors.partition.max_source_bytes = MAX_INPUT_BYTES as u64;
                limits.vectors.partition.max_document_bytes = MAX_RECORD_BYTES as u64;
                let (generation, reader, documents) =
                    NativeBuiltShardedHybridIndex::open_selected_lexical(
                        cx,
                        &selection.directory,
                        &expected,
                        limits,
                    )
                    .await?;
                (generation, reader, documents, "sharded")
            }
            _ => {
                return Err(bad(
                    "unknown keyword selection layout; no discovery or fallback",
                ));
            }
        };
        if generation != selection.generation || documents.len() != selection.documents {
            return Err(bad(
                "keyword snapshot differs from the trusted receipt's generation or count",
            ));
        }
        checkpoint(cx)?;
        Ok(Self {
            generation,
            reader,
            documents,
            layout,
        })
    }

    fn document(&self, id: &str) -> Option<&IndexableDocument> {
        let ordinal = self
            .documents
            .binary_search_by(|doc| doc.id.as_str().cmp(id))
            .ok()?;
        self.documents.get(ordinal)
    }

    async fn search(
        &self,
        cx: &Cx,
        text: &str,
        limit: usize,
        filters: filter::Filters<'_>,
    ) -> Result<serde_json::Value> {
        checkpoint(cx)?;
        validate_query(text)?;
        if limit == 0 || limit > 1_000 {
            return Err(bad("keyword result limit must be between 1 and 1000"));
        }
        for filter in filters.iter().flatten() {
            filter.validate()?;
        }
        let allowed = if filters.iter().any(Option::is_some) {
            let mut allowed = BTreeSet::new();
            for document in &self.documents {
                checkpoint(cx)?;
                // A request may narrow a fixed server scope, never replace it.
                if filters
                    .iter()
                    .flatten()
                    .all(|filter| filter.matches(document))
                {
                    allowed.insert(document.id.as_str());
                }
            }
            Some(allowed)
        } else {
            None
        };
        let eligible = allowed.as_ref().map_or(self.documents.len(), BTreeSet::len);
        let target = limit.min(eligible);
        let mut results = Vec::new();
        if target != 0 {
            // An immutable reader makes one complete candidate request safe.
            // Selective scopes collect the entire matching population BEFORE
            // cutoff/hydration, rather than underfilling an unfiltered top-k.
            // This deliberately trades candidate memory for a simple exact
            // scope contract; the already-admitted source ceiling bounds it.
            let width = if eligible == self.documents.len() {
                target
            } else {
                self.documents.len()
            };
            let query_text = self.reader.query_text(text);
            let batch = self
                .reader
                .search_candidates(cx, &query_text, width)
                .await?;
            checkpoint(cx)?;
            if batch.results().len() > width {
                return Err(bad(
                    "keyword candidate response exceeds its declared window",
                ));
            }
            let mut seen = BTreeSet::new();
            for candidate in batch.results() {
                checkpoint(cx)?;
                if !candidate.score.is_finite()
                    || self.document(&candidate.doc_id).is_none()
                    || !seen.insert(candidate.doc_id.as_str())
                {
                    return Err(bad(
                        "keyword candidates must be finite unique members of the selected source",
                    ));
                }
            }
            drop(seen);
            let (candidates, context) = batch.into_parts();
            for candidate in candidates {
                checkpoint(cx)?;
                if results.len() < target
                    && allowed
                        .as_ref()
                        .is_none_or(|ids| ids.contains(candidate.doc_id.as_str()))
                {
                    results.push(candidate);
                }
            }
            let identities = results
                .iter()
                .map(|hit| (hit.doc_id.clone(), hit.score.to_bits()))
                .collect::<Vec<_>>();
            self.reader
                .hydrate_candidates(cx, context.as_ref(), &mut results)
                .await?;
            checkpoint(cx)?;
            for (result, (id, score)) in results.iter_mut().zip(identities) {
                if result.doc_id != id || result.score.to_bits() != score {
                    return Err(bad(
                        "keyword hydration changed the selected document or raw score",
                    ));
                }
                // No semantic image was admitted. Never advertise a bare row as
                // a physical vector address, even if a backend supplies an index.
                result.index = None;
            }
        }
        checkpoint(cx)?;
        let mut payload = serde_json::json!({
            "schema": SCHEMA, "ok": true, "event": "results", "phase": "lexical",
            "generation": self.generation, "source_layout": self.layout,
            "retrieval": "lexical", "semantic_components_verified": false,
            "results": results,
        });
        if allowed.is_some() {
            payload["scope"] = serde_json::json!({
                "filtered": true, "eligible_documents": eligible,
            });
        }
        Ok(payload)
    }
}

pub async fn execute(cx: &Cx, options: &Options, output: &mut impl Write) -> Result<()> {
    checkpoint(cx)?;
    let selection = Selection::read(&options.receipt)?;
    // Explicit keyword selection bypasses model startup, not integrity checks.
    // Admission is startup work outside the query deadline, as in hybrid search.
    let index = Opened::open(cx, &selection).await?;
    let policy = query::Policy::new(cx, options.timeout_ms)?;
    execute_opened(cx, options, &index, &policy, output).await
}

async fn execute_opened(
    cx: &Cx,
    options: &Options,
    index: &Opened,
    policy: &query::Policy,
    output: &mut impl Write,
) -> Result<()> {
    let text = options
        .query
        .as_deref()
        .ok_or_else(|| bad("keyword search requires --query"))?;
    validate_query(text)?;
    if options.stream {
        let request = serve::Request {
            id: None,
            query: text.to_owned(),
            mode: None,
            limit: Some(options.limit),
            filter: None,
            timeout_ms: None,
        };
        return stream_one(
            cx,
            index,
            &request,
            1,
            options.limit,
            options.filter.as_ref(),
            policy,
            output,
        )
        .await
        .map_err(|failure| match failure {
            serve::Failure::Query(error) | serve::Failure::Delivery(error) => error,
        });
    }
    let deadline = policy.start(cx, None)?;
    let page = query::within(
        cx,
        deadline.as_ref(),
        index.search(cx, text, options.limit, [None, options.filter.as_ref()]),
    )
    .await?;
    emit(output, &page)
}

/// Share request validation, sequencing, bounded delivery and deadline policy
/// with the ordinary server. Query errors are fully reported but remain typed
/// for the one-shot CLI; a warm session may continue after them. Delivery errors
/// are never converted into recoverable query errors or followed by more JSON.
#[allow(clippy::too_many_arguments)]
async fn stream_one<W: Write>(
    cx: &Cx,
    index: &Opened,
    request: &serve::Request,
    ordinal: u64,
    default_limit: usize,
    base_filter: Option<&filter::Filter>,
    policy: &query::Policy,
    output: &mut W,
) -> std::result::Result<(), serve::Failure> {
    use serve::Failure::{Delivery, Query};

    let id = request
        .id
        .as_deref()
        .filter(|id| serve::validate_id(Some(id)).is_ok());
    let mut frames = serve::Frames::new(output, ordinal, id, index.generation);
    let result = async {
        checkpoint(cx).map_err(Query)?;
        request.validate().map_err(Query)?;
        if request.mode.is_some() || policy.rerank.is_some() {
            return Err(Query(bad(
                "keyword-only queries cannot request semantic modes or reranking",
            )));
        }
        let limit = request.limit.unwrap_or(default_limit);
        let deadline = policy.start(cx, request.timeout_ms).map_err(Query)?;
        frames
            .send(serde_json::json!({
                "ok": true, "event": "started", "mode": "lexical", "limit": limit,
                "retrieval": "lexical", "source_layout": index.layout,
                "semantic_components_verified": false,
            }))
            .map_err(Delivery)?;
        let page = query::within(
            cx,
            deadline.as_ref(),
            index.search(
                cx,
                &request.query,
                limit,
                [base_filter, request.filter.as_ref()],
            ),
        )
        .await
        .map_err(Query)?;
        frames.send(page).map_err(Delivery)?;
        // Results are already delivered. No late cancel/timeout can retract
        // that success; a broken terminal write is solely a delivery failure.
        frames
            .send(serde_json::json!({
                "event": "terminal", "ok": true, "status": "complete",
                "partial_results": false, "retrieval": "lexical",
                "semantic_components_verified": false,
            }))
            .map_err(Delivery)
    }
    .await;
    match result {
        Err(Query(error)) => {
            let mut terminal = query::failure(error.as_ref());
            terminal["event"] = serde_json::json!("terminal");
            terminal["ok"] = serde_json::json!(false);
            terminal["partial_results"] = serde_json::json!(false);
            terminal["retrieval"] = serde_json::json!("lexical");
            terminal["semantic_components_verified"] = serde_json::json!(false);
            frames.send(terminal).map_err(Delivery)?;
            Err(Query(error))
        }
        other => other,
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Message {
    Search(serve::Request),
    Control(Control),
}

// This is the existing status operation only. Unknown controls and fields
// cannot be deserialized as a search or cause access to a supplied pathname.
#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Control {
    Status { id: Option<String> },
}

pub async fn serve_with_io<R: BufRead, W: Write>(
    cx: &Cx,
    options: &Options,
    input: &mut R,
    output: &mut W,
) -> Result<()> {
    checkpoint(cx)?;
    if options.activation != serve::ActivationPermission::Disabled || options.allow_updates {
        return Err(bad(
            "keyword-only serving retains a fixed snapshot and does not accept write controls",
        ));
    }
    let selection = Selection::read(&options.receipt)?;
    // This is the ONLY open in the session. Nothing after ready re-resolves a
    // receipt, descriptor, source file or model. The mapped Quill files must
    // remain immutable for this reader's lifetime, as for one-shot queries.
    let index = Opened::open(cx, &selection).await?;
    let policy = query::Policy::new(cx, options.timeout_ms)?;
    run_opened(
        cx,
        &index,
        input,
        output,
        options.limit,
        options.filter.as_ref(),
        &policy,
    )
    .await
}

fn describe(
    index: &Opened,
    base_filter: Option<&filter::Filter>,
    policy: &query::Policy,
) -> serde_json::Value {
    serde_json::json!({
        "schema": SCHEMA, "ok": true, "generation": index.generation,
        "documents": index.documents.len(), "source_layout": index.layout,
        "retrieval": "lexical", "semantic_components_verified": false,
        "selection_policy": "fixed_snapshot", "activation_enabled": false,
        "updates_enabled": false, "default_filter_applied": base_filter.is_some(),
        "maximum_timeout_ms": policy.maximum_ms(),
    })
}

#[allow(clippy::too_many_arguments)]
async fn run_opened<R: BufRead, W: Write>(
    cx: &Cx,
    index: &Opened,
    input: &mut R,
    output: &mut W,
    default_limit: usize,
    base_filter: Option<&filter::Filter>,
    policy: &query::Policy,
) -> Result<()> {
    checkpoint(cx)?;
    if default_limit == 0 || default_limit > 1_000 || policy.rerank.is_some() {
        return Err(bad("invalid keyword server limit or unsupported reranker"));
    }
    if let Some(filter) = base_filter {
        filter.validate()?;
    }
    let mut ready = describe(index, base_filter, policy);
    ready["event"] = serde_json::json!("ready");
    emit(output, &ready)?;
    let mut record = Vec::new();
    let mut ordinal = 0_u64;
    while serve::read_request(cx, input, &mut record)? {
        ordinal = ordinal
            .checked_add(1)
            .ok_or_else(|| bad("request ordinal exhausted"))?;
        let message = match serde_json::from_slice::<Message>(record.trim_ascii()) {
            Ok(message) => message,
            Err(error) => {
                let mut frames = serve::Frames::new(output, ordinal, None, index.generation);
                frames.send(serde_json::json!({
                    "event": "terminal", "ok": false, "status": "failed",
                    "partial_results": false, "retrieval": "lexical",
                    "semantic_components_verified": false,
                    "error": format!("invalid keyword request ({:?} at column {}); only search and status are supported", error.classify(), error.column()),
                }))?;
                continue;
            }
        };
        match message {
            Message::Search(request) => {
                match stream_one(
                    cx,
                    index,
                    &request,
                    ordinal,
                    default_limit,
                    base_filter,
                    policy,
                    output,
                )
                .await
                {
                    Ok(()) | Err(serve::Failure::Query(_)) => {}
                    Err(serve::Failure::Delivery(error)) => return Err(error),
                }
            }
            Message::Control(Control::Status { id }) => {
                let valid_id = id
                    .as_deref()
                    .filter(|id| serve::validate_id(Some(id)).is_ok());
                let mut payload = match serve::validate_id(id.as_deref()) {
                    Ok(()) => {
                        let mut status = describe(index, base_filter, policy);
                        status["status"] = serde_json::json!("complete");
                        status
                    }
                    Err(error) => {
                        let mut failure = query::failure(error.as_ref());
                        failure["ok"] = serde_json::json!(false);
                        failure
                    }
                };
                payload["event"] = serde_json::json!("terminal");
                payload["operation"] = serde_json::json!("status");
                payload["retrieval"] = serde_json::json!("lexical");
                payload["semantic_components_verified"] = serde_json::json!(false);
                payload["partial_results"] = serde_json::json!(false);
                payload["selection_changed"] = serde_json::json!(false);
                serve::Frames::new(output, ordinal, valid_id, index.generation).send(payload)?;
            }
        }
    }
    Ok(())
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
#[path = "lexical_tests.rs"]
mod tests;
