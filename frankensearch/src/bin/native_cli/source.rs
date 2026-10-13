//! Bounded source retrieval from the exact snapshot that produced a result.
//!
//! No document ID is interpreted as a path. Hybrid and keyword-only serving
//! supply their retained source lookup, not a new read of the current filesystem.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    ArtifactGenerationIdentityV1, Cx, IndexableDocument, Result, SCHEMA, Write, bad, encode,
    filter, query, serve,
};

pub const MAX_DOCUMENTS: usize = 32;
pub const MAX_CONTENT_BYTES: usize = 64 * 1024;
pub const MAX_RESPONSE_BYTES: usize = 1024 * 1024;
const MAX_ID_BYTES: usize = 64 * 1024;

const fn default_content_bytes() -> usize {
    8 * 1024
}

/// A UTF-8 byte window in one selected document, not an arbitrary file read.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DocumentRequest {
    doc_id: String,
    #[serde(default)]
    offset_bytes: usize,
    #[serde(default = "default_content_bytes")]
    max_bytes: usize,
    #[serde(default)]
    include_metadata: bool,
}

/// The expected generation is mandatory: a later live selection must not answer
/// a request for the text behind an earlier search result, even for the same ID.
#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Request {
    Fetch {
        id: Option<String>,
        expected_generation: ArtifactGenerationIdentityV1,
        documents: Vec<DocumentRequest>,
        filter: Option<filter::Filter>,
        timeout_ms: Option<u64>,
    },
}

impl Request {
    pub fn id(&self) -> Option<&str> {
        let Self::Fetch { id, .. } = self;
        id.as_deref()
    }

    fn validate(&self) -> Result<()> {
        let Self::Fetch {
            expected_generation,
            documents,
            filter,
            timeout_ms,
            ..
        } = self;
        serve::validate_id(self.id())?;
        expected_generation.validate()?;
        query::validate_timeout(*timeout_ms)?;
        if documents.is_empty() || documents.len() > MAX_DOCUMENTS {
            return Err(bad("fetch requires between 1 and 32 document windows"));
        }
        let mut remaining = MAX_ID_BYTES;
        for document in documents {
            if document.doc_id.trim().is_empty()
                || document.doc_id.contains('\0')
                || document.doc_id.len() > usize::from(u16::MAX)
                || document.doc_id.len() > remaining
            {
                return Err(bad(
                    "fetch IDs must be nonblank, NUL-free, at most 65535 bytes each and 64 KiB in total",
                ));
            }
            remaining -= document.doc_id.len();
            // Four bytes fit any one UTF-8 scalar. End rounding can therefore
            // never return a non-final empty window that makes no progress.
            if !(4..=MAX_CONTENT_BYTES).contains(&document.max_bytes) {
                return Err(bad("fetch max_bytes must be between 4 and 65536"));
            }
        }
        if let Some(filter) = filter {
            filter.validate()?;
        }
        Ok(())
    }
}

#[derive(Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum DocumentResponse<'a> {
    Found {
        doc_id: &'a str,
        content: &'a str,
        offset_bytes: usize,
        end_offset_bytes: usize,
        total_bytes: usize,
        next_offset_bytes: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        metadata: Option<&'a HashMap<String, String>>,
    },
    // Missing and out-of-scope sources have the identical response shape. No
    // source length, title, metadata, or reason can disclose which case occurred.
    Unavailable { doc_id: &'a str },
}

#[derive(Serialize)]
struct Response<'a> {
    schema: &'static str,
    event: &'static str,
    operation: &'static str,
    status: &'static str,
    ok: bool,
    partial_results: bool,
    request: u64,
    id: Option<&'a str>,
    seq: u64,
    generation: ArtifactGenerationIdentityV1,
    source: &'static str,
    documents: Vec<DocumentResponse<'a>>,
}

fn checkpoint(cx: &Cx) -> Result<()> {
    cx.checkpoint().map_err(|_| frankensearch::SearchError::Cancelled {
        phase: "native_cli.source".to_owned(),
        reason: "source retrieval cancelled".to_owned(),
    })?;
    Ok(())
}

fn render<'a>(
    cx: &Cx,
    generation: ArtifactGenerationIdentityV1,
    request: &'a Request,
    ordinal: u64,
    base_filter: Option<&filter::Filter>,
    lookup: impl Fn(&str) -> Option<&'a IndexableDocument>,
) -> Result<Vec<u8>> {
    checkpoint(cx)?;
    request.validate()?;
    let Request::Fetch {
        expected_generation,
        documents,
        filter,
        ..
    } = request;
    if *expected_generation != generation {
        return Err(bad("fetch expected_generation differs from the retained snapshot; search again or use the original selection"));
    }
    if let Some(filter) = base_filter {
        filter.validate()?;
    }
    let mut responses = Vec::with_capacity(documents.len());
    for window in documents {
        checkpoint(cx)?;
        let document = lookup(&window.doc_id);
        checkpoint(cx)?;
        let Some(document) = document.filter(|document| {
            [base_filter, filter.as_ref()]
                .into_iter()
                .flatten()
                .all(|filter| filter.matches(document))
        }) else {
            responses.push(DocumentResponse::Unavailable {
                doc_id: &window.doc_id,
            });
            continue;
        };
        // The supplied reader must resolve the requested ID, never merely the
        // same ordinal in another partition. Do not echo foreign source fields.
        if document.id != window.doc_id {
            return Err(bad("retained source lookup returned a different document ID"));
        }
        let start = window.offset_bytes;
        let text = &document.content;
        if start > text.len() || !text.is_char_boundary(start) {
            return Err(bad("fetch offset_bytes must be a UTF-8 boundary within the document"));
        }
        let mut end = start.saturating_add(window.max_bytes).min(text.len());
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        responses.push(DocumentResponse::Found {
            doc_id: &document.id,
            content: &text[start..end],
            offset_bytes: start,
            end_offset_bytes: end,
            total_bytes: text.len(),
            next_offset_bytes: (end < text.len()).then_some(end),
            title: if window.include_metadata {
                document.title.as_deref()
            } else {
                None
            },
            metadata: window.include_metadata.then_some(&document.metadata),
        });
    }
    checkpoint(cx)?;
    // Borrow source bodies and metadata directly. The existing LimitedBuffer
    // bounds encoded bytes (including escaping and the newline) BEFORE output;
    // do not first clone unbounded fields into a serde_json::Value.
    let outcome = encode(
        &Response {
            schema: SCHEMA,
            event: "terminal",
            operation: "fetch",
            status: "complete",
            ok: true,
            partial_results: false,
            request: ordinal,
            id: request.id(),
            seq: 0,
            generation,
            source: "retained_snapshot",
            documents: responses,
        },
        MAX_RESPONSE_BYTES,
    );
    checkpoint(cx)?;
    outcome
}

/// Execute on one already-pinned source owner. The server retains that owner
/// through delivery; no worker, model call, source reopen or live refresh occurs.
/// Errors are one terminal frame. A failed write/flush stops serving without a
/// second frame, since some bytes may already be visible.
#[allow(clippy::too_many_arguments)]
pub async fn execute<'a, W: Write>(
    cx: &Cx,
    generation: ArtifactGenerationIdentityV1,
    request: &'a Request,
    ordinal: u64,
    base_filter: Option<&filter::Filter>,
    policy: &query::Policy,
    lookup: impl Fn(&str) -> Option<&'a IndexableDocument>,
    output: &mut W,
) -> Result<()> {
    let result = async {
        checkpoint(cx)?;
        request.validate()?;
        let Request::Fetch { timeout_ms, .. } = request;
        let deadline = policy.start(cx, *timeout_ms)?;
        query::within(cx, deadline.as_ref(), async {
            render(cx, generation, request, ordinal, base_filter, lookup)
        })
        .await
    }
    .await;
    let bytes = match result {
        Ok(bytes) => bytes,
        Err(error) => {
            let mut payload = query::failure(error.as_ref());
            payload["schema"] = serde_json::json!(SCHEMA);
            payload["event"] = serde_json::json!("terminal");
            payload["operation"] = serde_json::json!("fetch");
            payload["ok"] = serde_json::json!(false);
            payload["partial_results"] = serde_json::json!(false);
            payload["generation"] = serde_json::json!(generation);
            payload["request"] = serde_json::json!(ordinal);
            payload["id"] = serde_json::json!(request.id().filter(|id| serve::validate_id(Some(id)).is_ok()));
            payload["seq"] = serde_json::json!(0);
            encode(&payload, MAX_RESPONSE_BYTES)?
        }
    };
    // Delivery is outside the deadline. No post-write checkpoint can describe
    // a successfully delivered frame as cancelled or rolled back.
    output.write_all(&bytes)?;
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests;
