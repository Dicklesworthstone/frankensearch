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
    ArtifactGenerationIdentityV1, Cx, GenerationComponentReceiptV1, IndexableDocument,
    MAX_DOCUMENTS, MAX_INPUT_BYTES, MAX_RECORD_BYTES, NativeBuiltHybridIndex, Options, Result,
    SCHEMA, SELECTION_SCHEMA, Selection, Write, bad, emit, filter, query, sharded, validate_query,
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
                let (generation, reader, documents) = NativeBuiltHybridIndex::open_selected_lexical(
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
        filter: Option<&filter::Filter>,
    ) -> Result<serde_json::Value> {
        checkpoint(cx)?;
        validate_query(text)?;
        if limit == 0 || limit > 1_000 {
            return Err(bad("keyword result limit must be between 1 and 1000"));
        }
        let allowed = match filter {
            Some(filter) => {
                filter.validate()?;
                let mut allowed = BTreeSet::new();
                for document in &self.documents {
                    checkpoint(cx)?;
                    if filter.matches(document) {
                        allowed.insert(document.id.as_str());
                    }
                }
                Some(allowed)
            }
            None => None,
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
            let batch = self.reader.search_candidates(cx, &query_text, width).await?;
            checkpoint(cx)?;
            if batch.results().len() > width {
                return Err(bad("keyword candidate response exceeds its declared window"));
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
    let deadline = policy.start(cx, None)?;
    if options.stream {
        emit(
            output,
            &serde_json::json!({
                "schema": SCHEMA, "ok": true, "event": "started", "mode": "lexical",
                "generation": index.generation, "request": 1, "id": null, "seq": 0,
                "limit": options.limit, "semantic_components_verified": false,
            }),
        )?;
    }
    let result = query::within(
        cx,
        deadline.as_ref(),
        index.search(cx, text, options.limit, options.filter.as_ref()),
    )
    .await;
    match result {
        Ok(mut page) => {
            if options.stream {
                page["request"] = serde_json::json!(1);
                page["id"] = serde_json::Value::Null;
                page["seq"] = serde_json::json!(1);
            }
            emit(output, &page)?;
            if options.stream {
                emit(
                    output,
                    &serde_json::json!({
                        "schema": SCHEMA, "event": "terminal", "ok": true, "status": "complete",
                        "generation": index.generation, "request": 1, "id": null, "seq": 2,
                        "partial_results": false, "retrieval": "lexical",
                    }),
                )?;
            }
            Ok(())
        }
        Err(error) => {
            if options.stream {
                let mut terminal = query::failure(error.as_ref());
                terminal["schema"] = serde_json::json!(SCHEMA);
                terminal["event"] = serde_json::json!("terminal");
                terminal["ok"] = serde_json::json!(false);
                terminal["generation"] = serde_json::json!(index.generation);
                terminal["request"] = serde_json::json!(1);
                terminal["id"] = serde_json::Value::Null;
                terminal["seq"] = serde_json::json!(1);
                terminal["partial_results"] = serde_json::json!(false);
                emit(output, &terminal)?;
            }
            Err(error)
        }
    }
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
#[path = "lexical_tests.rs"]
mod tests;
