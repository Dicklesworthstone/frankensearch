//! Bound native ingestion requests before invoking either model.
//!
//! Count limits do not bound the bytes in a request. This module partitions
//! prepared documents without copying, truncating, retokenizing or omitting
//! content. The ordinary tier writers retain all identity and output checks.

use super::{NativeIndexBuilder, checkpoint, invalid};
use crate::{Cx, IndexableDocument, SearchResult};

impl NativeIndexBuilder {
    /// Limit the sum of prepared UTF-8 content bytes in each inference batch.
    ///
    /// Off by default. Combines with [`Self::with_batch_size`]: a batch must
    /// satisfy BOTH limits. Documents are not split or truncated, and empty
    /// content still consumes one document slot. Both tiers receive the same
    /// document groups; incremental reuse can make a tier's actual request
    /// smaller. Titles and metadata are not charged because native embedding
    /// input is exactly `IndexableDocument::content`.
    ///
    /// An individually oversized document rejects the whole build during source
    /// preflight, before directory creation or inference, even when that document
    /// could otherwise reuse a predecessor's vector. Prepare smaller documents
    /// explicitly or choose a larger limit; no content is silently discarded.
    /// Failure splitting, when enabled, operates within these bounded groups.
    ///
    /// This is a request-input bound, NOT a tokenizer, model scratch, source
    /// residency, vector output, graph or process-memory ceiling. The builder
    /// still owns its complete source cohort. It is a per-build scheduling
    /// policy, like batch size, not a persisted producer identity or reopen
    /// setting. Set it explicitly on subsequent updates too.
    ///
    /// # Errors
    /// Rejects zero without filesystem or model work.
    pub fn with_max_batch_input_bytes(mut self, max_bytes: usize) -> SearchResult<Self> {
        if max_bytes == 0 {
            return Err(invalid(
                "builder.max_batch_input_bytes",
                "0",
                "the per-batch input byte limit must be positive",
            ));
        }
        self.max_batch_input_bytes = Some(max_bytes);
        Ok(self)
    }
}

pub(super) fn validate_document_size(bytes: usize, limit: Option<usize>) -> SearchResult<()> {
    if let Some(limit) = limit
        && (limit == 0 || bytes > limit)
    {
        return Err(invalid(
            "builder.max_batch_input_bytes",
            &bytes.to_string(),
            "a prepared document exceeds the input byte limit; explicitly prepare smaller content or increase the limit",
        ));
    }
    Ok(())
}

/// Return an exclusive source offset, using O(1) additional memory. The
/// unconfigured path keeps the old count-only partition with no length scan.
/// Subtraction from the remaining budget avoids an overflowing byte sum.
pub(super) fn batch_end(
    cx: &Cx,
    documents: &[IndexableDocument],
    start: usize,
    max_documents: usize,
    max_bytes: Option<usize>,
) -> SearchResult<usize> {
    checkpoint(cx, "native_ann.builder.input_batch")?;
    if max_documents == 0 || max_bytes == Some(0) {
        return Err(invalid(
            "builder.input_batch",
            "zero-limit",
            "input batches require positive configured limits",
        ));
    }
    let remaining = documents.get(start..).ok_or_else(|| {
        invalid(
            "builder.input_batch",
            "source-offset",
            "batch offset is outside the prepared source cohort",
        )
    })?;
    let Some(mut available) = max_bytes else {
        return Ok(start + max_documents.min(remaining.len()));
    };
    let mut end = start;
    for document in remaining.iter().take(max_documents) {
        checkpoint(cx, "native_ann.builder.input_batch")?;
        let bytes = document.content.len();
        validate_document_size(bytes, max_bytes)?;
        if bytes > available {
            break;
        }
        available -= bytes;
        end += 1;
    }
    checkpoint(cx, "native_ann.builder.input_batch_complete")?;
    Ok(end)
}

#[cfg(test)]
mod tests;
