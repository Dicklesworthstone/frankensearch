//! All-or-error source recovery over the original ordered partition receipts.
//! Ordinary source selection/decoding owns the file format and integrity checks.

use std::path::Path;

use frankensearch_core::generation::{ArtifactGenerationIdentityV1, GenerationComponentReceiptV1};

use super::{
    MAX_DESCRIPTOR_BYTES, Manifest, NativeBuiltShardedIndex, NativeShardedReopenLimits,
    SNAPSHOT_FILE, SNAPSHOT_SCHEMA, partition_name, rejected,
};
use crate::native_ann::builder::snapshot::{
    Artifact, SelectedSource, checked_directory, read_selected, select_source,
};
use crate::native_ann::builder::NativeReopenLimits;
use crate::native_ann::checkpoint;
use crate::{Cx, IndexableDocument, SearchResult};

impl NativeBuiltShardedIndex {
    /// Recover a complete selected source cohort despite damaged search artifacts.
    ///
    /// Authenticate the root and EVERY child descriptor, common generation,
    /// required-tier topology, counts and aggregate input budget before decoding
    /// any source. Then verify every full source stream through the ordinary
    /// native reader, including cross-partition ordering and unique membership.
    /// A missing last partition never becomes a successful recovered prefix.
    ///
    /// No original model, vector image, graph or lexical file is needed or read.
    /// The returned owned documents preserve prepared content, titles and
    /// metadata; they are rebuild input, NOT an admitted searchable index.
    /// Feed them to a fresh complete builder and independently retain its receipt.
    /// This neither repairs old artifacts nor selects or deletes a generation.
    ///
    /// `max_artifact_bytes` bounds root/child descriptors plus encoded sources
    /// for this operation; ignored derived artifacts consume no input budget.
    /// Partition source/record limits and the aggregate document/shard ceilings
    /// remain enforced. No all-corpus encoded buffer or vector slab is allocated.
    /// Decoded sources remain resident. Files must remain trusted and immutable;
    /// synchronous hashing/reads are cancellation-aware, not preemptible.
    ///
    /// # Errors
    /// Returns invalid selection, source membership/integrity, resource, I/O or
    /// cancellation errors without returning partial documents or modifying files.
    pub fn recover_selected_source(
        cx: &Cx,
        directory: impl AsRef<Path>,
        expected: &GenerationComponentReceiptV1,
        limits: NativeShardedReopenLimits,
    ) -> SearchResult<(ArtifactGenerationIdentityV1, Vec<IndexableDocument>)> {
        let selected = select_sources(cx, directory.as_ref(), expected, limits)?;
        let generation = selected.generation;
        Ok((generation, selected.read(cx)?))
    }
}

// Private source authority lets the hybrid wrapper join its own generation and
// count BEFORE decoding the first source, without reparsing an inner descriptor.
pub(super) struct SelectedSources {
    pub(super) generation: ArtifactGenerationIdentityV1,
    pub(super) documents: usize,
    partitions: Vec<SelectedSource>,
    limits: NativeReopenLimits,
}

impl SelectedSources {
    pub(super) fn read(self, cx: &Cx) -> SearchResult<Vec<IndexableDocument>> {
        let mut documents: Vec<IndexableDocument> = Vec::new();
        for source in self.partitions {
            checkpoint(cx, "native_ann.sharded_recovery.partition")?;
            let mut recovered = source.read(cx, self.limits)?;
            if let (Some(previous), Some(first)) = (documents.last(), recovered.first())
                && previous.id.as_str() >= first.id.as_str()
            {
                return Err(rejected(
                    "source_order",
                    "source partitions overlap or are reordered",
                ));
            }
            documents
                .try_reserve(recovered.len())
                .map_err(|_| rejected("allocation", "cannot retain complete recovered source"))?;
            documents.append(&mut recovered);
        }
        if documents.len() != self.documents {
            return Err(rejected("documents", "recovered source count changed"));
        }
        checkpoint(cx, "native_ann.sharded_recovery.complete")?;
        Ok(documents)
    }
}

pub(super) fn select_sources(
    cx: &Cx,
    directory: &Path,
    expected: &GenerationComponentReceiptV1,
    limits: NativeShardedReopenLimits,
) -> SearchResult<SelectedSources> {
    checkpoint(cx, "native_ann.sharded_recovery.start")?;
    limits.validate()?;
    let directory = checked_directory(directory)?;
    let bytes = read_selected(
        cx,
        &directory.join(SNAPSHOT_FILE),
        Artifact {
            byte_len: expected.byte_len,
            sha256: expected.sha256,
        },
        MAX_DESCRIPTOR_BYTES.min(limits.max_artifact_bytes),
    )?;
    let saved: Manifest = serde_json::from_slice(&bytes)
        .map_err(|_| rejected("schema", "malformed sharded snapshot descriptor"))?;
    if saved.schema != SNAPSHOT_SCHEMA
        || saved.partitions.is_empty()
        || saved.partitions.len() > limits.max_shards
        || saved.documents > limits.max_documents
    {
        return Err(rejected("source_inventory", "invalid selected source inventory"));
    }
    saved.generation.validate()?;
    let mut partitions = Vec::with_capacity(saved.partitions.len());
    let mut documents = 0_usize;
    let mut total = expected.byte_len;
    for (ordinal, receipt) in saved.partitions.iter().enumerate() {
        checkpoint(cx, "native_ann.sharded_recovery.preflight")?;
        if receipt.byte_len > limits.max_artifact_bytes.saturating_sub(total) {
            return Err(rejected("source_budget", "complete recovery input exceeds its limit"));
        }
        let source = select_source(
            cx,
            &directory.join(partition_name(ordinal)),
            *receipt,
            limits.partition,
        )?;
        if source.generation != saved.generation
            || source.quality != saved.quality
            || (source.documents == 0 && saved.partitions.len() != 1)
        {
            return Err(rejected(
                "source_inventory",
                "partition generation, topology or empty membership disagrees",
            ));
        }
        documents = documents
            .checked_add(source.documents)
            .ok_or_else(|| rejected("documents", "source count overflowed"))?;
        total = total
            .checked_add(receipt.byte_len)
            .and_then(|bytes| bytes.checked_add(source.byte_len()))
            .ok_or_else(|| rejected("source_budget", "recovery input bytes overflowed"))?;
        if documents > saved.documents || total > limits.max_artifact_bytes {
            return Err(rejected("source_budget", "complete recovery input exceeds its limit"));
        }
        partitions.push(source);
    }
    if documents != saved.documents {
        return Err(rejected("documents", "partition sources do not cover the selected count"));
    }
    // The small authenticated selections retain the exact original source
    // digests. No second descriptor read can rebind a shard between passes.
    Ok(SelectedSources {
        generation: saved.generation,
        documents,
        partitions,
        limits: limits.partition,
    })
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
mod tests;
