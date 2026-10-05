//! One source cohort partitioned into complete native fast/quality shard pairs.
//!
//! Construction, binding, inference, graph persistence and per-shard admission
//! reuse the ordinary builder. Retrieval reuses `NativeShardSet`; there is no
//! second merger or per-shard query embedding. One caller-retained receipt binds
//! the complete ordered partition inventory on restart. This is not a CURRENT
//! publisher, an external rollback floor, or permission to mutate sealed files.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use frankensearch_core::generation::{ArtifactGenerationIdentityV1, GenerationComponentReceiptV1};
use serde::{Deserialize, Serialize};

use super::snapshot::{
    Artifact, checked_directory, create_private_new, ensure_absent, read_selected,
    require_seal_platform, selected_footprint, sync_directory,
};
use super::{NativeBuildRetrieval, NativeBuiltIndex, NativeIndexBuilder, NativeReopenLimits};
use crate::native_ann::{NativeShardHit, NativeShardSet, checkpoint, invalid};
use crate::{Cx, Embedder, IndexableDocument, SearchError, SearchResult};

#[cfg(feature = "quill")]
mod hybrid;
#[cfg(feature = "quill")]
pub use hybrid::{NativeBuiltShardedHybridIndex, NativeShardedHybridReopenLimits};

const SNAPSHOT_FILE: &str = "native.sharded.json";
const SNAPSHOT_SCHEMA: &str = "frankensearch.native-sharded-source-vector.v1";
const MAX_DESCRIPTOR_BYTES: u64 = 1024 * 1024;
/// Hard bound on one native partition inventory, including its empty partition.
pub const MAX_NATIVE_BUILD_SHARDS: usize = 1024;

/// Aggregate admission limits as well as the existing per-partition limits.
/// These bound declared artifact bytes, not decoded objects, model scratch or RSS.
#[derive(Debug, Clone, Copy)]
pub struct NativeShardedReopenLimits {
    /// Maximum partitions; must lie in 1..=MAX_NATIVE_BUILD_SHARDS.
    pub max_shards: usize,
    /// Maximum source documents across ALL partitions.
    pub max_documents: usize,
    /// Aggregate descriptor, source, vector and graph byte allowance. Graph
    /// receipt files are conservatively charged their full per-file size bound.
    pub max_artifact_bytes: u64,
    /// Individual source-record, source-stream, vector and graph ceilings.
    pub partition: NativeReopenLimits,
}

impl Default for NativeShardedReopenLimits {
    fn default() -> Self {
        Self {
            max_shards: MAX_NATIVE_BUILD_SHARDS,
            max_documents: 10_000_000,
            max_artifact_bytes: 16 * 1024 * 1024 * 1024,
            partition: NativeReopenLimits::default(),
        }
    }
}

impl NativeShardedReopenLimits {
    fn validate(self) -> SearchResult<()> {
        self.partition.validate()?;
        if self.max_shards == 0 || self.max_shards > MAX_NATIVE_BUILD_SHARDS
            || self.max_artifact_bytes == 0
        {
            return Err(rejected("limits", "invalid aggregate shard admission limits"));
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    generation: ArtifactGenerationIdentityV1,
    documents: usize,
    quality: bool,
    // Names are derived from ordinals, never supplied by on-disk JSON.
    partitions: Vec<Artifact>,
}

/// Complete source/vector partitions from one build or exact selected restart.
///
/// All partitions retain the same generation and per-tier producers. Documents
/// are partitioned in ascending ID order; FSVI physical rows retain their own
/// order. Empty cohorts have exactly one identity-bearing empty partition.
/// The shared shard sets retain the original owners/graphs without copying them.
/// Input bodies and vector images remain resident across the entire inventory;
/// partition size is not an aggregate memory or query-latency guarantee.
/// No lexical reader is attested by this vector/source-only handle.
pub struct NativeBuiltShardedIndex {
    directory: PathBuf,
    partitions: Vec<NativeBuiltIndex>,
    fast: NativeShardSet,
    quality: Option<NativeShardSet>,
    documents: usize,
}

impl NativeIndexBuilder {
    /// Partition already-prepared documents and build every configured tier.
    ///
    /// Input order is canonicalized before partitioning. All source IDs, input
    /// budgets and provider/graph policies are checked before creating the root.
    /// Each child uses the ordinary native builder with the original generation,
    /// producers, precision, graph and batch policies. The last child failing
    /// aborts the whole operation; no successful prefix or fallback is returned.
    /// Sources move into children, rather than cloning all document bodies.
    ///
    /// Existing roots are refused. No files are removed on failure or drop.
    /// CPU/filesystem work remains on the caller's lane. No workers are spawned.
    ///
    /// # Errors
    /// Rejects zero partition size, more than 1024 partitions, invalid sources,
    /// producer drift, unsupported reuse, and ordinary build/admission failures.
    pub async fn build_sharded(
        mut self,
        cx: &Cx,
        max_documents_per_shard: usize,
    ) -> SearchResult<NativeBuiltShardedIndex> {
        checkpoint(cx, "native_ann.sharded_build.start")?;
        if max_documents_per_shard == 0 {
            return Err(rejected("partition_size", "partition size must be positive"));
        }
        self.validate_documents(cx)?;
        let count = self.documents.len().div_ceil(max_documents_per_shard).max(1);
        if count > MAX_NATIVE_BUILD_SHARDS || self.reuse.is_some() {
            return Err(rejected("partition_count", "shard limit exceeded or unpartitioned reuse supplied"));
        }
        for tier in std::iter::once(&self.fast).chain(self.quality.iter()) {
            tier.binding(&self.generation)?;
            if let NativeBuildRetrieval::Hnsw { params, .. } = tier.retrieval {
                params.validate()?;
            }
        }
        let directory = new_root(&self.directory)?;
        checkpoint(cx, "native_ann.sharded_build.create")?;
        fs::create_dir(&directory)?;
        let mut documents = self.documents.into_iter();
        let mut partitions = Vec::with_capacity(count);
        for ordinal in 0..count {
            checkpoint(cx, "native_ann.sharded_build.partition")?;
            let builder = Self {
                directory: directory.join(partition_name(ordinal)),
                generation: self.generation,
                fast: self.fast.clone(),
                quality: self.quality.clone(),
                batch_size: self.batch_size,
                max_batch_input_bytes: self.max_batch_input_bytes,
                split_failed_batches: self.split_failed_batches,
                documents: documents.by_ref().take(max_documents_per_shard).collect(),
                reuse: None,
            };
            partitions.push(Box::pin(builder.build(cx)).await?);
        }
        NativeBuiltShardedIndex::from_partitions(cx, directory, partitions)
    }
}

impl NativeBuiltShardedIndex {
    fn from_partitions(
        cx: &Cx,
        directory: PathBuf,
        partitions: Vec<NativeBuiltIndex>,
    ) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.sharded_build.admission")?;
        let Some(first) = partitions.first() else {
            return Err(rejected("inventory", "a complete cohort needs at least one partition"));
        };
        let quality_present = first.quality.is_some();
        let mut documents = 0_usize;
        let mut previous_id: Option<&str> = None;
        for (ordinal, partition) in partitions.iter().enumerate() {
            checkpoint(cx, "native_ann.sharded_build.membership")?;
            if partition.directory != directory.join(partition_name(ordinal))
                || partition.quality.is_some() != quality_present
                || (partition.documents.is_empty() && partitions.len() != 1)
            {
                return Err(rejected("inventory", "partition paths, membership or tier topology disagree"));
            }
            for document in partition.documents.iter() {
                checkpoint(cx, "native_ann.sharded_build.source_join")?;
                if previous_id.is_some_and(|id| id >= document.id.as_str()) {
                    return Err(rejected("source_order", "partition sources overlap or are out of order"));
                }
                previous_id = Some(&document.id);
            }
            documents = documents.checked_add(partition.documents.len())
                .ok_or_else(|| rejected("documents", "aggregate document count overflowed"))?;
        }
        // These witnesses come from completed native builds or children whose
        // original receipts were authenticated by the selected root descriptor.
        // No arbitrary directory is hashed to manufacture an expected witness.
        let assemble = |quality: bool| -> SearchResult<NativeShardSet> {
            let mut owners = Vec::with_capacity(partitions.len());
            for partition in &partitions {
                let tier = if quality {
                    partition.quality.as_ref().ok_or_else(|| rejected("quality", "missing required tier"))?
                } else {
                    &partition.fast
                };
                owners.push(Arc::clone(&tier.index));
            }
            let expected = owners.iter().map(|index| index.owner_witness().clone()).collect::<Vec<_>>();
            NativeShardSet::admit(cx, &expected, owners)
        };
        let fast = assemble(false)?;
        let quality = quality_present.then(|| assemble(true)).transpose()?;
        checkpoint(cx, "native_ann.sharded_build.complete")?;
        Ok(Self { directory, partitions, fast, quality, documents })
    }

    /// Canonical directory containing this complete ordered inventory.
    #[must_use]
    pub fn directory(&self) -> &Path { &self.directory }

    /// Read-only partitions in the exact order used by physical shard coordinates.
    #[must_use]
    pub fn partitions(&self) -> &[NativeBuiltIndex] { &self.partitions }

    /// Aggregate source count, not the row count of the first partition.
    #[must_use]
    pub const fn document_count(&self) -> usize { self.documents }

    /// Source records in strictly increasing ID order, with no body copies.
    pub fn documents(&self) -> impl Iterator<Item = &IndexableDocument> {
        self.partitions.iter().flat_map(|partition| partition.documents.iter())
    }

    /// Look up the exact retained source, independent of FSVI physical row order.
    #[must_use]
    pub fn document(&self, id: &str) -> Option<&IndexableDocument> {
        let ordinal = self.partitions.partition_point(|partition| {
            partition.documents.last().is_some_and(|last| last.id.as_str() < id)
        });
        self.partitions.get(ordinal)?.document(id)
    }

    /// Fast retrieval inventory with separately scoped physical shard rows.
    #[must_use]
    pub const fn fast(&self) -> &NativeShardSet { &self.fast }

    /// Independent quality inventory, absent only when explicitly unconfigured.
    #[must_use]
    pub const fn quality(&self) -> Option<&NativeShardSet> { self.quality.as_ref() }

    /// Complete generation identity shared by every partition and tier.
    #[must_use]
    pub fn generation(&self) -> ArtifactGenerationIdentityV1 {
        self.partitions[0].fast.index.owner_witness().generation
    }

    /// Embed once, then retrieve globally across the complete fast inventory.
    ///
    /// # Errors
    /// Propagates identity, inference, partition, budget and cancellation failures.
    pub async fn search_fast(&self, cx: &Cx, text: &str, k: usize) -> SearchResult<Vec<NativeShardHit>> {
        self.fast.search_text(cx, self.partitions[0].fast.embedder(), text, k, None).await
    }

    /// Retrieve the quality inventory without invoking the fast provider.
    ///
    /// # Errors
    /// Refuses absent quality and propagates the ordinary native query errors.
    pub async fn search_quality(&self, cx: &Cx, text: &str, k: usize) -> SearchResult<Vec<NativeShardHit>> {
        checkpoint(cx, "native_ann.sharded_build.quality")?;
        let tier = self.partitions[0].quality.as_ref()
            .ok_or_else(|| rejected("quality", "quality retrieval was not configured"))?;
        let shards = self.quality.as_ref()
            .ok_or_else(|| rejected("quality", "quality retrieval was not configured"))?;
        shards.search_text(cx, tier.embedder(), text, k, None).await
    }

    /// Seal every child and then the exact ordered inventory for selected restart.
    ///
    /// Retain the returned receipt independently. Neither this directory nor a
    /// self-consistent descriptor selects a generation. A partial seal remains
    /// unselected, is not removed, and cannot be overwritten by retrying in place.
    /// No inference, current-pointer update or live activation is performed.
    ///
    /// # Errors
    /// Rejects preexisting/changed artifacts, cancellation before root commit,
    /// unsupported sealing platforms, serialization, I/O and durability failures.
    pub fn seal_for_reopen(&self, cx: &Cx) -> SearchResult<GenerationComponentReceiptV1> {
        checkpoint(cx, "native_ann.sharded_snapshot.seal")?;
        require_seal_platform()?;
        let directory = checked_directory(&self.directory)?;
        let path = directory.join(SNAPSHOT_FILE);
        ensure_absent(&path)?;
        let mut partitions = Vec::with_capacity(self.partitions.len());
        for partition in &self.partitions {
            let receipt = partition.seal_for_reopen(cx)?;
            partitions.push(Artifact { byte_len: receipt.byte_len, sha256: receipt.sha256 });
        }
        let saved = Manifest {
            schema: SNAPSHOT_SCHEMA.to_owned(), generation: self.generation(),
            documents: self.documents, quality: self.quality.is_some(), partitions,
        };
        let bytes = serde_json::to_vec(&saved)
            .map_err(|_| rejected("encoding", "cannot encode the complete shard inventory"))?;
        if bytes.len() as u64 > MAX_DESCRIPTOR_BYTES {
            return Err(rejected("descriptor_size", "shard inventory exceeds its descriptor bound"));
        }
        checkpoint(cx, "native_ann.sharded_snapshot.seal_commit")?;
        let mut file = create_private_new(&path)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        sync_directory(&directory)?;
        if let Some(parent) = directory.parent() { sync_directory(parent)?; }
        Ok(Artifact::from_bytes(&bytes).receipt())
    }

    /// Reopen all selected source/vector/native-graph partitions with retained models.
    ///
    /// Authenticating every child descriptor and the aggregate input budget
    /// precedes the first vector/graph allocation. Every selected child is then
    /// opened through the strict ordinary native snapshot loader. Missing or
    /// corrupt partitions/graphs cannot yield partial results or exact fallback.
    /// Models are supplied by the caller, never discovered or downloaded.
    /// Paths remain a trusted immutable-directory protocol, not hostile-writer
    /// descriptor-relative admission or a durable publication/rollback authority.
    ///
    /// # Errors
    /// Returns selection, topology, size, identity, artifact or cancellation errors.
    pub fn open_selected(
        cx: &Cx,
        directory: impl AsRef<Path>,
        expected: &GenerationComponentReceiptV1,
        fast: Arc<dyn Embedder>,
        quality: Option<Arc<dyn Embedder>>,
        limits: NativeShardedReopenLimits,
    ) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.sharded_snapshot.open")?;
        limits.validate()?;
        let directory = checked_directory(directory.as_ref())?;
        let expected = Artifact { byte_len: expected.byte_len, sha256: expected.sha256 };
        let bytes = read_selected(cx, &directory.join(SNAPSHOT_FILE), expected, MAX_DESCRIPTOR_BYTES)?;
        let saved: Manifest = serde_json::from_slice(&bytes)
            .map_err(|_| rejected("schema", "malformed sharded snapshot descriptor"))?;
        if saved.schema != SNAPSHOT_SCHEMA || saved.partitions.is_empty()
            || saved.partitions.len() > limits.max_shards || saved.documents > limits.max_documents
            || saved.quality != quality.is_some()
        {
            return Err(rejected("inventory", "selected shard inventory, source count or topology is invalid"));
        }
        saved.generation.validate()?;
        let mut documents = 0_usize;
        let mut artifact_bytes = expected.byte_len;
        for (ordinal, receipt) in saved.partitions.iter().enumerate() {
            checkpoint(cx, "native_ann.sharded_snapshot.preflight")?;
            let (generation, count, has_quality, bytes) = selected_footprint(
                cx, &directory.join(partition_name(ordinal)), *receipt, limits.partition,
                Arc::clone(&fast), quality.as_ref().map(Arc::clone),
            )?;
            if generation != saved.generation || has_quality != saved.quality {
                return Err(rejected("generation", "every partition must share the selected generation and tiers"));
            }
            documents = documents.checked_add(count)
                .ok_or_else(|| rejected("documents", "aggregate source count overflowed"))?;
            artifact_bytes = artifact_bytes.checked_add(bytes)
                .ok_or_else(|| rejected("artifact_size", "aggregate artifact bytes overflowed"))?;
            if documents > saved.documents || artifact_bytes > limits.max_artifact_bytes {
                return Err(rejected("budget", "the complete selected inventory exceeds its aggregate bound"));
            }
        }
        if documents != saved.documents {
            return Err(rejected("documents", "selected partitions do not cover the declared source count"));
        }
        let mut partitions = Vec::with_capacity(saved.partitions.len());
        for (ordinal, receipt) in saved.partitions.iter().enumerate() {
            partitions.push(NativeBuiltIndex::open_selected_with_limits(
                cx, directory.join(partition_name(ordinal)), &receipt.receipt(),
                Arc::clone(&fast), quality.as_ref().map(Arc::clone), limits.partition,
            )?);
        }
        let opened = Self::from_partitions(cx, directory, partitions)?;
        if opened.generation() != saved.generation || opened.document_count() != saved.documents {
            return Err(rejected("membership", "admitted partitions differ from the selected cohort"));
        }
        Ok(opened)
    }
}

fn partition_name(ordinal: usize) -> String { format!("shard-{ordinal:06}") }

fn new_root(path: &Path) -> SearchResult<PathBuf> {
    let parent = path.parent().ok_or_else(|| rejected("directory", "missing root parent"))?;
    let name = path.file_name().ok_or_else(|| rejected("directory", "missing root name"))?;
    let parent = fs::canonicalize(parent)?;
    for ancestor in parent.ancestors() {
        for marker in [SNAPSHOT_FILE, "native.sharded-hybrid.json", "native.snapshot.json", "native.hybrid.json", "FSFS-BUNDLE.json"] {
            match fs::symlink_metadata(ancestor.join(marker)) {
                Ok(_) => return Err(rejected("directory", "cannot build inside a sealed generation")),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
    }
    let directory = parent.join(name);
    ensure_absent(&directory)?;
    Ok(directory)
}

fn rejected(field: &str, reason: &str) -> SearchError {
    invalid(&format!("sharded_builder.{field}"), "rejected", reason)
}

#[cfg(test)]
mod tests;
