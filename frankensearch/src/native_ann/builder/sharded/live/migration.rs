//! Whole-inventory model migration through the existing sharded live selector.
//!
//! A fresh builder re-embeds the complete retained source and builds one global
//! Quill population. No old vector, graph, cache or lexical writer is adopted.
//! Build/reopen admission is separate from the ordinary same-model update path;
//! both produce candidates for the same exact-predecessor installation fence.

use std::path::Path;
use std::sync::Arc;

use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1, GenerationComponentReceiptV1,
};

use super::{NativeShardedCandidate, NativeShardedSnapshot};
use crate::native_ann::builder::sharded::{
    MAX_NATIVE_BUILD_SHARDS, NativeBuiltShardedHybridIndex, NativeShardedHybridReopenLimits,
};
use crate::native_ann::builder::{
    NativeBuildPrecision, NativeBuildRetrieval, NativeBuiltTier, NativeIndexBuilder, TierPlan,
};
use crate::native_ann::{checkpoint, invalid};
use crate::{Cx, Embedder, SearchResult};

/// An explicit model replacement for every partition of one retained snapshot.
///
/// Models, dimensions, semantic/control kind and quality presence may change.
/// The target must preserve each retained tier's complete prepared-input
/// contract. Adding quality requires the retained fast input contract. Changing
/// canonicalization or document-ID semantics needs freshly prepared sources,
/// not relabeling this cohort. No original provider needs to run for migration.
///
/// Every target tier receives every source document, even when its model is
/// unchanged. Partition size may change independently of the predecessor. One
/// global Quill reader preserves whole-corpus lexical statistics and sources.
/// Defaults are F32/exact retrieval, not conversions of old artifacts. Configure
/// target storage and inference budgets explicitly. Building retains the full
/// decoded source and vector inventory; partition size is not an RSS bound.
///
/// This prepares a process-local candidate, not durable publication, retention
/// or an anti-rollback floor. The existing `install` compares the exact base pin
/// so a concurrent source update cannot be discarded by a higher sequence.
/// Old mapped files must remain immutable and present while readers live.
/// No models are downloaded, tasks detached or files deleted.
///
/// ```no_run
/// use std::{path::Path, sync::Arc};
/// use frankensearch::{Cx, Embedder, SearchResult};
/// use frankensearch::native_ann::builder::sharded::live::NativeLiveShardedHybridIndex;
/// use frankensearch_core::generation::ArtifactGenerationIdentityV1;
///
/// async fn upgrade(
///     live: &NativeLiveShardedHybridIndex,
///     cx: &Cx,
///     directory: &Path,
///     generation: ArtifactGenerationIdentityV1,
///     fast: Arc<dyn Embedder>,
///     quality: Option<Arc<dyn Embedder>>,
/// ) -> SearchResult<()> {
///     let base = live.snapshot(cx).await?;
///     let candidate = base
///         .begin_model_migration(cx, directory, generation, fast, quality)?
///         .with_batch_size(32)?
///         .build(cx, 5_000).await?;
///     // Separately seal/publish through the caller's durable authority as needed.
///     live.install(cx, &candidate).await?;
///     Ok(())
/// }
/// ```
pub struct NativeLiveShardedMigration {
    base: NativeShardedSnapshot,
    builder: NativeIndexBuilder,
}

impl NativeShardedSnapshot {
    /// Freeze replacement providers against this snapshot without starting work.
    ///
    /// `None` explicitly requests a fast-only target; it is not permission to
    /// discard failed quality inference. Sources are cloned only during build.
    /// No filesystem writes, inference or serving-selection lock is acquired.
    /// Ordinary `begin_update`/`prepare_selected` keep their same-model rules.
    ///
    /// # Errors
    /// Refuses cancellation, invalid/non-newer generations, invalid target
    /// producers, incompatible tier kinds or any changed prepared-input contract.
    pub fn begin_model_migration(
        &self,
        cx: &Cx,
        directory: impl AsRef<Path>,
        generation: ArtifactGenerationIdentityV1,
        fast: Arc<dyn Embedder>,
        quality: Option<Arc<dyn Embedder>>,
    ) -> SearchResult<NativeLiveShardedMigration> {
        checkpoint(cx, "native_ann.sharded_live.migration.prepare")?;
        generation.validate()?;
        if generation.sequence <= self.generation().sequence {
            return Err(invalid(
                "sharded_live.migration.generation",
                "not-newer",
                "model migration requires a strictly newer artifact generation",
            ));
        }
        let admission = NativeIndexBuilder::new(directory, generation, fast);
        checkpoint(cx, "native_ann.sharded_live.migration.fast_admitted")?;
        let mut builder = admission?;
        if let Some(quality) = quality {
            let admission = builder.with_quality_embedder(quality);
            checkpoint(cx, "native_ann.sharded_live.migration.quality_admitted")?;
            builder = admission?;
        }
        // Check captured source contracts across the entire inventory. Do not
        // ask the old providers for new metadata: a failed old model must not
        // prevent an explicit replacement using already-retained source bytes.
        for partition in self.index().vectors().partitions() {
            checkpoint(cx, "native_ann.sharded_live.migration.input")?;
            let fast_input = &partition.fast().producer_identity.input;
            let quality_input = partition
                .quality()
                .map_or(fast_input, |tier| &tier.producer_identity.input);
            if &builder.fast.identity.input != fast_input
                || builder
                    .quality
                    .as_ref()
                    .is_some_and(|tier| &tier.identity.input != quality_input)
            {
                return Err(invalid(
                    "sharded_live.migration.input",
                    "changed",
                    "every partition must preserve its retained prepared-input contract",
                ));
            }
        }
        checkpoint(cx, "native_ann.sharded_live.migration.prepared")?;
        Ok(NativeLiveShardedMigration {
            base: self.clone(),
            builder,
        })
    }
}

impl NativeLiveShardedMigration {
    /// Bound the document count of each target inference group.
    ///
    /// # Errors
    /// Refuses zero before inference or writes.
    pub fn with_batch_size(mut self, size: usize) -> SearchResult<Self> {
        self.builder = self.builder.with_batch_size(size)?;
        Ok(self)
    }

    /// Bound prepared content bytes per group, not aggregate memory or RSS.
    ///
    /// # Errors
    /// Refuses zero. Oversized sources fail build before any partition is created.
    pub fn with_max_batch_input_bytes(mut self, max_bytes: usize) -> SearchResult<Self> {
        self.builder = self.builder.with_max_batch_input_bytes(max_bytes)?;
        Ok(self)
    }

    /// Opt in to bounded recovery of ordinary native-batch inference failures.
    /// Bound-single overrides, identity refusals and cancellation remain intact.
    #[must_use]
    pub fn with_batch_failure_splitting(mut self) -> Self {
        self.builder = self.builder.with_batch_failure_splitting();
        self
    }

    /// Choose target fast precision and exact/HNSW retrieval for every partition.
    #[must_use]
    pub fn with_fast_storage(
        mut self,
        precision: NativeBuildPrecision,
        retrieval: NativeBuildRetrieval,
    ) -> Self {
        self.builder = self.builder.with_fast_storage(precision, retrieval);
        self
    }

    /// Choose target quality storage/retrieval for every partition.
    ///
    /// # Errors
    /// Refuses a quality policy when no target quality provider was supplied.
    pub fn with_quality_storage(
        mut self,
        precision: NativeBuildPrecision,
        retrieval: NativeBuildRetrieval,
    ) -> SearchResult<Self> {
        self.builder = self.builder.with_quality_storage(precision, retrieval)?;
        Ok(self)
    }

    /// Re-embed all retained documents into a new, complete sharded hybrid cohort.
    ///
    /// Every partition and configured tier must succeed before global lexical
    /// construction and candidate admission. There is no vector reuse, successful
    /// prefix, automatic fallback or install. Changing shard size never changes
    /// source membership; empty cohorts retain one identity-bearing partition.
    /// A dropped future releases its owned provider without detached work. The
    /// caller owns blocking/CPU placement; failed new directories are preserved.
    ///
    /// # Errors
    /// Returns partition/resource, source, model, artifact or cancellation errors.
    /// The live head and previously retained query snapshots are unchanged.
    pub async fn build(
        mut self,
        cx: &Cx,
        max_documents_per_shard: usize,
    ) -> SearchResult<NativeShardedCandidate> {
        checkpoint(cx, "native_ann.sharded_live.migration.source")?;
        let sources = self.base.index().vectors();
        if max_documents_per_shard == 0
            || sources
                .document_count()
                .div_ceil(max_documents_per_shard)
                .max(1)
                > MAX_NATIVE_BUILD_SHARDS
        {
            return Err(invalid(
                "sharded_live.migration.partitions",
                "invalid",
                "partition size must be positive and the inventory must fit the shard limit",
            ));
        }
        self.builder
            .documents
            .try_reserve_exact(sources.document_count())
            .map_err(|_| source_error("allocation", "cannot retain complete migration sources"))?;
        for document in sources.documents() {
            checkpoint(cx, "native_ann.sharded_live.migration.source_document")?;
            self.builder.documents.push(document.clone());
        }
        let target = TargetModels::capture(&self.builder);
        let outcome = Box::pin(
            self.builder
                .build_sharded_hybrid(cx, max_documents_per_shard),
        )
        .await;
        checkpoint(cx, "native_ann.sharded_live.migration.built")?;
        target.admit(cx, self.base, outcome?)
    }

    /// Reopen a sealed model replacement without rebuilding or inferring vectors.
    ///
    /// The directory supplied to `begin_model_migration` names the existing
    /// candidate. Supply its original trusted hybrid receipt, never a digest
    /// computed from an unknown directory. The ordinary selected reader admits
    /// every partition, required graph, source and global Quill artifact under
    /// its existing aggregate/per-file limits and exact source/lexical census.
    /// A missing final partition cannot become a successful prefix or fallback.
    ///
    /// The complete reopened cohort must match the explicitly requested full
    /// generation (including nonce), models, quality presence and per-tier
    /// precision/graph policy. Configure matching storage policies for a sealed
    /// F16/HNSW target. All retained source fields must equal the base snapshot,
    /// even when the selected receipt validly authenticates another source set.
    /// The receipt selects partition boundaries; they may differ from the base.
    /// Inference batch settings affect builds only, not these admission limits.
    ///
    /// This returns a candidate for the SAME exact-predecessor `install`, not
    /// permission to discard a source update committed while reopening. No
    /// model inference, source cloning, discovery, repair, file mutation or
    /// activation occurs. Callers still own durable selection, anti-rollback
    /// policy and immutable-file retention. Blocking work stays on their lane.
    ///
    /// # Errors
    /// Returns receipt, resource, identity, source/policy, artifact, I/O or
    /// cancellation errors. Every failure preserves the existing live selection.
    pub async fn open_selected(
        self,
        cx: &Cx,
        expected: &GenerationComponentReceiptV1,
        limits: NativeShardedHybridReopenLimits,
    ) -> SearchResult<NativeShardedCandidate> {
        checkpoint(cx, "native_ann.sharded_live.migration.reopen")?;
        let target = TargetModels::capture(&self.builder);
        let outcome = Box::pin(NativeBuiltShardedHybridIndex::open_selected(
            cx,
            &self.builder.directory,
            expected,
            Arc::clone(&self.builder.fast.embedder),
            self.builder
                .quality
                .as_ref()
                .map(|tier| Arc::clone(&tier.embedder)),
            limits,
        ))
        .await;
        checkpoint(cx, "native_ann.sharded_live.migration.reopened")?;
        target.admit(cx, self.base, outcome?)
    }
}

// Both fresh construction and selected restart must satisfy this same target
// contract. Only this admission can produce a model-changing live candidate.
struct TargetModels {
    generation: ArtifactGenerationIdentityV1,
    fast: TargetTier,
    quality: Option<TargetTier>,
}

struct TargetTier {
    identity: EmbeddingIdentityBundleV1,
    precision: NativeBuildPrecision,
    retrieval: NativeBuildRetrieval,
}

impl TargetTier {
    fn capture(plan: &TierPlan) -> Self {
        Self {
            identity: plan.identity.clone(),
            precision: plan.precision,
            retrieval: plan.retrieval,
        }
    }

    fn matches(&self, tier: &NativeBuiltTier) -> bool {
        if self.identity != tier.producer_identity || self.precision != tier.precision {
            return false;
        }
        match (self.retrieval, tier.graph_receipt.as_ref()) {
            (NativeBuildRetrieval::Exact, None) => true,
            (NativeBuildRetrieval::Hnsw { params, seed }, Some(receipt)) => {
                receipt.seed == seed
                    && usize::try_from(receipt.params.m).ok() == Some(params.m)
                    && usize::try_from(receipt.params.m0).ok() == Some(params.m0)
                    && usize::try_from(receipt.params.ef_construction).ok()
                        == Some(params.ef_construction)
                    && usize::try_from(receipt.params.ef_search).ok() == Some(params.ef_search)
            }
            _ => false,
        }
    }
}

impl TargetModels {
    fn capture(builder: &NativeIndexBuilder) -> Self {
        Self {
            generation: builder.generation,
            fast: TargetTier::capture(&builder.fast),
            quality: builder.quality.as_ref().map(TargetTier::capture),
        }
    }

    fn admit(
        self,
        cx: &Cx,
        base: NativeShardedSnapshot,
        next: NativeBuiltShardedHybridIndex,
    ) -> SearchResult<NativeShardedCandidate> {
        checkpoint(cx, "native_ann.sharded_live.migration.admit")?;
        let before = base.index().vectors();
        let after = next.vectors();
        for partition in after.partitions() {
            checkpoint(cx, "native_ann.sharded_live.migration.target")?;
            let quality_matches = match (&self.quality, partition.quality()) {
                (None, None) => true,
                (Some(target), Some(tier)) => {
                    target.matches(tier)
                        && tier.index().owner_witness().generation == self.generation
                }
                _ => false,
            };
            if partition.fast().index().owner_witness().generation != self.generation
                || !self.fast.matches(partition.fast())
                || !quality_matches
            {
                return Err(invalid(
                    "sharded_live.migration.target",
                    "mismatch",
                    "every partition must match the requested generation, models and storage policy",
                ));
            }
        }
        if before.document_count() != after.document_count() {
            return Err(source_error(
                "changed",
                "migration source count differs from the base",
            ));
        }
        // Partition ordinals may change. Compare complete original fields in
        // global ID order, not per-partition counts or source-body hashes alone.
        for (old, new) in before.documents().zip(after.documents()) {
            checkpoint(cx, "native_ann.sharded_live.migration.source_join")?;
            if old.id != new.id
                || old.content != new.content
                || old.title != new.title
                || old.metadata != new.metadata
            {
                return Err(source_error(
                    "changed",
                    "migration must preserve every source field",
                ));
            }
        }
        checkpoint(cx, "native_ann.sharded_live.migration.ready")?;
        Ok(NativeShardedCandidate {
            base,
            next: NativeShardedSnapshot {
                index: Arc::new(next),
            },
        })
    }
}

fn source_error(value: &str, reason: &str) -> crate::SearchError {
    invalid("sharded_live.migration.source", value, reason)
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
mod tests;
