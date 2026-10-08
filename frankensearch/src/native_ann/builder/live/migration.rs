//! Explicit whole-cohort model migration without changing serving handles.
//!
//! Ordinary live updates retain their original model contract. This separate
//! intent freezes replacement producers, rebuilds every vector and the lexical
//! cohort, and uses the SAME expected-snapshot installation fence. No admitted
//! vector is relabeled or reused across model identities.

use std::path::Path;
use std::sync::Arc;

use frankensearch_core::generation::{ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1};

use super::{NativeHybridCandidate, NativeHybridSnapshot};
use crate::native_ann::builder::{
    NativeBuildPrecision, NativeBuildRetrieval, NativeBuiltHybridIndex, NativeIndexBuilder,
};
use crate::native_ann::{checkpoint, invalid};
use crate::{Cx, Embedder, SearchResult};

/// An explicit replacement-model build against one exact live predecessor.
///
/// The complete retained source is the migration input, including titles and
/// metadata. All configured tiers are rebuilt; even an unchanged model receives
/// every document. No predecessor vector, graph, lexical writer or cache is used
/// as successor state. Adding or removing quality is explicit through the
/// optional quality provider passed to `begin_model_migration`.
///
/// Replacement models must accept the retained prepared-input contract. A new
/// canonicalization or document-ID policy requires freshly prepared sources
/// and a new index, not relabeling these documents. Model weights, tokenizer,
/// vector space, dimension and producer implementation may change. Changing
/// semantic/control kind is explicit and also requires complete re-embedding.
///
/// Defaults are the ordinary builder's F32/exact retrieval and batch settings,
/// not an inferred conversion of predecessor artifacts. Configure storage and
/// graph policies explicitly below. Source cloning and building have full-cohort
/// memory/I/O costs; the input-byte option is not a total memory bound.
///
/// This is process-local preparation, NOT durable selection or anti-rollback
/// authority. Seal the completed candidate through its existing API and keep
/// old mapped artifacts immutable while any readers retain them. Nothing here
/// deletes files, creates a runtime, downloads models or installs a candidate.
pub struct NativeLiveHybridMigration {
    base: NativeHybridSnapshot,
    builder: NativeIndexBuilder,
    target: TargetModels,
}

struct TargetModels {
    generation: ArtifactGenerationIdentityV1,
    fast: EmbeddingIdentityBundleV1,
    quality: Option<EmbeddingIdentityBundleV1>,
}

impl NativeHybridSnapshot {
    /// Prepare an explicit complete model migration from this retained source.
    ///
    /// No inference, source cloning or filesystem writes occur here. Supply
    /// both target providers deliberately: `None` requests a fast-only target;
    /// it never means to silently retain or drop quality after an error. The
    /// target generation must be strictly newer than this pin, including when
    /// the source is empty. Installation later compares the exact predecessor,
    /// so a concurrent source update cannot be lost behind a model upgrade.
    ///
    /// Each retained tier's complete prepared-input identity must be preserved.
    /// A newly added quality tier must share the fast tier's input contract.
    /// Models and dimensions need not match the old generation. A provider
    /// failure or drift is terminal and does not switch back to the old model.
    ///
    /// # Errors
    /// Refuses cancellation, non-newer/invalid generations, invalid target
    /// models, incompatible tier kinds or a changed prepared-input contract.
    pub fn begin_model_migration(
        &self,
        cx: &Cx,
        directory: impl AsRef<Path>,
        generation: ArtifactGenerationIdentityV1,
        fast: Arc<dyn Embedder>,
        quality: Option<Arc<dyn Embedder>>,
    ) -> SearchResult<NativeLiveHybridMigration> {
        checkpoint(cx, "native_ann.live.migration.prepare")?;
        generation.validate()?;
        if generation.sequence <= self.generation().sequence {
            return Err(invalid(
                "live.migration.generation",
                "not-newer",
                "model migration requires a strictly newer artifact generation",
            ));
        }
        let admission = NativeIndexBuilder::new(directory, generation, fast);
        checkpoint(cx, "native_ann.live.migration.fast_admitted")?;
        let mut builder = admission?;
        if let Some(quality) = quality {
            let admission = builder.with_quality_embedder(quality);
            checkpoint(cx, "native_ann.live.migration.quality_admitted")?;
            builder = admission?;
        }
        let before = self.index().vectors();
        let expected_quality_input = before
            .quality()
            .map_or(&before.fast().producer_identity.input, |tier| {
                &tier.producer_identity.input
            });
        if builder.fast.identity.input != before.fast().producer_identity.input
            || builder
                .quality
                .as_ref()
                .is_some_and(|tier| &tier.identity.input != expected_quality_input)
        {
            return Err(invalid(
                "live.migration.input",
                "changed",
                "retained prepared sources cannot be relabeled under another input contract",
            ));
        }
        let target = TargetModels {
            generation,
            fast: builder.fast.identity.clone(),
            quality: builder.quality.as_ref().map(|tier| tier.identity.clone()),
        };
        checkpoint(cx, "native_ann.live.migration.prepared")?;
        Ok(NativeLiveHybridMigration {
            base: self.clone(),
            builder,
            target,
        })
    }
}

impl NativeLiveHybridMigration {
    /// Bound the number of documents submitted in each inference group.
    ///
    /// # Errors
    /// Refuses zero before files or inference.
    pub fn with_batch_size(mut self, size: usize) -> SearchResult<Self> {
        self.builder = self.builder.with_batch_size(size)?;
        Ok(self)
    }

    /// Bound prepared input bytes per inference group in every target tier.
    ///
    /// All source documents are checked before candidate creation. This does
    /// not bound resident source bodies, vector images or model scratch memory.
    ///
    /// # Errors
    /// Refuses zero before files or inference.
    pub fn with_max_batch_input_bytes(mut self, max_bytes: usize) -> SearchResult<Self> {
        self.builder = self.builder.with_max_batch_input_bytes(max_bytes)?;
        Ok(self)
    }

    /// Opt in to the ordinary builder's bounded native-batch failure splitting.
    /// Bound-single providers retain their own admission path. Identity drift,
    /// malformed outputs and cancellation never trigger weaker inference.
    #[must_use]
    pub fn with_batch_failure_splitting(mut self) -> Self {
        self.builder = self.builder.with_batch_failure_splitting();
        self
    }

    /// Select target fast precision and fresh exact/HNSW retrieval.
    #[must_use]
    pub fn with_fast_storage(
        mut self,
        precision: NativeBuildPrecision,
        retrieval: NativeBuildRetrieval,
    ) -> Self {
        self.builder = self.builder.with_fast_storage(precision, retrieval);
        self
    }

    /// Select target quality precision and fresh exact/HNSW retrieval.
    ///
    /// # Errors
    /// Refuses a quality storage policy when the explicit target has no quality.
    pub fn with_quality_storage(
        mut self,
        precision: NativeBuildPrecision,
        retrieval: NativeBuildRetrieval,
    ) -> SearchResult<Self> {
        self.builder = self.builder.with_quality_storage(precision, retrieval)?;
        Ok(self)
    }

    /// Rebuild the complete cohort and return a stale-safe installable candidate.
    ///
    /// Every document is inferred by every requested model. Failure of a required
    /// tier or Quill aborts the whole migration, never returning an installable
    /// prefix. Dropping pending work drops the owned provider future; no task is
    /// detached. An unsuccessful candidate directory is retained, not deleted.
    ///
    /// The serving-selection lock is never acquired. Existing queries and source
    /// updates may finish while this build runs; a competing install makes the
    /// resulting candidate stale regardless of its higher generation number.
    /// Use `NativeLiveHybridIndex::install` for the single atomic swap after
    /// inspecting or sealing the candidate. It changes no durable selector.
    ///
    /// # Errors
    /// Returns source allocation, inference, identity, artifact, lexical or
    /// cancellation errors. The old snapshot and live selection remain intact.
    pub async fn build(mut self, cx: &Cx) -> SearchResult<NativeHybridCandidate> {
        checkpoint(cx, "native_ann.live.migration.source")?;
        let sources = self.base.index().vectors().documents();
        self.builder
            .documents
            .try_reserve_exact(sources.len())
            .map_err(|_| {
                invalid(
                    "live.migration.source",
                    "allocation",
                    "cannot retain the complete migration source cohort",
                )
            })?;
        for source in sources {
            checkpoint(cx, "native_ann.live.migration.source_document")?;
            self.builder.documents.push(source.clone());
        }
        checkpoint(cx, "native_ann.live.migration.build")?;
        let next = Box::pin(self.builder.build_hybrid(cx)).await?;
        self.target.admit(cx, self.base, next)
    }
}

impl TargetModels {
    fn admit(
        self,
        cx: &Cx,
        base: NativeHybridSnapshot,
        next: NativeBuiltHybridIndex,
    ) -> SearchResult<NativeHybridCandidate> {
        checkpoint(cx, "native_ann.live.migration.admit")?;
        let after = next.vectors();
        if after.fast().index().owner_witness().generation != self.generation
            || after.fast().producer_identity != self.fast
            || after.quality().map(|tier| &tier.producer_identity) != self.quality.as_ref()
            || after.quality().is_some_and(|tier| {
                tier.index().owner_witness().generation != self.generation
            })
        {
            return Err(invalid(
                "live.migration.target",
                "mismatch",
                "the complete candidate must match every explicitly requested model and tier",
            ));
        }
        let before = base.index().vectors().documents();
        if before.len() != after.documents().len() {
            return Err(source_changed());
        }
        for (old, new) in before.iter().zip(after.documents()) {
            checkpoint(cx, "native_ann.live.migration.source_join")?;
            if old.id != new.id
                || old.content != new.content
                || old.title != new.title
                || old.metadata != new.metadata
            {
                return Err(source_changed());
            }
        }
        checkpoint(cx, "native_ann.live.migration.ready")?;
        Ok(NativeHybridCandidate {
            base,
            next: NativeHybridSnapshot {
                index: Arc::new(next),
            },
        })
    }
}

fn source_changed() -> crate::SearchError {
    invalid(
        "live.migration.source",
        "changed",
        "model migration must preserve the exact retained source documents",
    )
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
mod tests;