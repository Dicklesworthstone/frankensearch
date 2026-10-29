//! Incremental inference into a new, complete native generation.
//!
//! Source edits never mutate the retained predecessor. Reuse is by exact source
//! ID, content, producer identity and storage precision, not by row number. New
//! FSVI rows and optional graphs always bind to the successor's membership.
//! Compatible native graphs retain their topology and insert only new documents.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use super::super::NativeAnnIndex;
use super::sharded::NativeBuiltShardedIndex;
use super::{
    NativeBuildPrecision, NativeBuildRetrieval, NativeBuiltIndex, NativeBuiltTier,
    NativeIndexBuilder, TierPlan, checkpoint, invalid,
};
use crate::{Cx, IndexableDocument, SearchResult};
use frankensearch_core::generation::ArtifactGenerationIdentityV1;
use frankensearch_index::native_hnsw::HnswParams;
use frankensearch_index::{ValidatedFsviBytes, VectorIndexWriter};

#[cfg(feature = "quill")]
use super::sharded::NativeBuiltShardedHybridIndex;
#[cfg(feature = "quill")]
use super::{NativeBuiltExactGeneration, NativeBuiltHybridIndex};
#[cfg(feature = "quill")]
use crate::native_ann::NativeGenerationSnapshot;
#[cfg(feature = "quill")]
use frankensearch_core::generation::{
    CommitRange, EmbedderTierTag, FrozenEmbeddingIdentityBundleV1,
};
#[cfg(feature = "quill")]
use frankensearch_index::generation_root::generation_reader::OpenedGenerationSnapshotV1;

/// Pending upserts/deletes against one immutable source/vector cohort.
///
/// The last edit for an ID wins; deleting an absent ID is a no-op. A successful
/// build contains every surviving document in every configured tier. Unchanged
/// content reuses its admitted vector; title/metadata edits still replace the
/// retained source document. Prepared `content` is the complete embedding input,
/// just as in [`NativeIndexBuilder`]. A rename is a delete plus an upsert and
/// deliberately re-embeds the new ID rather than assuming cross-ID equivalence.
///
/// Ordinary updates retain their producers and tier presence. An explicitly
/// authorized authority model migration may replace producers independently;
/// only tiers with an identical complete identity can reuse retained vectors.
/// Precision and graph policies are inherited but may be explicitly overridden.
/// A precision change re-embeds that entire tier: decoded F16 values are not
/// original F32 model outputs. Reuse does not skip producer admission, even for
/// empty builds.
///
/// This stages a successor, not a CURRENT switch or a durable rollback floor.
/// The predecessor remains searchable throughout a failed, cancelled or dropped
/// update. A caller must seal/select the completed successor through its existing
/// publication policy. Source and vector artifacts retain their full-cohort
/// memory and I/O costs. A compatible native graph is cloned and extended when
/// all predecessor rows survive unchanged; edits, deletions or graph-policy
/// changes rebuild it. Both paths persist a complete successor graph/receipt.
/// Construct with [`NativeBuiltIndex::begin_update`] or
/// [`NativeBuiltShardedIndex::begin_update`]. Either source layout may produce
/// an ordinary or partitioned successor. Repartitioning preserves inference
/// reuse by source identity, not the old partition or physical row ordinal.
pub struct NativeIndexUpdate {
    builder: NativeIndexBuilder,
    edits: BTreeMap<String, Option<IndexableDocument>>,
}

#[cfg(feature = "quill")]
#[derive(Clone, Copy)]
enum AuthorityUpdateIntent {
    RetainProducers,
    MigrateModels,
}

impl NativeBuiltIndex {
    /// Prepare a strictly newer generation while retaining the original sources.
    ///
    /// No files are created and no inference starts. The old owner is retained
    /// directly, not reopened by pathname. Physical row maps are built from each
    /// tier's exact document IDs, independently of source order and each other.
    /// Models retain their original identity and ANN parameters/seed; changing a
    /// model requires a fresh [`NativeIndexBuilder`], not silent vector reuse.
    ///
    /// # Errors
    /// Rejects non-newer generation sequences, cancelled work, changed providers,
    /// invalid parameters or inconsistent source/row membership.
    pub fn begin_update(
        &self,
        cx: &Cx,
        directory: impl AsRef<Path>,
        generation: ArtifactGenerationIdentityV1,
    ) -> SearchResult<NativeIndexUpdate> {
        NativeIndexUpdate::from_partitions(cx, directory, generation, std::slice::from_ref(self))
    }
}

impl NativeBuiltShardedIndex {
    /// Prepare source edits against every partition of this retained cohort.
    ///
    /// The returned update owns source and vector pins, not references to this
    /// handle or paths it must reopen. All partitions must have the same
    /// per-tier producer, precision and graph policy to inherit; mixed policies
    /// are refused rather than silently choosing one. No files or models start.
    /// Use `build_sharded` or `build_sharded_hybrid` to choose new partition sizes,
    /// or explicitly `build`/`build_hybrid` to coalesce into one ordinary cohort.
    ///
    /// # Errors
    /// Refuses non-newer generations, drift, mixed policies, inconsistent source
    /// membership or cancellation before staging any successor files.
    pub fn begin_update(
        &self,
        cx: &Cx,
        directory: impl AsRef<Path>,
        generation: ArtifactGenerationIdentityV1,
    ) -> SearchResult<NativeIndexUpdate> {
        NativeIndexUpdate::from_partitions(cx, directory, generation, self.partitions())
    }
}

impl NativeIndexUpdate {
    fn from_partitions(
        cx: &Cx,
        directory: impl AsRef<Path>,
        generation: ArtifactGenerationIdentityV1,
        partitions: &[NativeBuiltIndex],
    ) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.update.prepare")?;
        let first = partitions.first().ok_or_else(|| {
            invalid(
                "update.inventory",
                "empty",
                "a retained predecessor must have an owner",
            )
        })?;
        let predecessor = first.fast.index.owner_witness().generation;
        if generation.sequence <= predecessor.sequence {
            return Err(invalid(
                "update.generation",
                "not-newer",
                "an update requires a strictly higher generation sequence",
            ));
        }
        let mut builder =
            NativeIndexBuilder::new(directory, generation, Arc::clone(&first.fast.embedder))?;
        builder.fast = retained_plan(&first.fast)?;
        builder.quality = first.quality.as_ref().map(retained_plan).transpose()?;
        // Admit the new generation binding even when no changed documents will
        // need inference. An empty successor is not an identity bypass.
        builder.fast.binding(&generation)?;
        if let Some(quality) = &builder.quality {
            quality.binding(&generation)?;
        }
        let mut retained = Vec::with_capacity(partitions.len());
        let mut previous_id: Option<&str> = None;
        for partition in partitions {
            checkpoint(cx, "native_ann.update.partition")?;
            if partition.fast.index.owner_witness().generation != predecessor
                || partition.quality.is_some() != builder.quality.is_some()
                || (partition.documents.is_empty() && partitions.len() != 1)
            {
                return Err(invalid(
                    "update.inventory",
                    "mismatch",
                    "all predecessor partitions must share one generation and tier topology",
                ));
            }
            admit_retained_policy(&builder.fast, &partition.fast)?;
            if let (Some(plan), Some(tier)) = (&builder.quality, &partition.quality) {
                if tier.index.owner_witness().generation != predecessor {
                    return Err(invalid(
                        "update.inventory",
                        "generation",
                        "quality belongs to another generation",
                    ));
                }
                admit_retained_policy(plan, tier)?;
            }
            for document in partition.documents.iter() {
                checkpoint(cx, "native_ann.update.partition_sources")?;
                if document.id.is_empty()
                    || previous_id.is_some_and(|id| id >= document.id.as_str())
                {
                    return Err(invalid(
                        "update.source_join",
                        "order",
                        "retained sources must be nonempty IDs in strictly increasing order",
                    ));
                }
                previous_id = Some(&document.id);
            }
            retained.push(ReusePartition {
                documents: Arc::clone(&partition.documents),
                fast: ReuseTier::new(cx, &partition.fast, &partition.documents)?,
                quality: partition
                    .quality
                    .as_ref()
                    .map(|tier| ReuseTier::new(cx, tier, &partition.documents))
                    .transpose()?,
            });
        }
        builder.reuse = Some(Arc::new(ReuseSource {
            partitions: retained,
            #[cfg(feature = "quill")]
            authority: None,
        }));
        checkpoint(cx, "native_ann.update.prepared")?;
        Ok(Self {
            builder,
            edits: BTreeMap::new(),
        })
    }
}

impl NativeIndexUpdate {
    #[cfg(feature = "quill")]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::native_ann) fn from_authority(
        cx: &Cx,
        directory: impl AsRef<Path>,
        generation: ArtifactGenerationIdentityV1,
        provider: Arc<dyn crate::Embedder>,
        snapshot: &NativeGenerationSnapshot,
        max_source_bytes: usize,
        index: Arc<NativeAnnIndex>,
        quality: Option<(Arc<dyn crate::Embedder>, Arc<NativeAnnIndex>)>,
    ) -> SearchResult<Self> {
        Self::from_authority_with_intent(
            cx,
            directory,
            generation,
            provider,
            snapshot,
            max_source_bytes,
            index,
            quality,
            AuthorityUpdateIntent::RetainProducers,
        )
    }

    #[cfg(feature = "quill")]
    #[allow(clippy::too_many_arguments)]
    pub(in crate::native_ann) fn from_authority_model_migration(
        cx: &Cx,
        directory: impl AsRef<Path>,
        generation: ArtifactGenerationIdentityV1,
        provider: Arc<dyn crate::Embedder>,
        snapshot: &NativeGenerationSnapshot,
        max_source_bytes: usize,
        index: Arc<NativeAnnIndex>,
        quality: Option<(Arc<dyn crate::Embedder>, Arc<NativeAnnIndex>)>,
    ) -> SearchResult<Self> {
        Self::from_authority_with_intent(
            cx,
            directory,
            generation,
            provider,
            snapshot,
            max_source_bytes,
            index,
            quality,
            AuthorityUpdateIntent::MigrateModels,
        )
    }

    #[cfg(feature = "quill")]
    #[allow(clippy::too_many_arguments)]
    fn from_authority_with_intent(
        cx: &Cx,
        directory: impl AsRef<Path>,
        generation: ArtifactGenerationIdentityV1,
        provider: Arc<dyn crate::Embedder>,
        snapshot: &NativeGenerationSnapshot,
        max_source_bytes: usize,
        index: Arc<NativeAnnIndex>,
        quality: Option<(Arc<dyn crate::Embedder>, Arc<NativeAnnIndex>)>,
        intent: AuthorityUpdateIntent,
    ) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.update.authority")?;
        // An explicitly reopened predecessor carries the newer selection's
        // physical slot pair. That pair cannot authorize incremental edits to
        // its historical source. A previously current retained snapshot keeps
        // its own original pair and remains eligible for ordinary stale fencing.
        if snapshot
            .retained_snapshot()
            .selected_predecessor_of()
            .is_some()
        {
            return Err(invalid(
                "update.authority",
                "reopened-predecessor",
                "incremental updates require a snapshot with its originally selected authority pair",
            ));
        }
        let predecessor = index.owner_witness().generation;
        if generation.sequence <= predecessor.sequence {
            return Err(invalid(
                "update.generation",
                "not-newer",
                "an update requires a strictly higher generation sequence",
            ));
        }
        if quality.is_some()
            != (snapshot.topology() == crate::native_ann::NativeGenerationTopology::FullProgressive)
        {
            return Err(invalid(
                "update.inventory",
                "topology",
                "retained update providers must match the complete selected topology",
            ));
        }
        let precision = match index.owner_witness().quantization {
            frankensearch_index::Quantization::F32 => NativeBuildPrecision::F32,
            frankensearch_index::Quantization::F16 => NativeBuildPrecision::F16,
        };
        let mut builder = NativeIndexBuilder::new(directory, generation, provider)?;
        builder.fast.precision = precision;
        builder.fast.retrieval = snapshot.retrieval_configuration();
        builder.build_pool(cx)?;
        let tier = snapshot.tier();
        let key = match tier {
            EmbedderTierTag::Fast => "fast",
            EmbedderTierTag::Quality => "quality",
        };
        // The plan retains the actual provider output identity. Normalize only
        // its persistence encoding, exactly as the original writer did, before
        // comparing the COMPLETE authenticated identity, including storage.
        let fast_fingerprint = authority_reuse_fingerprint(
            &builder.fast,
            snapshot.metadata().embedders.get(key),
            predecessor,
            &index,
            intent,
        )?;
        let quality = if let Some((provider, index)) = quality {
            if index.owner_witness().generation != predecessor {
                return Err(invalid(
                    "update.inventory",
                    "generation",
                    "quality belongs to another generation",
                ));
            }
            let precision = match index.owner_witness().quantization {
                frankensearch_index::Quantization::F32 => NativeBuildPrecision::F32,
                frankensearch_index::Quantization::F16 => NativeBuildPrecision::F16,
            };
            builder = builder
                .with_quality_embedder(provider)?
                .with_quality_storage(
                    precision,
                    snapshot.quality_retrieval_configuration().ok_or_else(|| {
                        invalid(
                            "update.inventory",
                            "quality-policy",
                            "the complete generation must retain its independent quality retrieval policy",
                        )
                    })?,
                )?;
            let plan = builder.quality.as_ref().ok_or_else(|| {
                invalid(
                    "update.producer",
                    "missing",
                    "an independently supplied quality provider is required",
                )
            })?;
            let fingerprint = authority_reuse_fingerprint(
                plan,
                snapshot.metadata().embedders.get("quality"),
                predecessor,
                &index,
                intent,
            )?;
            Some((index, fingerprint, precision))
        } else {
            None
        };
        let mut documents = snapshot.source_documents(cx, max_source_bytes)?;
        documents.sort_by(|left, right| left.id.cmp(&right.id));
        let fast = ReuseTier::from_owner(cx, index, &documents, fast_fingerprint, precision)?;
        let quality = quality
            .map(|(index, fingerprint, precision)| {
                ReuseTier::from_owner(cx, index, &documents, fingerprint, precision)
            })
            .transpose()?;
        let progressive = quality.is_some();
        builder.reuse = Some(Arc::new(ReuseSource {
            partitions: vec![ReusePartition {
                documents: documents.into(),
                fast,
                quality,
            }],
            authority: Some(AuthorityReuse {
                snapshot: Arc::clone(snapshot.retained_snapshot()),
                tier,
                progressive,
                source: snapshot.metadata().commit_range.clone(),
            }),
        }));
        checkpoint(cx, "native_ann.update.authority_prepared")?;
        Ok(Self {
            builder,
            edits: BTreeMap::new(),
        })
    }

    /// Finalize successor vectors and graphs on this caller-owned pool.
    ///
    /// Overrides the pool inherited through the build Cx. The predecessor is
    /// never changed by worker completion; installation remains a caller action.
    #[must_use]
    pub fn with_blocking_pool(
        mut self,
        pool: asupersync::runtime::blocking_pool::BlockingPoolHandle,
    ) -> Self {
        self.builder = self.builder.with_blocking_pool(pool);
        self
    }

    /// Insert a new document or replace the complete source document for this ID.
    /// Later edits for the same ID replace this edit. Empty IDs fail before build.
    #[must_use]
    pub fn upsert_document(mut self, document: IndexableDocument) -> Self {
        self.edits.insert(document.id.clone(), Some(document));
        self
    }

    /// Apply upserts in iteration order, with the same last-edit-wins semantics.
    #[must_use]
    pub fn upsert_documents(self, documents: impl IntoIterator<Item = IndexableDocument>) -> Self {
        documents.into_iter().fold(self, Self::upsert_document)
    }

    /// Remove this ID from all successor tiers and retained source documents.
    /// Old readers keep their old membership; an absent ID is a no-op.
    #[must_use]
    pub fn delete_document(mut self, id: impl Into<String>) -> Self {
        self.edits.insert(id.into(), None);
        self
    }

    /// Bound the number of documents submitted to a provider in a batch.
    ///
    /// # Errors
    /// Refuses zero, as [`NativeIndexBuilder::with_batch_size`] does.
    pub fn with_batch_size(mut self, batch_size: usize) -> SearchResult<Self> {
        self.builder = self.builder.with_batch_size(batch_size)?;
        Ok(self)
    }

    /// Bound prepared content bytes in each successor inference batch.
    ///
    /// This is an explicit per-update setting, not inherited from a completed
    /// generation. The final source cohort is checked before candidate creation,
    /// including unchanged survivors; deleted documents are not charged. Actual
    /// model requests contain only changed inputs eligible for that tier.
    /// See [`NativeIndexBuilder::with_max_batch_input_bytes`] for exact scope.
    ///
    /// # Errors
    /// Refuses zero before building the successor.
    pub fn with_max_batch_input_bytes(mut self, max_bytes: usize) -> SearchResult<Self> {
        self.builder = self.builder.with_max_batch_input_bytes(max_bytes)?;
        Ok(self)
    }

    /// Override the successor's fast storage or graph policy, not its model.
    /// Changing precision re-embeds all surviving fast documents.
    #[must_use]
    pub fn with_fast_storage(
        mut self,
        precision: NativeBuildPrecision,
        retrieval: NativeBuildRetrieval,
    ) -> Self {
        self.builder = self.builder.with_fast_storage(precision, retrieval);
        self
    }

    /// Override a configured quality tier's storage or graph policy.
    /// Changing precision re-embeds all surviving quality documents.
    ///
    /// # Errors
    /// Refuses an absent quality tier; updates cannot silently add or remove one.
    pub fn with_quality_storage(
        mut self,
        precision: NativeBuildPrecision,
        retrieval: NativeBuildRetrieval,
    ) -> SearchResult<Self> {
        self.builder = self.builder.with_quality_storage(precision, retrieval)?;
        Ok(self)
    }

    /// Build both required tiers into a new directory without touching the old one.
    ///
    /// # Errors
    /// Propagates edit, identity, inference, storage, graph and cancellation errors.
    /// A failed attempt never returns a partial successor or overwrites its leaf.
    pub async fn build(self, cx: &Cx) -> SearchResult<NativeBuiltIndex> {
        Box::pin(self.prepare(cx)?.build(cx)).await
    }

    /// Build complete authority-publishable components from these source edits.
    ///
    /// Unchanged content reuses its retained vector. Every successor vector,
    /// lexical, metadata and optional graph image is rebuilt and admitted as one
    /// complete cohort. This creates no authority selection and never changes
    /// the predecessor. Use the ordinary publisher with its original expected
    /// authority pair; a concurrent publication may still win that comparison.
    ///
    /// Updates begun from an authority snapshot preserve its tier label and may
    /// not move either source-checkpoint boundary backward. As with
    /// [`NativeIndexBuilder::build_exact_generation`], this route requires one
    /// provider. Deleting the final document produces complete empty components
    /// that a later update can repopulate with the same provider. Finalization
    /// inherits the configured/caller-context pool through the ordinary builder.
    ///
    /// # Errors
    /// Includes edit/identity/inference/build errors and refuses an authority
    /// tier change, backward source range or invalid image budget.
    #[cfg(feature = "quill")]
    pub async fn build_exact_generation(
        self,
        cx: &Cx,
        tier: EmbedderTierTag,
        source: CommitRange,
        max_image_bytes: usize,
    ) -> SearchResult<NativeBuiltExactGeneration> {
        checkpoint(cx, "native_ann.update.generation")?;
        if let Some(authority) = self
            .builder
            .reuse
            .as_ref()
            .and_then(|reuse| reuse.authority.as_ref())
            && (tier != authority.tier
                || authority.progressive
                || source.low < authority.source.low
                || source.high < authority.source.high)
        {
            return Err(invalid(
                "update.authority_contract",
                "changed",
                "authority updates must preserve their tier and advance source boundaries monotonically",
            ));
        }
        // The vector worker owns a reuse pin, but drops it before the composite
        // builder exports and admits Quill. Keep this independent pin through
        // that complete operation as well.
        let authority_pin = self
            .builder
            .reuse
            .as_ref()
            .and_then(|reuse| reuse.authority.as_ref())
            .map(|authority| Arc::clone(&authority.snapshot));
        let result =
            Box::pin(
                self.prepare(cx)?
                    .build_exact_generation(cx, tier, source, max_image_bytes),
            )
            .await;
        drop(authority_pin);
        result
    }

    /// Build a complete fast-and-quality successor using per-tier retained reuse.
    ///
    /// Both original providers and independent storage/retrieval plans remain explicit.
    /// Changed content is embedded independently in each tier, while unchanged
    /// vectors are copied only from their matching admitted owner. The original
    /// authority pin survives until all new components pass joint admission.
    /// Empty successors preserve both identities and require no document inference.
    /// This constructs a candidate and never publishes or modifies a predecessor.
    ///
    /// # Errors
    /// Refuses a topology change, backward source boundaries,
    /// invalid ANN plans, identity drift, budget exhaustion or cancellation.
    #[cfg(feature = "quill")]
    pub async fn build_progressive_generation(
        self,
        cx: &Cx,
        source: CommitRange,
        max_image_bytes: usize,
    ) -> SearchResult<NativeBuiltExactGeneration> {
        checkpoint(cx, "native_ann.update.progressive_generation")?;
        if let Some(authority) = self
            .builder
            .reuse
            .as_ref()
            .and_then(|reuse| reuse.authority.as_ref())
            && (!authority.progressive
                || source.low < authority.source.low
                || source.high < authority.source.high)
        {
            return Err(invalid(
                "update.authority_contract",
                "changed",
                "authority updates must preserve both tiers and advance source boundaries monotonically",
            ));
        }
        let authority_pin = self
            .builder
            .reuse
            .as_ref()
            .and_then(|reuse| reuse.authority.as_ref())
            .map(|authority| Arc::clone(&authority.snapshot));
        let result = Box::pin(self.prepare(cx)?.build_progressive_generation(
            cx,
            source,
            max_image_bytes,
        ))
        .await;
        drop(authority_pin);
        result
    }

    /// Repartition the edited source cohort without re-embedding unchanged rows.
    ///
    /// The ordinary builder recreates all vector images. Vector reuse looks
    /// up the exact ID/content/producer/precision in the shared predecessor
    /// inventory, independent of both old and new shard boundaries. A graph can
    /// be extended only when a predecessor partition survives completely within
    /// one successor partition; other partition changes rebuild that graph. All child
    /// builds must succeed; no admitted prefix or live selection is returned.
    /// Empty output retains one identity-bearing partition. Choose a new root.
    ///
    /// # Errors
    /// Includes the ordinary update and `build_sharded` input, shard-count,
    /// inference, artifact, cancellation and destination errors.
    pub async fn build_sharded(
        self,
        cx: &Cx,
        max_documents_per_shard: usize,
    ) -> SearchResult<NativeBuiltShardedIndex> {
        Box::pin(self.prepare(cx)?.build_sharded(cx, max_documents_per_shard)).await
    }

    /// Build a complete successor source/vector/Quill cohort from these edits.
    ///
    /// Vector inference is incremental, but Quill is rebuilt from the complete
    /// final source cohort in its own new directory. It cannot retain deleted
    /// documents or old text/metadata by adopting the predecessor's publication.
    /// The returned reader is read-only and the existing hybrid seal/reopen APIs
    /// work unchanged. No old files, writer locks or CURRENT pointer are changed.
    ///
    /// # Errors
    /// Includes every required vector and lexical build/admission error. A failed
    /// lexical stage never returns the already-built vectors as hybrid success.
    #[cfg(feature = "quill")]
    pub async fn build_hybrid(self, cx: &Cx) -> SearchResult<NativeBuiltHybridIndex> {
        Box::pin(self.prepare(cx)?.build_hybrid(cx)).await
    }

    /// Rebuild edited vector partitions and one global Quill/source cohort.
    ///
    /// Uses the existing complete sharded hybrid builder and its exact lexical
    /// census. Deleted IDs and old text cannot survive in one component. Old
    /// readers keep their old cohort; the caller separately seals/selects this
    /// complete successor. Only inference is incremental, not artifact writes.
    ///
    /// # Errors
    /// Includes all required vector, graph, lexical and source admission errors.
    #[cfg(feature = "quill")]
    pub async fn build_sharded_hybrid(
        self,
        cx: &Cx,
        max_documents_per_shard: usize,
    ) -> SearchResult<NativeBuiltShardedHybridIndex> {
        Box::pin(
            self.prepare(cx)?
                .build_sharded_hybrid(cx, max_documents_per_shard),
        )
        .await
    }

    fn prepare(mut self, cx: &Cx) -> SearchResult<NativeIndexBuilder> {
        checkpoint(cx, "native_ann.update.edits")?;
        self.builder.build_pool(cx)?;
        if self.edits.contains_key("") {
            return Err(invalid(
                "update.document_id",
                "empty",
                "upserts and deletes require nonempty document IDs",
            ));
        }
        let source = self.builder.reuse.as_ref().ok_or_else(|| {
            invalid(
                "update.source",
                "missing",
                "updates require a retained predecessor",
            )
        })?;
        let mut documents = Vec::new();
        for previous in source.documents() {
            checkpoint(cx, "native_ann.update.merge_source")?;
            match self.edits.remove(previous.id.as_str()) {
                Some(Some(document)) => documents.push(document),
                Some(None) => {}
                None => documents.push(previous.clone()),
            }
        }
        for (_, edit) in self.edits {
            checkpoint(cx, "native_ann.update.merge_insert")?;
            if let Some(document) = edit {
                documents.push(document);
            }
        }
        self.builder.documents = documents;
        Ok(self.builder)
    }
}

#[cfg(feature = "quill")]
impl NativeBuiltHybridIndex {
    /// Stage source edits while this complete hybrid generation remains readable.
    ///
    /// Finish with [`NativeIndexUpdate::build_hybrid`] to rebuild lexical and both
    /// required vector tiers together. `build` is explicitly vector/source-only.
    /// This does not activate the successor or replace an existing reader.
    ///
    /// # Errors
    /// Has the same admission errors as [`NativeBuiltIndex::begin_update`].
    pub fn begin_update(
        &self,
        cx: &Cx,
        directory: impl AsRef<Path>,
        generation: ArtifactGenerationIdentityV1,
    ) -> SearchResult<NativeIndexUpdate> {
        self.vectors().begin_update(cx, directory, generation)
    }
}

#[cfg(feature = "quill")]
impl NativeBuiltShardedHybridIndex {
    /// Stage edits against the exact source/vector cohort behind this Quill view.
    ///
    /// Finish with `build_sharded_hybrid` to reconstruct the complete global
    /// lexical population with the edited vector partitions. Existing progressive
    /// queries continue borrowing their original complete object.
    ///
    /// # Errors
    /// Has the same admission errors as `NativeBuiltShardedIndex::begin_update`.
    pub fn begin_update(
        &self,
        cx: &Cx,
        directory: impl AsRef<Path>,
        generation: ArtifactGenerationIdentityV1,
    ) -> SearchResult<NativeIndexUpdate> {
        self.vectors().begin_update(cx, directory, generation)
    }
}

pub(super) struct ReuseSource {
    // Shared by ALL successor children: no repeated source-body, vector-slab,
    // or physical-row-map copies when repartitioning the predecessor.
    partitions: Vec<ReusePartition>,
    #[cfg(feature = "quill")]
    authority: Option<AuthorityReuse>,
}

#[cfg(feature = "quill")]
struct AuthorityReuse {
    // Keep the original root/closure/reader pin even when only this prepared
    // update remains alive, including while a blocking finalizer owns it.
    snapshot: Arc<OpenedGenerationSnapshotV1>,
    tier: EmbedderTierTag,
    progressive: bool,
    source: CommitRange,
}

struct ReusePartition {
    documents: Arc<[IndexableDocument]>,
    fast: ReuseTier,
    quality: Option<ReuseTier>,
}

#[derive(Clone, Copy)]
pub(super) enum ReuseRole {
    Fast,
    Quality,
}

impl ReuseSource {
    /// Retain exact lexical bytes only for an identical complete source and cut.
    /// Provider migration may change vectors while these lexical inputs stay fixed.
    #[cfg(feature = "quill")]
    pub(super) fn unchanged_authority_lexical(
        &self,
        cx: &Cx,
        documents: &[IndexableDocument],
        source: &CommitRange,
    ) -> SearchResult<Option<Arc<OpenedGenerationSnapshotV1>>> {
        let Some(authority) = &self.authority else {
            return Ok(None);
        };
        if &authority.source != source || self.documents().count() != documents.len() {
            return Ok(None);
        }
        for document in documents {
            checkpoint(cx, "native_ann.update.lexical_reuse_source")?;
            let Some((partition, position)) = self.find(&document.id) else {
                return Ok(None);
            };
            let previous = &partition.documents[position];
            if document.content != previous.content
                || document.title != previous.title
                || document.metadata != previous.metadata
            {
                return Ok(None);
            }
        }
        Ok(Some(Arc::clone(&authority.snapshot)))
    }

    fn documents(&self) -> impl Iterator<Item = &IndexableDocument> {
        self.partitions
            .iter()
            .flat_map(|partition| partition.documents.iter())
    }

    fn find(&self, id: &str) -> Option<(&ReusePartition, usize)> {
        let ordinal = self.partitions.partition_point(|partition| {
            partition
                .documents
                .last()
                .is_some_and(|last| last.id.as_str() < id)
        });
        let partition = self.partitions.get(ordinal)?;
        let position = partition
            .documents
            .binary_search_by(|document| document.id.as_str().cmp(id))
            .ok()?;
        Some((partition, position))
    }

    /// Try one retained partition, selected by the first surviving source ID.
    /// The native boundary proves exact membership/vector compatibility before
    /// cloning anything. Trying every old partition could make repartitioning
    /// quadratic; an ineligible candidate deliberately selects a fresh build.
    pub(super) fn try_reuse_graph(
        &self,
        cx: &Cx,
        role: ReuseRole,
        owner: Arc<ValidatedFsviBytes>,
        documents: &[IndexableDocument],
        params: HnswParams,
        seed: u64,
    ) -> SearchResult<Option<NativeAnnIndex>> {
        for document in documents {
            checkpoint(cx, "native_ann.update.graph_candidate")?;
            let Some((partition, _)) = self.find(&document.id) else {
                continue;
            };
            let tier = match role {
                ReuseRole::Fast => &partition.fast,
                ReuseRole::Quality => partition.quality.as_ref().ok_or_else(|| {
                    invalid(
                        "update.graph_candidate",
                        "quality",
                        "missing retained quality owner",
                    )
                })?,
            };
            if tier.producer_fingerprint.is_none() {
                return Ok(None);
            }
            return tier.index.try_reuse_for_owner(cx, owner, params, seed);
        }
        Ok(None)
    }
}

struct ReuseTier {
    index: Arc<NativeAnnIndex>,
    owner: Arc<ValidatedFsviBytes>,
    physical_rows: Vec<usize>,
    // None explicitly forbids vector AND graph reuse for a migrated producer.
    // Source equality or coincidentally equal output values cannot admit it.
    producer_fingerprint: Option<String>,
    precision: NativeBuildPrecision,
}

impl ReuseTier {
    fn new(cx: &Cx, tier: &NativeBuiltTier, documents: &[IndexableDocument]) -> SearchResult<Self> {
        Self::from_owner(
            cx,
            Arc::clone(&tier.index),
            documents,
            Some(tier.producer_identity.fingerprint()),
            tier.precision,
        )
    }

    fn from_owner(
        cx: &Cx,
        index: Arc<NativeAnnIndex>,
        documents: &[IndexableDocument],
        producer_fingerprint: Option<String>,
        precision: NativeBuildPrecision,
    ) -> SearchResult<Self> {
        let owner = Arc::clone(&index.owner);
        if owner.live_count() != documents.len() {
            return Err(invalid(
                "update.source_join",
                "cardinality",
                "source/tier membership differs",
            ));
        }
        for (position, document) in documents.iter().enumerate() {
            checkpoint(cx, "native_ann.update.source_order")?;
            if document.id.is_empty() || (position > 0 && documents[position - 1].id >= document.id)
            {
                return Err(invalid(
                    "update.source_join",
                    "order",
                    "retained sources must have unique nonempty IDs in increasing order",
                ));
            }
        }
        let mut physical_rows = vec![usize::MAX; documents.len()];
        for physical in 0..owner.record_count() {
            checkpoint(cx, "native_ann.update.row_map")?;
            let row = owner.row(physical)?;
            if !row.flags().is_live() {
                continue;
            }
            let source = documents
                .binary_search_by(|document| document.id.as_str().cmp(row.doc_id()))
                .map_err(|_| {
                    invalid(
                        "update.source_join",
                        "unknown-id",
                        "row has no source document",
                    )
                })?;
            if std::mem::replace(&mut physical_rows[source], physical) != usize::MAX {
                return Err(invalid(
                    "update.source_join",
                    "duplicate",
                    "source rows must be live and unique",
                ));
            }
        }
        if physical_rows.contains(&usize::MAX) {
            return Err(invalid(
                "update.source_join",
                "missing",
                "every retained source must resolve to one live physical row",
            ));
        }
        Ok(Self {
            index,
            owner,
            physical_rows,
            producer_fingerprint,
            precision,
        })
    }
}

impl TierPlan {
    pub(super) async fn write_batch_reusing(
        &self,
        cx: &Cx,
        writer: &mut VectorIndexWriter,
        documents: &[IndexableDocument],
        reuse: Option<(&ReuseSource, ReuseRole)>,
    ) -> SearchResult<()> {
        let Some((source, role)) = reuse else {
            return self.write_batch(cx, writer, documents).await;
        };
        checkpoint(cx, "native_ann.update.before_batch")?;
        self.admit(self.embedder.identity()?)?;
        let producer = self.identity.fingerprint();
        let mut changed = Vec::new();
        for document in documents {
            checkpoint(cx, "native_ann.update.reuse_row")?;
            if let Some((partition, position)) = source.find(&document.id) {
                let tier = match role {
                    ReuseRole::Fast => &partition.fast,
                    ReuseRole::Quality => partition.quality.as_ref().ok_or_else(|| {
                        invalid(
                            "update.source_join",
                            "quality",
                            "missing retained quality owner",
                        )
                    })?,
                };
                let Some(retained_producer) = &tier.producer_fingerprint else {
                    changed.push(document.clone());
                    continue;
                };
                if &producer != retained_producer {
                    return Err(invalid(
                        "update.producer",
                        "changed",
                        "reused vectors require the original complete producer identity",
                    ));
                }
                if self.precision == tier.precision
                    && partition.documents[position].content == document.content
                {
                    let physical = tier.physical_rows[position];
                    let vector = tier.owner.vector_at_f32(physical)?;
                    // F16 -> F32 -> the SAME F16 storage is exact. Changed
                    // precision re-embeds instead of promoting rounded values.
                    writer.write_record(document.id.as_str(), &vector)?;
                    continue;
                }
            }
            changed.push(document.clone());
        }
        if !changed.is_empty() {
            self.write_batch(cx, writer, &changed).await?;
        }
        checkpoint(cx, "native_ann.update.after_batch")?;
        self.admit(self.embedder.identity()?)
    }
}

#[cfg(feature = "quill")]
fn authority_reuse_fingerprint(
    plan: &TierPlan,
    expected: Option<&FrozenEmbeddingIdentityBundleV1>,
    generation: ArtifactGenerationIdentityV1,
    index: &NativeAnnIndex,
    intent: AuthorityUpdateIntent,
) -> SearchResult<Option<String>> {
    let expected = expected.ok_or_else(|| {
        invalid(
            "update.producer",
            "missing",
            "the selected generation must retain the complete tier identity",
        )
    })?;
    let binding = plan.binding(&generation)?;
    let actual = binding.frozen_identity();
    if actual == expected {
        index.admit_identity(&plan.identity)?;
        return Ok(Some(plan.identity.fingerprint()));
    }
    if matches!(intent, AuthorityUpdateIntent::RetainProducers) {
        return Err(invalid(
            "update.producer",
            "changed",
            "incremental reuse requires the original complete producer and storage identity",
        ));
    }
    if actual.identity.input != expected.identity.input {
        return Err(invalid(
            "update.input",
            "changed",
            "retained source cannot migrate its prepared-input contract; supply freshly prepared documents",
        ));
    }
    // Keep the original owner as a pin and source join witness, never as a
    // provider of vectors under this new model's identity.
    Ok(None)
}

fn admit_retained_policy(reference: &TierPlan, tier: &NativeBuiltTier) -> SearchResult<()> {
    let actual = retained_plan(tier)?;
    reference.admit(&actual.identity)?;
    let same_retrieval = match (reference.retrieval, actual.retrieval) {
        (NativeBuildRetrieval::Exact, NativeBuildRetrieval::Exact) => true,
        (
            NativeBuildRetrieval::Hnsw {
                params: left,
                seed: left_seed,
            },
            NativeBuildRetrieval::Hnsw {
                params: right,
                seed: right_seed,
            },
        ) => left == right && left_seed == right_seed,
        _ => false,
    };
    if reference.precision != actual.precision || !same_retrieval {
        return Err(invalid(
            "update.partition_policy",
            "mixed",
            "inheritance requires uniform per-tier precision and graph policy across partitions",
        ));
    }
    Ok(())
}

fn retained_plan(tier: &NativeBuiltTier) -> SearchResult<TierPlan> {
    let mut plan = TierPlan::new(Arc::clone(&tier.embedder))?;
    plan.admit(&tier.producer_identity)?;
    plan.precision = tier.precision;
    if let Some(receipt) = &tier.graph_receipt {
        let convert = |value| {
            usize::try_from(value).map_err(|_| {
                invalid(
                    "update.graph_parameters",
                    "overflow",
                    "retained graph parameters must fit usize",
                )
            })
        };
        let params = HnswParams {
            m: convert(receipt.params.m)?,
            m0: convert(receipt.params.m0)?,
            ef_construction: convert(receipt.params.ef_construction)?,
            ef_search: convert(receipt.params.ef_search)?,
        };
        params.validate()?;
        plan.retrieval = NativeBuildRetrieval::Hnsw {
            params,
            seed: receipt.seed,
        };
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Waker};

    use crate::{Embedder, ModelCategory, SearchError, SearchFuture};
    use frankensearch_core::generation::EmbeddingIdentityBundleV1;
    use frankensearch_core::traits::IdentityBoundEmbedding;

    const OK: usize = 0;
    const FAIL: usize = 1;
    const FOREIGN: usize = 2;
    const PENDING: usize = 3;
    const CANCEL: usize = 4;
    const IDENTITY_DRIFT: usize = 5;

    struct Provider {
        identity: EmbeddingIdentityBundleV1,
        foreign: EmbeddingIdentityBundleV1,
        submitted: AtomicUsize,
        calls: AtomicUsize,
        drops: AtomicUsize,
        fault: AtomicUsize,
    }

    impl Provider {
        fn new(name: &str, dimension: u32) -> Arc<Self> {
            Self::from_identity(EmbeddingIdentityBundleV1::explicit_test_model(
                name, dimension,
            ))
        }

        fn from_identity(identity: EmbeddingIdentityBundleV1) -> Arc<Self> {
            identity.validate().unwrap();
            let mut foreign = identity.clone();
            "different-producer".clone_into(&mut foreign.producer.backend);
            Arc::new(Self {
                identity,
                foreign,
                submitted: AtomicUsize::new(0),
                calls: AtomicUsize::new(0),
                drops: AtomicUsize::new(0),
                fault: AtomicUsize::new(OK),
            })
        }

        fn values(&self, text: &str) -> Vec<f32> {
            let mut values = vec![0.0; self.dimension()];
            let pair = match text {
                "vertical" => [0.0, 1.0],
                "fractional" => [0.333_333_34, 0.666_666_7],
                "diagonal" => [0.75, 0.25],
                _ => [1.0, 0.0],
            };
            values[..2].copy_from_slice(&pair);
            values
        }
    }

    struct BatchDrop<'a>(&'a AtomicUsize);
    impl Drop for BatchDrop<'_> {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl Embedder for Provider {
        fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async move { Ok(self.values(text)) })
        }

        fn embed_batch_bound<'a>(
            &'a self,
            cx: &'a Cx,
            texts: &'a [&'a str],
        ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                self.submitted.fetch_add(texts.len(), Ordering::SeqCst);
                let _guard = BatchDrop(&self.drops);
                let fault = self.fault.load(Ordering::SeqCst);
                if fault == PENDING {
                    return std::future::pending().await;
                }
                if fault == FAIL {
                    return Err(invalid("test.update", "failed", "injected batch failure"));
                }
                let mut output: Vec<_> = texts
                    .iter()
                    .map(|text| IdentityBoundEmbedding {
                        values: self.values(text),
                        identity: self.identity.clone(),
                    })
                    .collect();
                if fault == FOREIGN {
                    output
                        .last_mut()
                        .unwrap()
                        .identity
                        .clone_from(&self.foreign);
                }
                if fault == CANCEL {
                    cx.set_cancel_requested(true);
                }
                Ok(output)
            })
        }

        // Same values and identity as `embed`; faults violate the advertised
        // native contract on purpose.
        fn bound_batch_is_native(&self) -> bool {
            true
        }

        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(if self.fault.load(Ordering::SeqCst) == IDENTITY_DRIFT {
                &self.foreign
            } else {
                &self.identity
            })
        }

        fn id(&self) -> &str {
            &self.identity.space.logical_model_id
        }

        fn model_name(&self) -> &str {
            self.id()
        }

        fn dimension(&self) -> usize {
            usize::try_from(self.identity.space.dimension).unwrap()
        }

        fn is_semantic(&self) -> bool {
            false
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::HashEmbedder
        }
    }

    fn generation(sequence: u64) -> ArtifactGenerationIdentityV1 {
        ArtifactGenerationIdentityV1::new(sequence, [0x55; 16]).unwrap()
    }

    fn documents() -> Vec<IndexableDocument> {
        [
            ("z-fast", "horizontal"),
            ("b", "vertical"),
            ("a", "fractional"),
            ("x", "diagonal"),
            ("c", "horizontal"),
        ]
        .map(|(id, text)| IndexableDocument::new(id, text))
        .into()
    }

    fn graph(ann: bool) -> NativeBuildRetrieval {
        if ann {
            NativeBuildRetrieval::Hnsw {
                params: HnswParams {
                    ef_search: 8,
                    ..HnswParams::default()
                },
                seed: 53,
            }
        } else {
            NativeBuildRetrieval::Exact
        }
    }

    fn builder(path: &Path, fast: &Arc<Provider>, quality: &Arc<Provider>) -> NativeIndexBuilder {
        NativeIndexBuilder::new(path, generation(1), fast.clone())
            .unwrap()
            .with_quality_embedder(quality.clone())
            .unwrap()
            .add_documents(documents())
    }

    fn assert_same_tier(actual: &NativeBuiltTier, expected: &NativeBuiltTier) {
        assert_eq!(actual.index.owner_witness(), expected.index.owner_witness());
        assert_eq!(actual.graph_receipt, expected.graph_receipt);
        for row in 0..expected.index.owner.record_count() {
            assert_eq!(
                actual.index.owner.doc_id_at(row).unwrap(),
                expected.index.owner.doc_id_at(row).unwrap()
            );
            assert_eq!(
                actual
                    .index
                    .owner
                    .vector_at_f32(row)
                    .unwrap()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .index
                    .owner
                    .vector_at_f32(row)
                    .unwrap()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn updates_match_a_full_rebuild_but_embed_only_changed_documents_in_each_tier() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for precision in [NativeBuildPrecision::F16, NativeBuildPrecision::F32] {
                for ann in [false, true] {
                    let dir = tempfile::tempdir().unwrap();
                    let fast = Provider::new("update-fast", 2);
                    let quality = Provider::new("update-quality", 3);
                    let other_precision = if precision == NativeBuildPrecision::F16 {
                        NativeBuildPrecision::F32
                    } else {
                        NativeBuildPrecision::F16
                    };
                    let old = builder(&dir.path().join("old"), &fast, &quality)
                        .with_fast_storage(precision, graph(ann))
                        .with_quality_storage(other_precision, graph(!ann))
                        .unwrap()
                        .build(&cx)
                        .await
                        .unwrap();
                    let old_fast_bytes = std::fs::read(old.fast.vector_path()).unwrap();
                    let old_quality_bytes =
                        std::fs::read(old.quality().unwrap().vector_path()).unwrap();
                    assert_eq!(fast.submitted.load(Ordering::SeqCst), 5);
                    assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
                    let revised_a = IndexableDocument::new("a", "fractional")
                        .with_title("new title")
                        .with_metadata("revision", "2");
                    let next = old
                        .begin_update(&cx, dir.path().join("next"), generation(2))
                        .unwrap()
                        .upsert_document(IndexableDocument::new("b", "horizontal"))
                        .delete_document("c")
                        .upsert_document(revised_a.clone())
                        .upsert_document(IndexableDocument::new("new", "vertical"))
                        .with_batch_size(2)
                        .unwrap()
                        .build(&cx)
                        .await
                        .unwrap();
                    assert_eq!(fast.submitted.load(Ordering::SeqCst), 7);
                    assert_eq!(quality.submitted.load(Ordering::SeqCst), 7);
                    assert_eq!(next.fast.graph_path.is_some(), ann);
                    assert_eq!(next.quality().unwrap().graph_path.is_some(), !ann);
                    assert!(next.document("c").is_none());
                    assert_eq!(next.document("a").unwrap().metadata["revision"], "2");
                    assert_eq!(
                        next.document("a").unwrap().title.as_deref(),
                        Some("new title")
                    );
                    // The baseline re-embeds the explicitly specified final cohort.
                    // It does not derive expected vectors or rows from the update.
                    let rebuilt = NativeIndexBuilder::new(
                        dir.path().join("rebuilt"),
                        generation(2),
                        fast.clone(),
                    )
                    .unwrap()
                    .with_quality_embedder(quality.clone())
                    .unwrap()
                    .with_fast_storage(precision, graph(ann))
                    .with_quality_storage(other_precision, graph(!ann))
                    .unwrap()
                    .add_documents([
                        revised_a,
                        IndexableDocument::new("b", "horizontal"),
                        IndexableDocument::new("x", "diagonal"),
                        IndexableDocument::new("z-fast", "horizontal"),
                        IndexableDocument::new("new", "vertical"),
                    ])
                    .build(&cx)
                    .await
                    .unwrap();
                    assert_same_tier(next.fast(), rebuilt.fast());
                    assert_same_tier(next.quality().unwrap(), rebuilt.quality().unwrap());
                    assert_eq!(old.document("b").unwrap().content, "vertical");
                    assert!(old.document("c").is_some() && old.document("new").is_none());
                    assert_eq!(
                        std::fs::read(old.fast.vector_path()).unwrap(),
                        old_fast_bytes
                    );
                    assert_eq!(
                        std::fs::read(old.quality().unwrap().vector_path()).unwrap(),
                        old_quality_bytes
                    );
                    assert_eq!(
                        old.fast().search(&cx, "vertical", 1).await.unwrap()[0].doc_id,
                        "b"
                    );
                    assert_eq!(
                        next.fast().search(&cx, "vertical", 1).await.unwrap()[0].doc_id,
                        "new"
                    );
                }
            }
        });
    }

    #[cfg(feature = "quill")]
    #[test]
    fn incremental_native_sources_build_a_complete_two_tier_authority_candidate() {
        use frankensearch_core::generation::{
            GenerationAuthorityActionV1, GenerationComponentRole, SourceCheckpointV1,
        };
        use frankensearch_index::vector_cohort::ExactVectorCohortV1;
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let fast = Provider::new("update-fast", 2);
            let quality = Provider::new("update-quality", 3);
            let old = builder(&directory.path().join("old"), &fast, &quality)
                .with_fast_storage(NativeBuildPrecision::F16, NativeBuildRetrieval::Exact)
                .with_quality_storage(NativeBuildPrecision::F32, NativeBuildRetrieval::Exact)
                .unwrap()
                .build(&cx)
                .await
                .unwrap();
            let old_fast = std::fs::read(old.fast.vector_path()).unwrap();
            let old_quality = std::fs::read(old.quality().unwrap().vector_path()).unwrap();
            let source = CommitRange { low: 3, high: 9 };
            let budget = 16 * 1024 * 1024;
            let built = old
                .begin_update(&cx, directory.path().join("next"), generation(2))
                .unwrap()
                .delete_document("x")
                .upsert_document(
                    IndexableDocument::new("a", "fractional").with_metadata("version", "updated"),
                )
                .upsert_document(IndexableDocument::new("b", "horizontal"))
                .upsert_document(IndexableDocument::new("inserted", "vertical"))
                .build_progressive_generation(&cx, source.clone(), budget)
                .await
                .unwrap();
            assert_eq!(fast.submitted.load(Ordering::SeqCst), 7);
            assert_eq!(quality.submitted.load(Ordering::SeqCst), 7);
            assert_eq!(std::fs::read(old.fast.vector_path()).unwrap(), old_fast);
            assert_eq!(
                std::fs::read(old.quality().unwrap().vector_path()).unwrap(),
                old_quality
            );
            let candidate = built
                .candidate(
                    [0xb1; 16],
                    None,
                    GenerationAuthorityActionV1::Activate,
                    [0xb2; 32],
                )
                .unwrap();
            built.validate_candidate(&cx, &candidate).unwrap();
            let cohort = ExactVectorCohortV1::admit(
                &cx,
                built
                    .component_bytes(GenerationComponentRole::Vector)
                    .unwrap(),
                built.binding(),
                built.quality_binding().unwrap(),
                SourceCheckpointV1::derive(&source),
                budget,
            )
            .unwrap();
            assert_eq!(cohort.fast().dimension(), 2);
            assert_eq!(cohort.quality().dimension(), 3);
            assert_eq!(cohort.fast().live_count(), 5);
            assert_eq!(cohort.quality().live_count(), 5);
            for (previous, next) in [
                (&old.fast.index.owner, cohort.fast()),
                (&old.quality().unwrap().index.owner, cohort.quality()),
            ] {
                let row = |owner: &ValidatedFsviBytes, id: &str| {
                    (0..owner.record_count())
                        .find(|physical| owner.row(*physical).unwrap().doc_id() == id)
                        .unwrap()
                };
                assert_eq!(
                    previous.vector_at_f32(row(previous, "a")).unwrap(),
                    next.vector_at_f32(row(next, "a")).unwrap()
                );
                assert!(
                    (0..next.record_count())
                        .all(|physical| next.row(physical).unwrap().doc_id() != "x")
                );
                assert_eq!(next.vector_at_f32(row(next, "b")).unwrap()[0], 1.0);
            }
            assert_eq!(built.metadata().total_documents, 5);
            assert_eq!(built.metadata().commit_range, source);
        });
    }

    #[test]
    fn metadata_only_noop_and_delete_all_updates_do_not_start_inference() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let old = builder(&dir.path().join("old"), &fast, &quality)
                .build(&cx)
                .await
                .unwrap();
            // Both providers would fail if any batch were submitted.
            fast.fault.store(FAIL, Ordering::SeqCst);
            quality.fault.store(FAIL, Ordering::SeqCst);
            let metadata = old
                .begin_update(&cx, dir.path().join("metadata"), generation(2))
                .unwrap()
                .upsert_document(IndexableDocument::new("b", "vertical").with_metadata("v", "new"))
                .delete_document("absent")
                .build(&cx)
                .await
                .unwrap();
            assert_eq!(metadata.document("b").unwrap().metadata["v"], "new");
            let noop = metadata
                .begin_update(&cx, dir.path().join("noop"), generation(3))
                .unwrap()
                .build(&cx)
                .await
                .unwrap();
            let mut update = noop
                .begin_update(&cx, dir.path().join("empty"), generation(4))
                .unwrap();
            for document in documents() {
                update = update.delete_document(document.id);
            }
            let empty = update.build(&cx).await.unwrap();
            assert!(empty.documents().is_empty());
            assert_eq!(empty.fast().index().live_count(), 0);
            assert_eq!(empty.quality().unwrap().index().live_count(), 0);
            assert_eq!(fast.submitted.load(Ordering::SeqCst), 5);
            assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
            assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
        });
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn append_updates_reuse_both_graphs_across_row_reordering_and_selected_restart() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let fast = Provider::new("append-fast", 2);
            let quality = Provider::new("append-quality", 3);
            let old = builder(&dir.path().join("old"), &fast, &quality)
                .with_fast_storage(NativeBuildPrecision::F16, graph(true))
                .with_quality_storage(NativeBuildPrecision::F32, graph(true))
                .unwrap()
                .build(&cx)
                .await
                .unwrap();
            let old_graph = std::fs::read(old.fast().graph_path().unwrap()).unwrap();
            let additions: Vec<_> = (0..20)
                .map(|n| IndexableDocument::new(format!("arrival-{n}"), "vertical"))
                .collect();
            let next_path = dir.path().join("next");
            let next = old
                .begin_update(&cx, &next_path, generation(2))
                .unwrap()
                .upsert_documents(additions.clone())
                .upsert_document(
                    IndexableDocument::new("a", "fractional").with_title("updated metadata"),
                )
                .build(&cx)
                .await
                .unwrap();
            assert_eq!(fast.submitted.load(Ordering::SeqCst), 25);
            assert_eq!(quality.submitted.load(Ordering::SeqCst), 25);
            assert!(
                (0..old.fast().index.owner.record_count()).any(|row| {
                    old.fast().index.owner.doc_id_at(row).unwrap()
                        != next.fast().index.owner.doc_id_at(row).unwrap()
                }),
                "the fixture must exercise physical-row remapping",
            );

            // An independently built graph starts with the successor's physical
            // row order; the extended graph must retain the old logical nodes.
            let rebuilt =
                NativeIndexBuilder::new(dir.path().join("rebuilt"), generation(2), fast.clone())
                    .unwrap()
                    .with_quality_embedder(quality.clone())
                    .unwrap()
                    .with_fast_storage(NativeBuildPrecision::F16, graph(true))
                    .with_quality_storage(NativeBuildPrecision::F32, graph(true))
                    .unwrap()
                    .add_documents(documents())
                    .add_documents(additions)
                    .build(&cx)
                    .await
                    .unwrap();
            for (actual, cold) in [
                (next.fast(), rebuilt.fast()),
                (next.quality().unwrap(), rebuilt.quality().unwrap()),
            ] {
                assert_eq!(actual.index.owner_witness(), cold.index.owner_witness());
                assert_ne!(
                    actual.graph_receipt.as_ref().unwrap().topology_sha256,
                    cold.graph_receipt.as_ref().unwrap().topology_sha256,
                    "an eligible update must extend the retained graph, not rebuild it",
                );
                for query in ["vertical", "horizontal", "diagonal", "fractional"] {
                    assert_eq!(
                        actual.search(&cx, query, 25).await.unwrap(),
                        cold.search(&cx, query, 25).await.unwrap(),
                    );
                    let hits = actual
                        .index()
                        .search_text_filtered(&cx, actual.embedder(), query, 1, Some(4), |id| {
                            id == "arrival-19"
                        })
                        .await
                        .unwrap();
                    assert_eq!(hits[0].doc_id, "arrival-19");
                }
            }
            assert_eq!(
                std::fs::read(old.fast().graph_path().unwrap()).unwrap(),
                old_graph
            );
            assert_eq!(old.fast().index().live_count(), 5);
            assert_eq!(
                old.fast().search(&cx, "vertical", 1).await.unwrap()[0].doc_id,
                "b"
            );
            let receipt = next.seal_for_reopen(&cx).unwrap();
            drop(next);
            let reopened = NativeBuiltIndex::open_selected(
                &cx,
                &next_path,
                &receipt,
                fast.clone(),
                Some(quality.clone()),
            )
            .unwrap();
            assert_eq!(
                reopened.document("a").unwrap().title.as_deref(),
                Some("updated metadata")
            );
            for (actual, cold) in [
                (reopened.fast(), rebuilt.fast()),
                (reopened.quality().unwrap(), rebuilt.quality().unwrap()),
            ] {
                assert_eq!(
                    actual.search(&cx, "vertical", 25).await.unwrap(),
                    cold.search(&cx, "vertical", 25).await.unwrap(),
                );
            }

            // Restarted handles retain the remapping for a later metadata-only
            // successor without invoking either provider again.
            fast.fault.store(FAIL, Ordering::SeqCst);
            quality.fault.store(FAIL, Ordering::SeqCst);
            let metadata = reopened
                .begin_update(&cx, dir.path().join("metadata"), generation(3))
                .unwrap()
                .upsert_document(IndexableDocument::new("b", "vertical").with_metadata("rev", "3"))
                .build(&cx)
                .await
                .unwrap();
            assert_eq!(
                metadata
                    .fast()
                    .graph_receipt
                    .as_ref()
                    .unwrap()
                    .topology_sha256,
                reopened
                    .fast()
                    .graph_receipt
                    .as_ref()
                    .unwrap()
                    .topology_sha256,
            );
            assert_eq!(metadata.document("b").unwrap().metadata["rev"], "3");
        });
    }

    #[test]
    fn precision_change_reembeds_only_that_tier_instead_of_promoting_rounded_vectors() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let old = builder(&dir.path().join("old"), &fast, &quality)
                .with_fast_storage(NativeBuildPrecision::F16, graph(false))
                .build(&cx)
                .await
                .unwrap();
            let next = old
                .begin_update(&cx, dir.path().join("next"), generation(2))
                .unwrap()
                .with_fast_storage(NativeBuildPrecision::F32, graph(true))
                .build(&cx)
                .await
                .unwrap();
            assert_eq!(fast.submitted.load(Ordering::SeqCst), 10);
            assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
            let owner = &next.fast.index.owner;
            let row = (0..owner.record_count())
                .find(|&row| owner.doc_id_at(row).unwrap() == "a")
                .unwrap();
            let actual = owner.vector_at_f32(row).unwrap();
            assert_eq!(actual, fast.values("fractional"));
            let old_owner = &old.fast.index.owner;
            let old_row = (0..old_owner.record_count())
                .find(|&row| old_owner.doc_id_at(row).unwrap() == "a")
                .unwrap();
            assert_ne!(actual, old_owner.vector_at_f32(old_row).unwrap());
        });
    }

    #[test]
    fn ordered_edits_do_not_resurrect_deletions_and_new_ids_are_embedded() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let fast = Provider::new("fast", 2);
            let old = NativeIndexBuilder::new(dir.path().join("old"), generation(1), fast.clone())
                .unwrap()
                .add_document(IndexableDocument::new("old-id", "vertical"))
                .build(&cx)
                .await
                .unwrap();
            let next = old
                .begin_update(&cx, dir.path().join("next"), generation(2))
                .unwrap()
                .delete_document("old-id")
                .upsert_document(IndexableDocument::new("new-id", "horizontal"))
                .delete_document("new-id")
                .upsert_document(IndexableDocument::new("new-id", "vertical"))
                .upsert_document(IndexableDocument::new("temporary", "horizontal"))
                .delete_document("temporary")
                .build(&cx)
                .await
                .unwrap();
            assert_eq!(next.documents().len(), 1);
            assert_eq!(next.documents()[0].id, "new-id");
            assert_eq!(next.documents()[0].content, "vertical");
            assert!(next.quality().is_none());
            assert_eq!(fast.submitted.load(Ordering::SeqCst), 2);
            assert_eq!(old.documents()[0].id, "old-id");
        });
    }

    #[test]
    fn failed_or_cancelled_quality_updates_preserve_the_complete_predecessor() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for fault in [FAIL, FOREIGN, CANCEL] {
                let dir = tempfile::tempdir().unwrap();
                let fast = Provider::new("fast", 2);
                let quality = Provider::new("quality", 3);
                let old = builder(&dir.path().join("old"), &fast, &quality)
                    .build(&cx)
                    .await
                    .unwrap();
                let old_bytes = std::fs::read(old.fast.vector_path()).unwrap();
                quality.fault.store(fault, Ordering::SeqCst);
                let path = dir.path().join("failed");
                let result = old
                    .begin_update(&cx, &path, generation(2))
                    .unwrap()
                    .upsert_document(IndexableDocument::new("new", "vertical"))
                    .build(&cx)
                    .await;
                assert!(result.is_err());
                if fault == CANCEL {
                    assert!(matches!(result, Err(SearchError::Cancelled { .. })));
                    cx.set_cancel_requested(false);
                }
                assert!(!path.join("native.snapshot.json").exists());
                assert!(!path.join("fast.fsvi").exists());
                assert_eq!(old.fast().index().live_count(), 5);
                assert_eq!(old.quality().unwrap().index().live_count(), 5);
                assert_eq!(std::fs::read(old.fast.vector_path()).unwrap(), old_bytes);
                assert_eq!(
                    old.fast().search(&cx, "vertical", 1).await.unwrap()[0].doc_id,
                    "b"
                );
                assert_eq!(
                    old.quality()
                        .unwrap()
                        .search(&cx, "vertical", 1)
                        .await
                        .unwrap()[0]
                        .doc_id,
                    "b"
                );
            }
        });
    }

    #[test]
    fn pending_update_keeps_old_readers_live_and_drop_does_not_publish_a_partial_generation() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let old = builder(&dir.path().join("old"), &fast, &quality)
                .build(&cx)
                .await
                .unwrap();
            quality.fault.store(PENDING, Ordering::SeqCst);
            let path = dir.path().join("abandoned");
            let update = old
                .begin_update(&cx, &path, generation(2))
                .unwrap()
                .upsert_document(IndexableDocument::new("new", "vertical"));
            let mut future = Box::pin(update.build(&cx));
            assert!(
                future
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            assert_eq!(
                old.fast().search(&cx, "vertical", 1).await.unwrap()[0].doc_id,
                "b"
            );
            assert_eq!(
                old.quality()
                    .unwrap()
                    .search(&cx, "vertical", 1)
                    .await
                    .unwrap()[0]
                    .doc_id,
                "b"
            );
            drop(future);
            assert_eq!(quality.drops.load(Ordering::SeqCst), 2);
            assert!(!path.join("fast.fsvi").exists());
            quality.fault.store(OK, Ordering::SeqCst);
            let retry = old
                .begin_update(&cx, dir.path().join("retry"), generation(2))
                .unwrap()
                .upsert_document(IndexableDocument::new("new", "vertical"))
                .build(&cx)
                .await
                .unwrap();
            assert_eq!(retry.documents().len(), 6);
            assert!(old.document("new").is_none());
        });
    }

    #[test]
    fn generation_provider_and_edit_admission_precede_any_files_or_inference() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let old = builder(&dir.path().join("old"), &fast, &quality)
                .build(&cx)
                .await
                .unwrap();
            let path = dir.path().join("rejected");
            assert!(old.begin_update(&cx, &path, generation(1)).is_err());
            assert!(
                old.begin_update(
                    &cx,
                    &path,
                    ArtifactGenerationIdentityV1::new(1, [0x56; 16]).unwrap(),
                )
                .is_err()
            );
            quality.fault.store(IDENTITY_DRIFT, Ordering::SeqCst);
            assert!(old.begin_update(&cx, &path, generation(2)).is_err());
            quality.fault.store(OK, Ordering::SeqCst);
            let update = old.begin_update(&cx, &path, generation(2)).unwrap();
            quality.fault.store(IDENTITY_DRIFT, Ordering::SeqCst);
            assert!(update.build(&cx).await.is_err());
            quality.fault.store(OK, Ordering::SeqCst);
            assert!(
                old.begin_update(&cx, &path, generation(2))
                    .unwrap()
                    .delete_document("")
                    .build(&cx)
                    .await
                    .is_err()
            );
            assert!(
                old.begin_update(&cx, &path, generation(2))
                    .unwrap()
                    .upsert_document(IndexableDocument::new("", "horizontal"))
                    .build(&cx)
                    .await
                    .is_err()
            );
            assert!(!path.exists());
            assert_eq!(fast.submitted.load(Ordering::SeqCst), 5);
            assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
            let result = old
                .begin_update(&cx, old.directory(), generation(2))
                .unwrap()
                .build(&cx)
                .await;
            assert!(result.is_err());
            assert_eq!(fast.submitted.load(Ordering::SeqCst), 5);
        });
    }

    #[test]
    fn reuse_reads_retained_owners_not_the_predecessors_current_pathnames() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let old_path = dir.path().join("old");
            let old = builder(&old_path, &fast, &quality)
                .build(&cx)
                .await
                .unwrap();
            let update = old
                .begin_update(&cx, dir.path().join("next"), generation(2))
                .unwrap();
            std::fs::rename(&old_path, dir.path().join("archived")).unwrap();
            assert!(!old_path.exists());
            drop(old);
            let next = update.build(&cx).await.unwrap();
            assert_eq!(next.documents().len(), 5);
            assert_eq!(fast.submitted.load(Ordering::SeqCst), 5);
            assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
            assert_eq!(
                next.fast().search(&cx, "vertical", 1).await.unwrap()[0].doc_id,
                "b"
            );
        });
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn selected_restart_can_stage_and_seal_another_update_without_reembedding_survivors() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let path = dir.path().join("old");
            let old = builder(&path, &fast, &quality)
                .with_fast_storage(NativeBuildPrecision::F16, graph(true))
                .build(&cx)
                .await
                .unwrap();
            let receipt = old.seal_for_reopen(&cx).unwrap();
            drop(old);
            let reopened = NativeBuiltIndex::open_selected(
                &cx,
                &path,
                &receipt,
                fast.clone(),
                Some(quality.clone()),
            )
            .unwrap();
            let next_path = dir.path().join("next");
            let next = reopened
                .begin_update(&cx, &next_path, generation(2))
                .unwrap()
                .delete_document("b")
                .build(&cx)
                .await
                .unwrap();
            assert_eq!(fast.submitted.load(Ordering::SeqCst), 5);
            assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
            assert!(next.fast.graph_path().is_some());
            let next_receipt = next.seal_for_reopen(&cx).unwrap();
            drop(next);
            let selected = NativeBuiltIndex::open_selected(
                &cx,
                &next_path,
                &next_receipt,
                fast.clone(),
                Some(quality.clone()),
            )
            .unwrap();
            assert!(selected.document("b").is_none());
            assert!(reopened.document("b").is_some());
            assert_eq!(
                selected.fast.index.owner_witness().generation,
                generation(2)
            );
            assert_eq!(
                selected.quality().unwrap().index.owner_witness().generation,
                generation(2)
            );
        });
    }

    #[cfg(all(feature = "quill", any(target_os = "linux", target_os = "macos")))]
    #[test]
    fn hybrid_updates_replace_lexical_text_and_membership_without_changing_old_phases() {
        use crate::native_ann::NativeSearchPhase;

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let old = NativeIndexBuilder::new(dir.path().join("old"), generation(1), fast.clone())
                .unwrap()
                .with_quality_embedder(quality.clone())
                .unwrap()
                .with_fast_storage(NativeBuildPrecision::F16, graph(true))
                .add_documents([
                    IndexableDocument::new("a", "retained common"),
                    IndexableDocument::new("b", "obsolete common").with_metadata("revision", "old"),
                    IndexableDocument::new("c", "removed common"),
                ])
                .build_hybrid(&cx)
                .await
                .unwrap();
            let mut old_stream = old.progressive(&cx, "obsolete", 1).unwrap();
            let Some(NativeSearchPhase::Initial { results, .. }) =
                old_stream.next_phase().await.unwrap()
            else {
                panic!("initial");
            };
            assert_eq!(results[0].doc_id, "b");
            assert!(results[0].lexical_score.is_some());

            let path = dir.path().join("next");
            let next = old
                .begin_update(&cx, &path, generation(2))
                .unwrap()
                .delete_document("c")
                .upsert_documents([
                    IndexableDocument::new("b", "replacement common")
                        .with_metadata("revision", "new"),
                    IndexableDocument::new("d", "arrival common"),
                ])
                .build_hybrid(&cx)
                .await
                .unwrap();
            assert_eq!(fast.submitted.load(Ordering::SeqCst), 5);
            assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
            assert_eq!(old.lexical().doc_count().unwrap(), 3);
            assert_eq!(next.lexical().doc_count().unwrap(), 3);
            assert!(
                next.lexical()
                    .search(&cx, "obsolete", 10)
                    .await
                    .unwrap()
                    .is_empty()
            );
            assert!(
                next.lexical()
                    .search(&cx, "removed", 10)
                    .await
                    .unwrap()
                    .is_empty()
            );
            assert_eq!(
                next.lexical().search(&cx, "replacement", 10).await.unwrap()[0].doc_id,
                "b"
            );
            assert_eq!(
                next.lexical().search(&cx, "arrival", 10).await.unwrap()[0].doc_id,
                "d"
            );
            assert_eq!(
                old.lexical().search(&cx, "removed", 10).await.unwrap()[0].doc_id,
                "c"
            );
            let Some(NativeSearchPhase::Refined { results, .. }) =
                old_stream.next_phase().await.unwrap()
            else {
                panic!("old stream retains its admitted generation");
            };
            assert_eq!(results[0].doc_id, "b");
            assert!(results[0].lexical_score.is_some());
            assert_eq!(
                old.vectors().document("b").unwrap().metadata["revision"],
                "old"
            );
            assert_eq!(
                next.vectors().document("b").unwrap().metadata["revision"],
                "new"
            );

            let receipt = next.seal_for_reopen(&cx).unwrap();
            drop(next);
            let reopened = NativeBuiltHybridIndex::open_selected(
                &cx,
                &path,
                &receipt,
                fast.clone(),
                Some(quality.clone()),
            )
            .await
            .unwrap();
            assert!(reopened.vectors().document("c").is_none());
            assert_eq!(
                reopened.lexical().search(&cx, "arrival", 10).await.unwrap()[0].doc_id,
                "d"
            );
            assert_eq!(
                reopened.vectors().document("b").unwrap().content,
                "replacement common"
            );
            assert_eq!(fast.submitted.load(Ordering::SeqCst), 5);
            assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
        });
    }

    #[cfg(feature = "quill")]
    #[test]
    fn pending_hybrid_update_does_not_hold_the_predecessors_lexical_writer_or_reader() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let old = builder(&dir.path().join("old"), &fast, &quality)
                .build_hybrid(&cx)
                .await
                .unwrap();
            quality.fault.store(PENDING, Ordering::SeqCst);
            let path = dir.path().join("pending");
            let update = old
                .begin_update(&cx, &path, generation(2))
                .unwrap()
                .delete_document("b")
                .upsert_document(IndexableDocument::new("new", "vertical"));
            let mut future = Box::pin(update.build_hybrid(&cx));
            assert!(
                future
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            assert_eq!(
                old.lexical().search(&cx, "vertical", 10).await.unwrap()[0].doc_id,
                "b"
            );
            assert_eq!(old.search(&cx, "vertical", 1).await.unwrap()[0].doc_id, "b");
            drop(future);
            assert_eq!(quality.drops.load(Ordering::SeqCst), 2);
            assert!(!path.join("native.hybrid.json").exists());
            assert!(!path.join("lexical").exists());
            assert_eq!(old.lexical().doc_count().unwrap(), 5);
        });
    }

    #[cfg(feature = "quill")]
    mod model_migrations {
        use super::*;
        use frankensearch_core::generation::{
            GenerationAuthorityActionV1, GenerationComponentRole, QuantizationFormat,
        };

        const IMAGE_BUDGET: usize = 8 * 1024 * 1024;

        fn replacement(provider: &Provider) -> Arc<Provider> {
            let mut identity = provider.identity.clone();
            identity
                .producer
                .implementation_revision
                .push_str("-migrated");
            Provider::from_identity(identity)
        }

        // Exercise the production identity admission and retained-owner reuse
        // against real FSVI images without requiring a qualified filesystem.
        // Selected-authority/source pinning is covered by the lifecycle tests.
        fn migration(
            cx: &Cx,
            path: &Path,
            old: &NativeBuiltIndex,
            fast: &Arc<Provider>,
            quality: &Arc<Provider>,
        ) -> SearchResult<NativeIndexUpdate> {
            let old_quality = old.quality().unwrap();
            let mut builder = NativeIndexBuilder::new(path, generation(2), fast.clone())?
                .with_fast_storage(old.fast.precision, retained_plan(&old.fast)?.retrieval)
                .with_quality_embedder(quality.clone())?
                .with_quality_storage(
                    old_quality.precision,
                    retained_plan(old_quality)?.retrieval,
                )?;
            let reuse_tier = |plan: &TierPlan, retained: &NativeBuiltTier| {
                let fingerprint = authority_reuse_fingerprint(
                    plan,
                    Some(retained.binding().frozen_identity()),
                    retained.binding().generation(),
                    retained.index(),
                    AuthorityUpdateIntent::MigrateModels,
                )?;
                ReuseTier::from_owner(
                    cx,
                    Arc::clone(&retained.index),
                    old.documents(),
                    fingerprint,
                    retained.precision,
                )
            };
            builder.reuse = Some(Arc::new(ReuseSource {
                partitions: vec![ReusePartition {
                    documents: Arc::clone(&old.documents),
                    fast: reuse_tier(&builder.fast, &old.fast)?,
                    quality: Some(reuse_tier(builder.quality.as_ref().unwrap(), old_quality)?),
                }],
                authority: None,
            }));
            Ok(NativeIndexUpdate {
                builder,
                edits: BTreeMap::new(),
            })
        }

        #[test]
        fn model_migration_reuses_only_the_identical_tier_and_matches_complete_cold_components() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for precision in [NativeBuildPrecision::F16, NativeBuildPrecision::F32] {
                    for ann in [false, true] {
                        for change_fast in [false, true] {
                            let dir = tempfile::tempdir().unwrap();
                            let fast = Provider::new("migration-fast", 2);
                            let quality = Provider::new("migration-quality", 3);
                            let quality_precision = if precision == NativeBuildPrecision::F16 {
                                NativeBuildPrecision::F32
                            } else {
                                NativeBuildPrecision::F16
                            };
                            let old = Box::pin(
                                builder(&dir.path().join("old"), &fast, &quality)
                                    .with_fast_storage(precision, graph(ann))
                                    .with_quality_storage(quality_precision, graph(!ann))
                                    .unwrap()
                                    .build(&cx),
                            )
                            .await
                            .unwrap();
                            let old_fast = std::fs::read(old.fast.vector_path()).unwrap();
                            let old_quality =
                                std::fs::read(old.quality().unwrap().vector_path()).unwrap();
                            let new_fast = if change_fast {
                                replacement(&fast)
                            } else {
                                fast.clone()
                            };
                            let new_quality = if change_fast {
                                quality.clone()
                            } else {
                                replacement(&quality)
                            };
                            let revised = IndexableDocument::new("a", "fractional")
                                .with_title("retained metadata updated");
                            let next = migration(
                                &cx,
                                &dir.path().join("next"),
                                &old,
                                &new_fast,
                                &new_quality,
                            )
                            .unwrap()
                            .upsert_document(revised.clone())
                            .with_batch_size(2)
                            .unwrap()
                            .build_progressive_generation(
                                &cx,
                                CommitRange { low: 1, high: 5 },
                                IMAGE_BUDGET,
                            )
                            .await
                            .unwrap();
                            // Replacement models deliberately have the SAME ID,
                            // dimension and output values. Revision alone must
                            // force inference for the complete migrated tier.
                            assert_eq!(fast.submitted.load(Ordering::SeqCst), 5);
                            assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
                            assert_eq!(new_fast.submitted.load(Ordering::SeqCst), 5);
                            assert_eq!(new_quality.submitted.load(Ordering::SeqCst), 5);
                            assert_eq!(next.retrieval_configuration(), graph(ann));
                            assert_eq!(next.quality_retrieval_configuration(), Some(graph(!ann)));
                            assert_eq!(next.metadata().total_documents, 5);
                            assert_eq!(
                                next.binding().frozen_identity()
                                    == old.fast.binding().frozen_identity(),
                                !change_fast
                            );
                            assert_eq!(
                                next.quality_binding().unwrap().frozen_identity()
                                    == old.quality().unwrap().binding().frozen_identity(),
                                change_fast
                            );
                            let candidate = next
                                .candidate(
                                    [0x38; 16],
                                    None,
                                    GenerationAuthorityActionV1::Activate,
                                    [0x39; 32],
                                )
                                .unwrap();
                            let preview = next.preview_candidate(&cx, &candidate).unwrap();
                            let hits = preview
                                .search_refined_text(
                                    &cx,
                                    new_fast.as_ref(),
                                    new_quality.as_ref(),
                                    "vertical",
                                    3,
                                )
                                .await
                                .unwrap();
                            assert!(hits.iter().any(|hit| hit.doc_id == "b"));
                            let cold = NativeIndexBuilder::new(
                                dir.path().join("cold"),
                                generation(2),
                                new_fast,
                            )
                            .unwrap()
                            .with_quality_embedder(new_quality)
                            .unwrap()
                            .with_fast_storage(precision, graph(ann))
                            .with_quality_storage(quality_precision, graph(!ann))
                            .unwrap()
                            .add_documents(documents().into_iter().map(|document| {
                                if document.id == "a" {
                                    revised.clone()
                                } else {
                                    document
                                }
                            }))
                            .build_progressive_generation(
                                &cx,
                                CommitRange { low: 1, high: 5 },
                                IMAGE_BUDGET,
                            )
                            .await
                            .unwrap();
                            for role in [
                                GenerationComponentRole::Vector,
                                GenerationComponentRole::Ann,
                            ] {
                                assert_eq!(next.component_bytes(role), cold.component_bytes(role));
                            }
                            assert_eq!(std::fs::read(old.fast.vector_path()).unwrap(), old_fast);
                            assert_eq!(
                                std::fs::read(old.quality().unwrap().vector_path()).unwrap(),
                                old_quality
                            );
                        }
                    }
                }
            });
        }

        #[test]
        fn model_migration_normalizes_storage_but_changed_source_still_requires_inference() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let dir = tempfile::tempdir().unwrap();
                let fast = Provider::new("migration-fast", 2);
                let quality = Provider::new("migration-quality", 3);
                let old = Box::pin(
                    builder(&dir.path().join("old"), &fast, &quality)
                        .with_fast_storage(NativeBuildPrecision::F16, graph(true))
                        .with_quality_storage(NativeBuildPrecision::F32, graph(false))
                        .unwrap()
                        .build(&cx),
                )
                .await
                .unwrap();
                let old_fast_bytes = std::fs::read(old.fast.vector_path()).unwrap();
                let old_quality_bytes =
                    std::fs::read(old.quality().unwrap().vector_path()).unwrap();
                let mut identity = fast.identity.clone();
                identity.storage.format = "in-memory-another-provider-output-v1".to_owned();
                let equivalent = Provider::from_identity(identity);
                let unchanged =
                    migration(&cx, &dir.path().join("same"), &old, &equivalent, &quality)
                        .unwrap()
                        .build(&cx)
                        .await
                        .unwrap();
                assert_eq!(equivalent.submitted.load(Ordering::SeqCst), 0);
                assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
                assert_eq!(
                    unchanged.fast.binding().frozen_identity(),
                    old.fast.binding().frozen_identity()
                );
                let changed = migration(
                    &cx,
                    &dir.path().join("changed"),
                    &old,
                    &equivalent,
                    &quality,
                )
                .unwrap()
                .upsert_document(IndexableDocument::new("a", "vertical"))
                .build(&cx)
                .await
                .unwrap();
                assert_eq!(equivalent.submitted.load(Ordering::SeqCst), 1);
                assert_eq!(quality.submitted.load(Ordering::SeqCst), 6);
                assert_eq!(changed.document("a").unwrap().content, "vertical");
                assert_eq!(old.document("a").unwrap().content, "fractional");
                // A provider returning Vec<f32> must still identify its actual
                // output storage, even when the retained FSVI uses f16.
                let mut incompatible_identity = fast.identity.clone();
                "fsvi-v2".clone_into(&mut incompatible_identity.storage.format);
                incompatible_identity.storage.quantization = QuantizationFormat::F16;
                "little-endian".clone_into(&mut incompatible_identity.storage.endianness);
                let incompatible = Provider::from_identity(incompatible_identity);
                let error = migration(
                    &cx,
                    &dir.path().join("misdeclared-output"),
                    &old,
                    &incompatible,
                    &quality,
                )
                .unwrap()
                .upsert_document(IndexableDocument::new("a", "vertical"))
                .build(&cx)
                .await
                .err()
                .expect("a Vec<f32> response cannot identify itself as f16");
                assert!(
                    matches!(error, crate::SearchError::InvalidConfig { field, .. }
                    if field == "identity_bound_embedding.storage.quantization")
                );
                assert_eq!(incompatible.submitted.load(Ordering::SeqCst), 1);
                // A distinct F32 output label can normalize for retained-vector
                // reuse, but actual Vec<f32> inference still needs an in-memory
                // format. A changed source must reach this response check.
                let mut invalid_format_identity = fast.identity.clone();
                invalid_format_identity.storage.format = "another-provider-output-label".to_owned();
                let invalid_format = Provider::from_identity(invalid_format_identity);
                let error = migration(
                    &cx,
                    &dir.path().join("misdeclared-format"),
                    &old,
                    &invalid_format,
                    &quality,
                )
                .unwrap()
                .upsert_document(IndexableDocument::new("a", "vertical"))
                .build(&cx)
                .await
                .err()
                .expect("a Vec<f32> response requires an in-memory format");
                assert!(
                    matches!(error, crate::SearchError::InvalidConfig { field, value, .. }
                    if field == "identity_bound_embedding.storage.format"
                        && value == "another-provider-output-label")
                );
                assert_eq!(invalid_format.submitted.load(Ordering::SeqCst), 1);
                assert_eq!(quality.submitted.load(Ordering::SeqCst), 6);
                assert_eq!(old.document("a").unwrap().content, "fractional");
                assert_eq!(
                    std::fs::read(old.fast.vector_path()).unwrap(),
                    old_fast_bytes
                );
                assert_eq!(
                    std::fs::read(old.quality().unwrap().vector_path()).unwrap(),
                    old_quality_bytes
                );
            });
        }

        #[test]
        fn model_migration_refuses_input_contract_changes_and_preserves_strict_normal_updates() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let dir = tempfile::tempdir().unwrap();
                let fast = Provider::new("migration-fast", 2);
                let quality = Provider::new("migration-quality", 3);
                let old = Box::pin(builder(&dir.path().join("old"), &fast, &quality).build(&cx))
                    .await
                    .unwrap();
                for change_fast in [false, true] {
                    let original = if change_fast { &fast } else { &quality };
                    let mut identity = original.identity.clone();
                    identity.input.canonicalization.push_str("-new-preparation");
                    identity.space.input_contract_fingerprint = identity.input.fingerprint();
                    identity.producer.space_fingerprint = identity.space.fingerprint();
                    let changed = Provider::from_identity(identity);
                    let path = dir.path().join(format!("input-{change_fast}"));
                    let result = migration(
                        &cx,
                        &path,
                        &old,
                        if change_fast { &changed } else { &fast },
                        if change_fast { &quality } else { &changed },
                    );
                    assert!(
                        matches!(result, Err(SearchError::InvalidConfig { field, .. }) if field == "native_ann.update.input")
                    );
                    assert!(!path.exists());
                    assert_eq!(changed.submitted.load(Ordering::SeqCst), 0);
                }
                let changed = replacement(&fast);
                let plan = TierPlan::new(changed.clone()).unwrap();
                assert!(
                    authority_reuse_fingerprint(
                        &plan,
                        Some(old.fast.binding().frozen_identity()),
                        generation(1),
                        old.fast.index(),
                        AuthorityUpdateIntent::RetainProducers,
                    )
                    .is_err()
                );
                assert_eq!(fast.submitted.load(Ordering::SeqCst), 5);
                assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
                assert_eq!(changed.submitted.load(Ordering::SeqCst), 0);
            });
        }

        #[test]
        fn model_migration_failure_cancellation_and_provider_drift_keep_the_predecessor_intact() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for fault in [FAIL, FOREIGN, CANCEL, IDENTITY_DRIFT] {
                    let dir = tempfile::tempdir().unwrap();
                    let fast = Provider::new("migration-fast", 2);
                    let quality = Provider::new("migration-quality", 3);
                    let old = Box::pin(
                        builder(&dir.path().join("old"), &fast, &quality)
                            .with_fast_storage(NativeBuildPrecision::F16, graph(true))
                            .build(&cx),
                    )
                    .await
                    .unwrap();
                    let old_fast = std::fs::read(old.fast.vector_path()).unwrap();
                    let old_quality = std::fs::read(old.quality().unwrap().vector_path()).unwrap();
                    let changed = replacement(&quality);
                    let path = dir.path().join("failed");
                    let prepared = migration(&cx, &path, &old, &fast, &changed).unwrap();
                    changed.fault.store(fault, Ordering::SeqCst);
                    let result = prepared
                        .build_progressive_generation(
                            &cx,
                            CommitRange { low: 1, high: 5 },
                            IMAGE_BUDGET,
                        )
                        .await;
                    assert!(result.is_err());
                    if fault == CANCEL {
                        assert!(matches!(result, Err(SearchError::Cancelled { .. })));
                        cx.set_cancel_requested(false);
                    }
                    if fault == IDENTITY_DRIFT {
                        assert!(!path.exists());
                        assert_eq!(changed.submitted.load(Ordering::SeqCst), 0);
                    }
                    assert_eq!(fast.submitted.load(Ordering::SeqCst), 5);
                    assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
                    assert_eq!(std::fs::read(old.fast.vector_path()).unwrap(), old_fast);
                    assert_eq!(
                        std::fs::read(old.quality().unwrap().vector_path()).unwrap(),
                        old_quality
                    );
                    assert_eq!(
                        old.fast().search(&cx, "vertical", 1).await.unwrap()[0].doc_id,
                        "b"
                    );
                    assert_eq!(
                        old.quality()
                            .unwrap()
                            .search(&cx, "vertical", 1)
                            .await
                            .unwrap()[0]
                            .doc_id,
                        "b"
                    );
                }
            });
        }
    }

    mod sharded_updates {
        use super::*;

        fn assert_same_shards(
            actual: &NativeBuiltShardedIndex,
            expected: &NativeBuiltShardedIndex,
        ) {
            assert_eq!(actual.document_count(), expected.document_count());
            assert_eq!(actual.partitions().len(), expected.partitions().len());
            for (actual, expected) in actual.partitions().iter().zip(expected.partitions()) {
                assert_same_tier(actual.fast(), expected.fast());
                match (actual.quality(), expected.quality()) {
                    (Some(actual), Some(expected)) => assert_same_tier(actual, expected),
                    (None, None) => {}
                    _ => panic!("required tier topology changed"),
                }
                for (actual, expected) in actual.documents().iter().zip(expected.documents()) {
                    assert_eq!(actual.id, expected.id);
                    assert_eq!(actual.content, expected.content);
                    assert_eq!(actual.title, expected.title);
                    assert_eq!(actual.metadata, expected.metadata);
                }
            }
        }

        #[test]
        fn repartitioned_updates_match_cold_builds_and_embed_only_three_changed_bodies() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for ann in [false, true] {
                    for precision in [NativeBuildPrecision::F16, NativeBuildPrecision::F32] {
                        let dir = tempfile::tempdir().unwrap();
                        let fast = Provider::new("fast", 2);
                        let quality = Provider::new("quality", 3);
                        let old = builder(&dir.path().join("old"), &fast, &quality)
                            .with_fast_storage(precision, graph(ann))
                            .with_quality_storage(NativeBuildPrecision::F32, graph(!ann))
                            .unwrap()
                            .build_sharded(&cx, 2)
                            .await
                            .unwrap();
                        assert_eq!(old.partitions().len(), 3);
                        let old_bytes: Vec<_> = old
                            .partitions()
                            .iter()
                            .map(|part| {
                                (
                                    std::fs::read(part.fast().vector_path()).unwrap(),
                                    std::fs::read(part.quality().unwrap().vector_path()).unwrap(),
                                )
                            })
                            .collect();
                        let revised = IndexableDocument::new("a", "fractional")
                            .with_title("new title")
                            .with_metadata("revision", "2");
                        let next = old
                            .begin_update(&cx, dir.path().join("next"), generation(2))
                            .unwrap()
                            .delete_document("c")
                            .upsert_documents([
                                revised.clone(),
                                IndexableDocument::new("aa", "diagonal"),
                                IndexableDocument::new("b", "horizontal"),
                                IndexableDocument::new("d", "vertical"),
                            ])
                            .with_batch_size(2)
                            .unwrap()
                            .build_sharded(&cx, 3)
                            .await
                            .unwrap();
                        assert_eq!(next.partitions().len(), 2);
                        assert_eq!(fast.submitted.load(Ordering::SeqCst), 8);
                        assert_eq!(quality.submitted.load(Ordering::SeqCst), 8);
                        // This unchanged row crossed a partition boundary. A
                        // shard-local reuse lookup would unnecessarily embed it.
                        assert!(old.partitions()[2].document("z-fast").is_some());
                        assert!(next.partitions()[1].document("z-fast").is_some());
                        let cold = NativeIndexBuilder::new(
                            dir.path().join("cold"),
                            generation(2),
                            fast.clone(),
                        )
                        .unwrap()
                        .with_quality_embedder(quality.clone())
                        .unwrap()
                        .with_fast_storage(precision, graph(ann))
                        .with_quality_storage(NativeBuildPrecision::F32, graph(!ann))
                        .unwrap()
                        .add_documents([
                            revised,
                            IndexableDocument::new("aa", "diagonal"),
                            IndexableDocument::new("b", "horizontal"),
                            IndexableDocument::new("d", "vertical"),
                            IndexableDocument::new("x", "diagonal"),
                            IndexableDocument::new("z-fast", "horizontal"),
                        ])
                        .build_sharded(&cx, 3)
                        .await
                        .unwrap();
                        assert_same_shards(&next, &cold);
                        assert_eq!(
                            old.search_fast(&cx, "vertical", 1).await.unwrap()[0].doc_id,
                            "b"
                        );
                        assert_eq!(
                            next.search_fast(&cx, "vertical", 1).await.unwrap()[0].doc_id,
                            "d"
                        );
                        assert_eq!(
                            next.search_quality(&cx, "vertical", 1).await.unwrap()[0].doc_id,
                            "d"
                        );
                        for (part, (fast_bytes, quality_bytes)) in
                            old.partitions().iter().zip(old_bytes)
                        {
                            assert_eq!(
                                std::fs::read(part.fast().vector_path()).unwrap(),
                                fast_bytes
                            );
                            assert_eq!(
                                std::fs::read(part.quality().unwrap().vector_path()).unwrap(),
                                quality_bytes
                            );
                        }
                    }
                }
            });
        }

        #[test]
        fn repartitioning_and_coalescing_keep_owned_sources_after_old_paths_disappear() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let dir = tempfile::tempdir().unwrap();
                let fast = Provider::new("fast", 2);
                let quality = Provider::new("quality", 3);
                let path = dir.path().join("old");
                let old = builder(&path, &fast, &quality)
                    .build_sharded(&cx, 1)
                    .await
                    .unwrap();
                let pending = old
                    .begin_update(&cx, dir.path().join("next"), generation(2))
                    .unwrap();
                std::fs::rename(&path, dir.path().join("archived")).unwrap();
                drop(old);
                fast.fault.store(FAIL, Ordering::SeqCst);
                quality.fault.store(FAIL, Ordering::SeqCst);
                let next = pending.build_sharded(&cx, 4).await.unwrap();
                assert_eq!(next.partitions().len(), 2);
                let single = next
                    .begin_update(&cx, dir.path().join("single"), generation(3))
                    .unwrap()
                    .build(&cx)
                    .await
                    .unwrap();
                let split = single
                    .begin_update(&cx, dir.path().join("split"), generation(4))
                    .unwrap()
                    .build_sharded(&cx, 2)
                    .await
                    .unwrap();
                assert_eq!(split.partitions().len(), 3);
                assert_eq!(split.document_count(), 5);
                assert_eq!(fast.submitted.load(Ordering::SeqCst), 5);
                assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
                assert_eq!(
                    split.search_quality(&cx, "vertical", 1).await.unwrap()[0].doc_id,
                    "b"
                );
            });
        }

        #[cfg(any(target_os = "linux", target_os = "macos"))]
        #[test]
        fn metadata_edits_delete_all_and_restart_retain_exact_empty_topology() {
            use crate::native_ann::builder::sharded::NativeShardedReopenLimits;

            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let dir = tempfile::tempdir().unwrap();
                let fast = Provider::new("fast", 2);
                let quality = Provider::new("quality", 3);
                let old = builder(&dir.path().join("old"), &fast, &quality)
                    .build_sharded(&cx, 2)
                    .await
                    .unwrap();
                fast.fault.store(FAIL, Ordering::SeqCst);
                quality.fault.store(FAIL, Ordering::SeqCst);
                let next = old
                    .begin_update(&cx, dir.path().join("metadata"), generation(2))
                    .unwrap()
                    .upsert_document(
                        IndexableDocument::new("b", "vertical").with_title("changed title"),
                    )
                    .upsert_document(IndexableDocument::new("temporary", "never embedded"))
                    .delete_document("temporary")
                    .delete_document("absent")
                    .build_sharded(&cx, 1)
                    .await
                    .unwrap();
                assert_eq!(
                    next.document("b").unwrap().title.as_deref(),
                    Some("changed title")
                );
                let mut update = next
                    .begin_update(&cx, dir.path().join("empty"), generation(3))
                    .unwrap();
                for document in documents() {
                    update = update.delete_document(document.id);
                }
                let empty = update.build_sharded(&cx, 2).await.unwrap();
                assert_eq!(empty.document_count(), 0);
                assert_eq!(empty.partitions().len(), 1);
                assert!(empty.quality().is_some());
                let receipt = empty.seal_for_reopen(&cx).unwrap();
                let reopened = NativeBuiltShardedIndex::open_selected(
                    &cx,
                    empty.directory(),
                    &receipt,
                    fast.clone(),
                    Some(quality.clone()),
                    NativeShardedReopenLimits::default(),
                )
                .unwrap();
                assert_eq!(reopened.document_count(), 0);
                assert!(
                    reopened
                        .search_quality(&cx, "vertical", 10)
                        .await
                        .unwrap()
                        .is_empty()
                );
                assert_eq!(fast.submitted.load(Ordering::SeqCst), 5);
                assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
                fast.fault.store(OK, Ordering::SeqCst);
                quality.fault.store(OK, Ordering::SeqCst);
                let restored = reopened
                    .begin_update(&cx, dir.path().join("restored"), generation(4))
                    .unwrap()
                    .upsert_document(IndexableDocument::new("new", "vertical"))
                    .build_sharded(&cx, 2)
                    .await
                    .unwrap();
                assert_eq!(restored.document_count(), 1);
                assert_eq!(fast.submitted.load(Ordering::SeqCst), 6);
                assert_eq!(quality.submitted.load(Ordering::SeqCst), 6);
                assert!(old.document("b").is_some());
            });
        }

        #[test]
        fn changed_precision_reembeds_only_its_tier_even_across_new_boundaries() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let dir = tempfile::tempdir().unwrap();
                let fast = Provider::new("fast", 2);
                let quality = Provider::new("quality", 3);
                let old = builder(&dir.path().join("old"), &fast, &quality)
                    .with_fast_storage(NativeBuildPrecision::F16, graph(false))
                    .build_sharded(&cx, 2)
                    .await
                    .unwrap();
                quality.fault.store(FAIL, Ordering::SeqCst);
                let next = old
                    .begin_update(&cx, dir.path().join("next"), generation(2))
                    .unwrap()
                    .with_fast_storage(NativeBuildPrecision::F32, graph(true))
                    .build_sharded(&cx, 1)
                    .await
                    .unwrap();
                assert_eq!(fast.submitted.load(Ordering::SeqCst), 10);
                assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
                for part in next.partitions() {
                    assert!(part.fast().graph_path().is_some());
                    assert!(part.quality().unwrap().graph_path().is_none());
                }
                let owner = &next.partitions()[0].fast.index.owner;
                assert_eq!(owner.doc_id_at(0).unwrap(), "a");
                assert_eq!(owner.vector_at_f32(0).unwrap(), fast.values("fractional"));
            });
        }

        #[test]
        fn invalid_sources_shard_limits_and_late_provider_drift_precede_creation() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let dir = tempfile::tempdir().unwrap();
                let fast = Provider::new("fast", 2);
                let quality = Provider::new("quality", 3);
                let old = builder(&dir.path().join("old"), &fast, &quality)
                    .build_sharded(&cx, 2)
                    .await
                    .unwrap();
                let path = dir.path().join("rejected");
                assert!(old.begin_update(&cx, &path, generation(1)).is_err());
                for fault in ["size", "empty", "count", "bytes", "drift"] {
                    let update = old.begin_update(&cx, &path, generation(2)).unwrap();
                    let result = match fault {
                        "size" => update.build_sharded(&cx, 0).await,
                        "empty" => update.delete_document("").build_sharded(&cx, 2).await,
                        "count" => {
                            update
                                .upsert_documents((0..1025).map(|i| {
                                    IndexableDocument::new(format!("new-{i:04}"), "vertical")
                                }))
                                .build_sharded(&cx, 1)
                                .await
                        }
                        "bytes" => {
                            update
                                .with_max_batch_input_bytes(3)
                                .unwrap()
                                .build_sharded(&cx, 2)
                                .await
                        }
                        _ => {
                            quality.fault.store(IDENTITY_DRIFT, Ordering::SeqCst);
                            update.build_sharded(&cx, 2).await
                        }
                    };
                    quality.fault.store(OK, Ordering::SeqCst);
                    assert!(result.is_err(), "{fault}");
                    assert!(!path.exists(), "{fault}");
                    assert_eq!(fast.submitted.load(Ordering::SeqCst), 5);
                    assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
                }
                cx.set_cancel_requested(true);
                let result = old.begin_update(&cx, &path, generation(2));
                cx.set_cancel_requested(false);
                assert!(matches!(result, Err(SearchError::Cancelled { .. })));
                assert!(!path.exists());
            });
        }

        #[test]
        fn last_partition_failure_or_drop_never_returns_a_successful_prefix() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for fault in [FAIL, FOREIGN, CANCEL, PENDING] {
                    let dir = tempfile::tempdir().unwrap();
                    let fast = Provider::new("fast", 2);
                    let quality = Provider::new("quality", 3);
                    let old = builder(&dir.path().join("old"), &fast, &quality)
                        .build_sharded(&cx, 2)
                        .await
                        .unwrap();
                    let path = dir.path().join("failed");
                    let update = old
                        .begin_update(&cx, &path, generation(2))
                        .unwrap()
                        .upsert_document(IndexableDocument::new("zz-new", "vertical"));
                    quality.fault.store(fault, Ordering::SeqCst);
                    if fault == PENDING {
                        let drops = quality.drops.load(Ordering::SeqCst);
                        let mut pending = Box::pin(update.build_sharded(&cx, 2));
                        assert!(
                            pending
                                .as_mut()
                                .poll(&mut Context::from_waker(Waker::noop()))
                                .is_pending()
                        );
                        assert_eq!(
                            old.search_quality(&cx, "vertical", 1).await.unwrap()[0].doc_id,
                            "b"
                        );
                        drop(pending);
                        assert_eq!(quality.drops.load(Ordering::SeqCst), drops + 1);
                    } else {
                        let result = update.build_sharded(&cx, 2).await;
                        assert!(result.is_err());
                        if fault == CANCEL {
                            assert!(matches!(result, Err(SearchError::Cancelled { .. })));
                            cx.set_cancel_requested(false);
                        }
                    }
                    // Earlier children really completed; no test that fails
                    // immediately can discharge the partial-inventory contract.
                    assert!(path.join("shard-000000/fast.fsvi").is_file());
                    assert!(path.join("shard-000001/quality.fsvi").is_file());
                    assert!(!path.join("native.sharded.json").exists());
                    assert!(!path.join("native.sharded-hybrid.json").exists());
                    assert_eq!(old.document_count(), 5);
                    assert!(old.document("zz-new").is_none());
                    quality.fault.store(OK, Ordering::SeqCst);
                    let retry = old
                        .begin_update(&cx, dir.path().join("retry"), generation(2))
                        .unwrap()
                        .upsert_document(IndexableDocument::new("zz-new", "vertical"))
                        .build_sharded(&cx, 2)
                        .await
                        .unwrap();
                    assert_eq!(retry.document_count(), 6);
                }
            });
        }

        #[test]
        fn every_retained_partition_policy_is_checked_before_staging() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for fault in [
                    "precision",
                    "graph",
                    "producer",
                    "generation",
                    "order",
                    "drift",
                ] {
                    let dir = tempfile::tempdir().unwrap();
                    let fast = Provider::new("fast", 2);
                    let other =
                        Provider::new(if fault == "producer" { "other" } else { "fast" }, 2);
                    let first =
                        NativeIndexBuilder::new(dir.path().join("a"), generation(1), fast.clone())
                            .unwrap()
                            .add_document(IndexableDocument::new("a", "horizontal"))
                            .build(&cx)
                            .await
                            .unwrap();
                    let mut last = NativeIndexBuilder::new(
                        dir.path().join("b"),
                        generation(if fault == "generation" { 2 } else { 1 }),
                        other.clone(),
                    )
                    .unwrap()
                    .add_document(IndexableDocument::new(
                        if fault == "order" { "a" } else { "b" },
                        "vertical",
                    ));
                    if fault == "precision" {
                        last = last.with_fast_storage(NativeBuildPrecision::F16, graph(false));
                    }
                    if fault == "graph" {
                        last = last.with_fast_storage(NativeBuildPrecision::F32, graph(true));
                    }
                    let last = last.build(&cx).await.unwrap();
                    if fault == "drift" {
                        other.fault.store(IDENTITY_DRIFT, Ordering::SeqCst);
                    }
                    let path = dir.path().join("next");
                    // Private constructor receives individually valid, retained
                    // owners. It must not trust only the first partition.
                    assert!(
                        NativeIndexUpdate::from_partitions(
                            &cx,
                            &path,
                            generation(3),
                            &[first, last],
                        )
                        .is_err(),
                        "{fault}"
                    );
                    assert!(!path.exists());
                    assert_eq!(fast.submitted.load(Ordering::SeqCst), 1);
                    assert_eq!(other.submitted.load(Ordering::SeqCst), 1);
                }
            });
        }

        #[cfg(all(feature = "quill", any(target_os = "linux", target_os = "macos")))]
        #[test]
        fn hybrid_updates_replace_global_lexical_membership_and_preserve_old_phases() {
            use crate::native_ann::NativeShardedSearchPhase;
            use crate::native_ann::builder::sharded::NativeShardedHybridReopenLimits;

            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let dir = tempfile::tempdir().unwrap();
                let fast = Provider::new("fast", 2);
                let quality = Provider::new("quality", 3);
                let old =
                    NativeIndexBuilder::new(dir.path().join("old"), generation(1), fast.clone())
                        .unwrap()
                        .with_quality_embedder(quality.clone())
                        .unwrap()
                        .with_fast_storage(NativeBuildPrecision::F16, graph(true))
                        .add_documents([
                            IndexableDocument::new("a", "retained common"),
                            IndexableDocument::new("b", "obsolete common")
                                .with_metadata("revision", "old"),
                            IndexableDocument::new("c", "removed common"),
                        ])
                        .build_sharded_hybrid(&cx, 2)
                        .await
                        .unwrap();
                let mut phases = old.progressive(&cx, "obsolete", 1).unwrap();
                let Some(NativeShardedSearchPhase::Initial { results, .. }) =
                    phases.next_phase().await.unwrap()
                else {
                    panic!("initial phase missing");
                };
                assert_eq!(results[0].result.doc_id, "b");
                let path = dir.path().join("next");
                let next = old
                    .begin_update(&cx, &path, generation(2))
                    .unwrap()
                    .delete_document("c")
                    .upsert_documents([
                        IndexableDocument::new("b", "replacement common")
                            .with_metadata("revision", "new"),
                        IndexableDocument::new("d", "arrival common"),
                    ])
                    .build_sharded_hybrid(&cx, 1)
                    .await
                    .unwrap();
                assert_eq!(next.vectors().partitions().len(), 3);
                assert_eq!(fast.submitted.load(Ordering::SeqCst), 5);
                assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
                for query in ["obsolete", "removed"] {
                    assert!(
                        next.lexical()
                            .search(&cx, query, 10)
                            .await
                            .unwrap()
                            .is_empty()
                    );
                }
                assert_eq!(
                    next.lexical().search(&cx, "arrival", 10).await.unwrap()[0].doc_id,
                    "d"
                );
                assert_eq!(
                    next.vectors().document("b").unwrap().metadata["revision"],
                    "new"
                );
                let Some(NativeShardedSearchPhase::Refined { results, .. }) =
                    phases.next_phase().await.unwrap()
                else {
                    panic!("old query lost its required quality phase");
                };
                assert_eq!(results[0].result.doc_id, "b");
                assert!(results[0].result.lexical_score.is_some());
                assert_eq!(
                    old.vectors().document("b").unwrap().metadata["revision"],
                    "old"
                );
                assert_eq!(
                    old.lexical().search(&cx, "removed", 10).await.unwrap()[0].doc_id,
                    "c"
                );
                let receipt = next.seal_for_reopen(&cx).unwrap();
                drop(next);
                let reopened = NativeBuiltShardedHybridIndex::open_selected(
                    &cx,
                    &path,
                    &receipt,
                    fast.clone(),
                    Some(quality.clone()),
                    NativeShardedHybridReopenLimits::default(),
                )
                .await
                .unwrap();
                assert!(reopened.vectors().document("c").is_none());
                for hit in reopened
                    .search_refined(&cx, "replacement", 10)
                    .await
                    .unwrap()
                {
                    assert!(hit.result.index.is_none());
                    if let Some(row) = hit.fast_row {
                        assert_eq!(
                            reopened.vectors().partitions()[row.shard]
                                .fast
                                .index
                                .owner
                                .doc_id_at(usize::try_from(row.physical_row).unwrap())
                                .unwrap(),
                            hit.result.doc_id
                        );
                    }
                    if let Some(row) = hit.quality_row {
                        assert_eq!(
                            reopened.vectors().partitions()[row.shard]
                                .quality
                                .as_ref()
                                .unwrap()
                                .index
                                .owner
                                .doc_id_at(usize::try_from(row.physical_row).unwrap())
                                .unwrap(),
                            hit.result.doc_id
                        );
                    }
                }
                assert_eq!(fast.submitted.load(Ordering::SeqCst), 5);
                assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
            });
        }
    }
}
