//! Document-to-native retrieval construction over one retained source cohort.
//!
//! Unlike the legacy convenience builder, this path writes identity-complete
//! FSVI v2, admits those exact bytes, and constructs the native retrieval arms.
//! It never discovers, downloads, or substitutes a model. The supplied content
//! is already prepared input: no hidden canonicalization, title concatenation,
//! normalization, or dimension truncation is performed by this builder.
//!
//! A build exclusively creates a NEW directory under an existing trusted parent.
//! It never overwrites a live generation, updates CURRENT, or selects a generation
//! for other processes. A failure or dropped future may leave an incomplete
//! directory for diagnostics; directory existence is NOT a completion receipt.
//! Only a successful returned handle represents a complete source/vector cohort.
//! Explicit `NativeBuiltIndex::seal_for_reopen` persists that cohort under a
//! caller-retained receipt; reopening never discovers or selects a generation.
//! Filesystem and graph work is synchronous on the polling execution lane; use
//! the caller's blocking lane for large builds. No task or runtime is spawned.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1, QuantizationFormat,
};
use frankensearch_core::traits::IdentityBoundEmbedding;
use frankensearch_index::native_hnsw::{HnswParams, NativeHnswGenerationReceiptV2};
use frankensearch_index::{
    FsviV2IdentityBinding, ValidatedFsviBytes, VectorIndex, VectorIndexWriter,
};

use super::{NativeAnnIndex, checkpoint, invalid};
use crate::{Cx, Embedder, IndexableDocument, SearchResult, VectorHit};

mod snapshot;
pub use snapshot::NativeReopenLimits;

mod update;
pub use update::NativeIndexUpdate;
use update::ReuseSource;

#[cfg(feature = "quill")]
mod hybrid;
#[cfg(feature = "quill")]
pub use hybrid::{NativeBuiltHybridIndex, NativeHybridReopenLimits};

/// Live whole-cohort serving and stale-safe installation of native hybrid updates.
#[cfg(feature = "quill")]
pub mod live;

/// Caller-clock deadlines for complete native hybrid queries.
#[cfg(feature = "quill")]
pub mod deadline;

/// Persisted vector precision; neither option changes the producing model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeBuildPrecision {
    /// Preserve the model's f32 outputs.
    #[default]
    F32,
    /// Use the existing FSVI f16 writer and its validation/rounding rules.
    F16,
}

/// Explicit construction policy, independent of semantic versus control models.
#[derive(Debug, Clone, Copy, Default)]
pub enum NativeBuildRetrieval {
    /// Do not allocate or persist a graph.
    #[default]
    Exact,
    /// Build and persist a native graph over the admitted vector owner.
    Hnsw {
        /// Existing native engine construction/search parameters.
        params: HnswParams,
        /// Deterministic graph construction seed.
        seed: u64,
    },
}

struct TierPlan {
    embedder: Arc<dyn Embedder>,
    identity: EmbeddingIdentityBundleV1,
    precision: NativeBuildPrecision,
    retrieval: NativeBuildRetrieval,
}

impl TierPlan {
    fn new(embedder: Arc<dyn Embedder>) -> SearchResult<Self> {
        let identity = embedder.identity()?.clone();
        identity.validate()?;
        if usize::try_from(identity.space.dimension).ok() != Some(embedder.dimension()) {
            return Err(invalid(
                "builder.dimension",
                "mismatch",
                "provider dimension disagrees with its identity",
            ));
        }
        Ok(Self {
            embedder,
            identity,
            precision: NativeBuildPrecision::F32,
            retrieval: NativeBuildRetrieval::Exact,
        })
    }

    fn binding(
        &self,
        generation: &ArtifactGenerationIdentityV1,
    ) -> SearchResult<FsviV2IdentityBinding> {
        self.admit(self.embedder.identity()?)?;
        let mut identity = self.identity.clone();
        "fsvi-v2".clone_into(&mut identity.storage.format);
        identity.storage.quantization = match self.precision {
            NativeBuildPrecision::F32 => QuantizationFormat::F32,
            NativeBuildPrecision::F16 => QuantizationFormat::F16,
        };
        "little-endian".clone_into(&mut identity.storage.endianness);
        FsviV2IdentityBinding::new(*generation, identity.freeze()?)
            .map_err(|error| invalid("builder.binding", "rejected", &error.to_string()))
    }

    fn admit(&self, identity: &EmbeddingIdentityBundleV1) -> SearchResult<()> {
        identity.validate()?;
        // Compare the provider's complete output identity, before changing only
        // the persisted storage encoding in the FSVI binding. A copied model
        // name, dimension, or golden-vector certificate is not sufficient.
        if identity.fingerprint() != self.identity.fingerprint() {
            return Err(invalid(
                "builder.producer",
                "changed",
                "batch provider changed its frozen identity",
            ));
        }
        Ok(())
    }

    async fn write_batch(
        &self,
        cx: &Cx,
        writer: &mut VectorIndexWriter,
        documents: &[IndexableDocument],
    ) -> SearchResult<()> {
        checkpoint(cx, "native_ann.builder.before_batch")?;
        self.admit(self.embedder.identity()?)?;
        let texts: Vec<_> = documents.iter().map(|doc| doc.content.as_str()).collect();
        let response = self.embedder.embed_batch_bound(cx, &texts).await;
        checkpoint(cx, "native_ann.builder.after_batch")?;
        let response = response?;
        if response.len() != documents.len() {
            return Err(invalid(
                "builder.batch",
                "cardinality",
                "one bound embedding is required per input document",
            ));
        }
        // Validate the entire returned batch before writing ANY row from it.
        for bound in &response {
            checkpoint(cx, "native_ann.builder.admit_output")?;
            self.admit_output(bound)?;
        }
        for (document, bound) in documents.iter().zip(response) {
            checkpoint(cx, "native_ann.builder.write_row")?;
            writer.write_record(document.id.as_str(), &bound.values)?;
        }
        Ok(())
    }

    fn admit_output(&self, bound: &IdentityBoundEmbedding) -> SearchResult<()> {
        bound.validate()?;
        self.admit(&bound.identity)?;
        if bound.values.len() != self.embedder.dimension()
            || bound.values.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "builder.output",
                "invalid",
                "embedding dimensions and finite values must match the frozen producer",
            ));
        }
        Ok(())
    }
}

/// Build one fast and optional quality tier from exactly the same source cohort.
///
/// Models and their identities are retained from construction through queries.
/// Each batch is embedded once per tier and written before advancing: the builder
/// does not retain an additional corpus-sized matrix of f32 model outputs.
/// Source documents remain owned because later reranking must use the same text.
/// The v2 writer and final admitted owners have their own storage requirements.
/// Batch outputs follow the core Embedder contract's input order; this builder
/// cannot authenticate an implementor that lies about the text it embedded.
pub struct NativeIndexBuilder {
    directory: PathBuf,
    generation: ArtifactGenerationIdentityV1,
    fast: TierPlan,
    quality: Option<TierPlan>,
    batch_size: usize,
    documents: Vec<IndexableDocument>,
    reuse: Option<ReuseSource>,
}

impl NativeIndexBuilder {
    /// Freeze a destination path and fast provider identity without creating files.
    ///
    /// # Errors
    /// Returns invalid provider identity/dimension or current-directory errors.
    pub fn new(
        directory: impl AsRef<Path>,
        generation: ArtifactGenerationIdentityV1,
        fast: Arc<dyn Embedder>,
    ) -> SearchResult<Self> {
        let directory = directory.as_ref();
        let directory = if directory.is_absolute() {
            directory.to_path_buf()
        } else {
            std::env::current_dir()?.join(directory)
        };
        let fast = TierPlan::new(fast)?;
        fast.binding(&generation)?;
        Ok(Self {
            directory,
            generation,
            fast,
            quality: None,
            batch_size: 64,
            documents: Vec::new(),
            reuse: None,
        })
    }

    /// Add an independently dimensioned quality model; it may not change document
    /// identity semantics or mix hash-control with semantic provenance.
    ///
    /// # Errors
    /// Returns identity, dimension, document-contract or model-kind disagreement.
    pub fn with_quality_embedder(mut self, embedder: Arc<dyn Embedder>) -> SearchResult<Self> {
        let quality = TierPlan::new(embedder)?;
        if self.fast.identity.input.doc_id_semantics != quality.identity.input.doc_id_semantics
            || self.fast.identity.space.kind != quality.identity.space.kind
        {
            return Err(invalid(
                "builder.quality",
                "incompatible",
                "both tiers require one document-ID contract and semantic/control kind",
            ));
        }
        self.quality = Some(quality);
        Ok(self)
    }

    /// Set the bounded inference batch size.
    ///
    /// # Errors
    /// Rejects zero, before filesystem or model work.
    pub fn with_batch_size(mut self, batch_size: usize) -> SearchResult<Self> {
        if batch_size == 0 {
            return Err(invalid(
                "builder.batch_size",
                "0",
                "batch size must be positive",
            ));
        }
        self.batch_size = batch_size;
        Ok(self)
    }

    /// Select fast storage and exact/native retrieval without changing the model.
    #[must_use]
    pub fn with_fast_storage(
        mut self,
        precision: NativeBuildPrecision,
        retrieval: NativeBuildRetrieval,
    ) -> Self {
        self.fast.precision = precision;
        self.fast.retrieval = retrieval;
        self
    }

    /// Select quality storage/retrieval after configuring the quality model.
    ///
    /// # Errors
    /// Rejects configuration of an absent quality tier.
    pub fn with_quality_storage(
        mut self,
        precision: NativeBuildPrecision,
        retrieval: NativeBuildRetrieval,
    ) -> SearchResult<Self> {
        let quality = self.quality.as_mut().ok_or_else(|| {
            invalid(
                "builder.quality",
                "absent",
                "configure the quality provider first",
            )
        })?;
        quality.precision = precision;
        quality.retrieval = retrieval;
        Ok(self)
    }

    /// Add already-prepared source content. Duplicate or empty IDs are refused by
    /// build before files or inference; metadata and titles remain with the source.
    #[must_use]
    pub fn add_document(mut self, document: IndexableDocument) -> Self {
        self.documents.push(document);
        self
    }

    /// Add an owned cohort; input order does not affect physical document order.
    #[must_use]
    pub fn add_documents(mut self, documents: impl IntoIterator<Item = IndexableDocument>) -> Self {
        self.documents.extend(documents);
        self
    }

    /// Build a complete retained native generation, or return an error.
    ///
    /// All configured tiers are required. Provider/cancellation/admission/graph
    /// failures never remove a document, drop quality, or return a partial cohort.
    /// Empty cohorts produce explicitly empty admitted indexes without inference.
    /// There is no filesystem completion marker or discovery fallback: consumers
    /// must retain this successful handle, retain its explicit `seal_for_reopen`
    /// receipt, or publish its exact witnesses through their own trusted selector.
    /// Failed directories are not
    /// automatically deleted and are never overwritten by a subsequent attempt.
    ///
    /// # Errors
    /// Propagates source, identity, filesystem, provider, graph and cancellation
    /// errors. An already-existing destination is always refused.
    pub async fn build(mut self, cx: &Cx) -> SearchResult<NativeBuiltIndex> {
        checkpoint(cx, "native_ann.builder.start")?;
        self.documents.sort_by(|a, b| a.id.cmp(&b.id));
        for (position, doc) in self.documents.iter().enumerate() {
            checkpoint(cx, "native_ann.builder.source")?;
            if doc.id.is_empty() || (position > 0 && self.documents[position - 1].id == doc.id) {
                return Err(invalid(
                    "builder.documents",
                    "empty-or-duplicate-id",
                    "source document IDs must be nonempty and unique",
                ));
            }
        }
        let fast_binding = self.fast.binding(&self.generation)?;
        let quality_binding = self
            .quality
            .as_ref()
            .map(|tier| tier.binding(&self.generation))
            .transpose()?;
        for tier in std::iter::once(&self.fast).chain(self.quality.iter()) {
            if let NativeBuildRetrieval::Hnsw { params, .. } = tier.retrieval {
                params.validate()?;
            }
        }
        checkpoint(cx, "native_ann.builder.create")?;
        // create_dir, not create_dir_all: an existing leaf (including symlinks)
        // cannot be adopted and a live directory can never be truncated.
        std::fs::create_dir(&self.directory)?;
        let directory = std::fs::canonicalize(&self.directory)?;
        let fast_path = directory.join("fast.fsvi");
        let quality_path = directory.join("quality.fsvi");
        let mut fast_writer = VectorIndex::create_v2(&fast_path, fast_binding.clone())?;
        let mut quality_writer = quality_binding
            .as_ref()
            .map(|binding| VectorIndex::create_v2(&quality_path, binding.clone()))
            .transpose()?;
        for batch in self.documents.chunks(self.batch_size) {
            self.fast
                .write_batch_reusing(
                    cx,
                    &mut fast_writer,
                    batch,
                    self.reuse.as_ref().map(|source| (source, &source.fast)),
                )
                .await?;
            if let (Some(tier), Some(writer)) = (&self.quality, &mut quality_writer) {
                let reuse = self.reuse.as_ref().and_then(|source| {
                    source.quality.as_ref().map(|quality| (source, quality))
                });
                tier.write_batch_reusing(cx, writer, batch, reuse).await?;
            }
        }
        checkpoint(cx, "native_ann.builder.finish_vectors")?;
        fast_writer.finish()?;
        if let Some(writer) = quality_writer {
            writer.finish()?;
        }
        let fast = finish_tier(cx, self.fast, fast_binding, fast_path, &self.documents)?;
        let quality = match (self.quality, quality_binding) {
            (Some(tier), Some(binding)) => Some(finish_tier(
                cx,
                tier,
                binding,
                quality_path,
                &self.documents,
            )?),
            (None, None) => None,
            _ => {
                return Err(invalid(
                    "builder.quality",
                    "inconsistent",
                    "quality plan and binding must travel together",
                ));
            }
        };
        checkpoint(cx, "native_ann.builder.complete")?;
        Ok(NativeBuiltIndex {
            directory,
            documents: self.documents.into(),
            fast,
            quality,
        })
    }
}

/// A successfully built native tier with its retained query provider and exact
/// reopen specification. Paths are diagnostics/reopen inputs, never query inputs.
pub struct NativeBuiltTier {
    index: NativeAnnIndex,
    embedder: Arc<dyn Embedder>,
    producer_identity: EmbeddingIdentityBundleV1,
    precision: NativeBuildPrecision,
    binding: FsviV2IdentityBinding,
    vector_path: PathBuf,
    graph_path: Option<PathBuf>,
    graph_receipt: Option<NativeHnswGenerationReceiptV2>,
}

impl NativeBuiltTier {
    /// The admitted native/exact index; no vector pathname is reopened by search.
    #[must_use]
    pub const fn index(&self) -> &NativeAnnIndex {
        &self.index
    }
    /// Provider used to produce every stored row.
    #[must_use]
    pub fn embedder(&self) -> &dyn Embedder {
        self.embedder.as_ref()
    }
    /// Exact persisted v2 identity binding for a caller-owned reopen specification.
    #[must_use]
    pub const fn binding(&self) -> &FsviV2IdentityBinding {
        &self.binding
    }
    /// Newly created vector artifact.
    #[must_use]
    pub fn vector_path(&self) -> &Path {
        &self.vector_path
    }
    /// Persisted native graph, absent for explicitly exact construction.
    #[must_use]
    pub fn graph_path(&self) -> Option<&Path> {
        self.graph_path.as_deref()
    }
    /// Query with the retained producing provider.
    ///
    /// # Errors
    /// Propagates native text-search admission, inference and cancellation errors.
    pub async fn search(&self, cx: &Cx, text: &str, k: usize) -> SearchResult<Vec<VectorHit>> {
        self.index
            .search_text(cx, self.embedder(), text, k, None)
            .await
    }
}

/// One build's complete source cohort and required vector tiers.
///
/// Source text, IDs, metadata and vector handles are immutable through this API.
/// Holding this object protects queries from later changes to the source files.
/// An explicitly selected source/vector seal can restore this cohort after
/// restart. It is not composite lexical/vector publication authority or an
/// attestation of arbitrary independently opened lexical readers.
pub struct NativeBuiltIndex {
    directory: PathBuf,
    documents: Arc<[IndexableDocument]>,
    fast: NativeBuiltTier,
    quality: Option<NativeBuiltTier>,
}

impl NativeBuiltIndex {
    /// Directory exclusively created by this successful build.
    #[must_use]
    pub fn directory(&self) -> &Path {
        &self.directory
    }
    /// Required fast tier.
    #[must_use]
    pub const fn fast(&self) -> &NativeBuiltTier {
        &self.fast
    }
    /// Quality tier, present only when explicitly configured and fully built.
    #[must_use]
    pub const fn quality(&self) -> Option<&NativeBuiltTier> {
        self.quality.as_ref()
    }
    /// Exact source cohort in canonical document-ID order.
    #[must_use]
    pub fn documents(&self) -> &[IndexableDocument] {
        &self.documents
    }
    /// Resolve the input document from the retained cohort, not current storage.
    #[must_use]
    pub fn document(&self, id: &str) -> Option<&IndexableDocument> {
        self.documents
            .binary_search_by(|doc| doc.id.as_str().cmp(id))
            .ok()
            .map(|position| &self.documents[position])
    }
}

fn finish_tier(
    cx: &Cx,
    plan: TierPlan,
    binding: FsviV2IdentityBinding,
    vector_path: PathBuf,
    documents: &[IndexableDocument],
) -> SearchResult<NativeBuiltTier> {
    checkpoint(cx, "native_ann.builder.admit_vectors")?;
    let bytes: Arc<[u8]> = std::fs::read(&vector_path)?.into();
    let owner = Arc::new(
        ValidatedFsviBytes::from_arc(bytes, &binding)
            .map_err(|error| invalid("builder.vector_admission", "rejected", &error.to_string()))?,
    );
    validate_source_membership(cx, &owner, documents)?;
    let (index, graph_path, graph_receipt) = match plan.retrieval {
        NativeBuildRetrieval::Exact => (NativeAnnIndex::exact(cx, owner)?, None, None),
        NativeBuildRetrieval::Hnsw { params, seed } => {
            let index = NativeAnnIndex::build(cx, owner, params, seed)?;
            let graph_path = vector_path.with_extension("fshnsw");
            let receipt = index.save(cx, &graph_path)?;
            (index, Some(graph_path), Some(receipt))
        }
    };
    index.admit_identity(plan.embedder.identity()?)?;
    Ok(NativeBuiltTier {
        index,
        embedder: plan.embedder,
        producer_identity: plan.identity,
        precision: plan.precision,
        binding,
        vector_path,
        graph_path,
        graph_receipt,
    })
}

/// Shared by initial build and selected reopen. `documents` has already been
/// admitted as strictly increasing by ID; physical FSVI order is independent.
fn validate_source_membership(
    cx: &Cx,
    owner: &ValidatedFsviBytes,
    documents: &[IndexableDocument],
) -> SearchResult<()> {
    if owner.record_count() != documents.len() || owner.live_count() != documents.len() {
        return Err(invalid(
            "builder.source_join",
            "cardinality",
            "admitted vector membership must equal the complete source cohort",
        ));
    }
    // FSVI sorts its physical rows by (document hash, document ID), NOT by
    // lexical ID order. Resolve each admitted row back to the canonical source
    // cohort, retaining its physical position for vector lookup. Cardinality
    // plus one-to-one membership proves equality without reordering the owner.
    let mut seen = vec![false; documents.len()];
    for physical in 0..owner.record_count() {
        checkpoint(cx, "native_ann.builder.source_join")?;
        let row = owner.row(physical)?;
        let position = documents
            .binary_search_by(|doc| doc.id.as_str().cmp(row.doc_id()))
            .map_err(|_| {
                invalid(
                    "builder.source_join",
                    "document-id",
                    "admitted vector row is absent from the source cohort",
                )
            })?;
        if !row.flags().is_live() || std::mem::replace(&mut seen[position], true) {
            return Err(invalid(
                "builder.source_join",
                "duplicate-or-deleted",
                "every source document must map to exactly one live vector row",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
