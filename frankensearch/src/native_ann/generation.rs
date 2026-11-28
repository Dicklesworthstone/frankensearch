//! Native retrieval over one authority-admitted composite generation.
//!
//! All engine readers consume the retained component images. No component is
//! rediscovered through a pathname, and a later authority activation cannot
//! change the bytes used by an already-admitted reader.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use asupersync::runtime::blocking_pool::BlockingPoolHandle;

use frankensearch_core::generation::{
    ActivationManifest, ArtifactGenerationIdentityV1, AuthorityRefV1, CanonicalDocsetV1,
    EmbedderTierTag, ExactComponentReceiptV1, ExactGenerationComponentsV1,
    GenerationAuthorityActionV1, GenerationComponentRole, GenerationManifest,
    GenerationRootSecurityProfileV1, InvariantKind, SourceCheckpointV1, resolve_authority_slots_v1,
    validate_manifest,
};
use frankensearch_core::{ParsedQuery, with_cancellation};
use frankensearch_index::ann_cohort::NativeAnnCohortV1;
use frankensearch_index::exact_component_adapters::{
    ann_component_receipt, vector_component_receipt,
};
use frankensearch_index::generation_root::GenerationRootObjectWitness;
use frankensearch_index::generation_root::generation_reader::{
    OpenedGenerationSnapshotV1, SnapshotHealthV1,
};
use frankensearch_index::native_hnsw::{
    HnswParams, NativeHnswGenerationReceiptV2, ValidatedNativeHnsw,
};
use frankensearch_index::vector_cohort::{ExactVectorCohortV1, VectorCohortCoverage};
use frankensearch_index::{FsviV2IdentityBinding, ValidatedFsviBytes};
use frankensearch_quill::{
    DEFAULT_SCHEMA, KeeperSnapshot, QuillConfig, QuillIndexError, QuillSnapshotIndex,
    SchemaDescriptor,
};
use sha2::Digest as _;

use super::builder::{
    NativeBuildRetrieval, NativeIndexUpdate, NativePartialQualityUpdate, NativeQualityCompletion,
};
use super::{
    NativeAnnIndex, NativeBackend, NativeProgressiveSearch, NativeRetrievalMode, checkpoint,
    invalid, lower_hex,
};
use crate::{
    Cx, Embedder, GenerationCandidateV1, IndexableDocument, LexicalRead, ScoreSource, ScoredResult,
    SearchResult,
};

#[cfg(test)]
mod tests;

mod source_recovery;
pub use source_recovery::{
    NativeGenerationQualityCompletionSource, NativeGenerationSource, NativeGenerationVectorSource,
};

// The selected metadata image authenticates the full owner-bound native receipt.
// Single-tier ANN components retain their raw graph image. Complete two-tier
// generations pack the independently receipted graphs into one ANN component.
pub(super) const NATIVE_ANN_RECEIPT_KEY: &str = "native_hnsw_receipt_v2";
pub(super) const NATIVE_QUALITY_ANN_RECEIPT_KEY: &str = "native_quality_hnsw_receipt_v2";
// Persist the coverage contract in authenticated metadata as well as vector
// framing. Source-only recovery must not mistake a partial owner for a full tier
// when its vector image is unavailable.
pub(super) const NATIVE_QUALITY_COVERAGE_KEY: &str = "native_quality_coverage_v1";
// The authenticated metadata must retain the actual subset, not only its size,
// so source-only repair can preserve coverage when the vector image is lost.
pub(super) const NATIVE_QUALITY_DOCUMENT_IDS_KEY: &str = "native_quality_document_ids_v1";

/// Decode the optional, canonical quality-membership witness.
///
/// Older partial generations may omit this witness and remain searchable from
/// their admitted vectors. They cannot authorize subset-preserving source-only
/// repair. A present witness is always checked, including on ordinary reads.
pub(super) fn declared_partial_quality_document_ids(
    cx: &Cx,
    metadata: &GenerationManifest,
) -> SearchResult<Option<Vec<String>>> {
    checkpoint(cx, "native_ann.generation.quality_membership")?;
    let Some(encoded) = metadata.metadata.get(NATIVE_QUALITY_DOCUMENT_IDS_KEY) else {
        return Ok(None);
    };
    if metadata
        .metadata
        .get(NATIVE_QUALITY_COVERAGE_KEY)
        .map(String::as_str)
        != Some("partial_quality")
        || metadata.vector_artifacts.len() != 2
        || metadata.vector_artifacts[0].embedder_tier != EmbedderTierTag::Fast
        || metadata.vector_artifacts[1].embedder_tier != EmbedderTierTag::Quality
    {
        return Err(refused("metadata", "quality-membership-contract-mismatch"));
    }
    let identifiers: Vec<String> = serde_json::from_str(encoded)
        .map_err(|_| refused("metadata", "invalid-quality-membership"))?;
    let count = u64::try_from(identifiers.len())
        .map_err(|_| refused("metadata", "invalid-quality-membership"))?;
    if count == 0
        || count >= metadata.total_documents
        || count > metadata.vector_artifacts[1].vector_count
    {
        return Err(refused("metadata", "invalid-quality-membership"));
    }
    let mut previous: Option<&str> = None;
    for identifier in &identifiers {
        checkpoint(cx, "native_ann.generation.quality_membership_id")?;
        if identifier.is_empty() || previous.is_some_and(|last| last >= identifier.as_str()) {
            return Err(refused("metadata", "invalid-quality-membership"));
        }
        previous = Some(identifier);
    }
    if serde_json::to_string(&identifiers)
        .map_err(|_| refused("metadata", "invalid-quality-membership"))?
        != *encoded
    {
        return Err(refused("metadata", "noncanonical-quality-membership"));
    }
    Ok(Some(identifiers))
}

/// Record membership from admitted live rows, never from requested counts or
/// historical physical row counts (which may include tombstones).
pub(super) fn encode_partial_quality_document_ids(
    cx: &Cx,
    quality: &ValidatedFsviBytes,
) -> SearchResult<String> {
    let mut identifiers = Vec::new();
    for physical in 0..quality.record_count() {
        checkpoint(cx, "native_ann.generation.quality_membership_encode")?;
        let row = quality.row(physical)?;
        if row.flags().is_live() {
            identifiers.push(row.doc_id().to_owned());
        }
    }
    identifiers.sort_unstable();
    serde_json::to_string(&identifiers)
        .map_err(|_| refused("metadata", "invalid-quality-membership"))
}

/// Independently admitted vector owners stored in a complete authority generation.
/// This describes stored shape, not semantic readiness or a request's chosen mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeGenerationTopology {
    /// One explicitly selected fast tier.
    FastOnly,
    /// One explicitly selected quality tier.
    QualityOnly,
    /// Complete independently identified fast and quality tiers.
    FullProgressive,
    /// A complete fast corpus and an explicitly admitted, nonempty quality subset.
    /// Counts describe the retained owners, not the query's result window.
    PartialQuality {
        /// Live documents represented by the quality owner.
        quality_documents: u64,
        /// Live documents represented by the complete fast and lexical owners.
        total_documents: u64,
        /// Quality coverage in whole parts per million, rounded down.
        coverage_ppm: u32,
    },
}

pub(in crate::native_ann) fn require_query_tier(
    topology: NativeGenerationTopology,
    tier: EmbedderTierTag,
) -> SearchResult<()> {
    if matches!(
        (topology, tier),
        (NativeGenerationTopology::FastOnly, EmbedderTierTag::Quality)
            | (NativeGenerationTopology::QualityOnly, EmbedderTierTag::Fast)
    ) {
        return Err(refused("query-tier", "unavailable"));
    }
    Ok(())
}

/// Query-capable readers for one fixed-authority native generation.
///
/// The authority-selected vector, lexical and metadata images are parsed and
/// joined before the handle becomes usable. The full FSVI generation and frozen
/// identity, the actual ordered live document sets, and the metadata's source
/// checkpoint must agree. Query-time producer admission is delegated to the
/// existing typed native retrieval path, including zero-result requests.
///
/// This composite route supports one independently identified vector tier
/// (fast or quality), with exact search or an authenticated native HNSW component.
/// A declared ANN requires its raw graph image and full native receipt in the
/// selected metadata; a missing or invalid accelerator is never replaced by
/// exact search. Complete fast-plus-quality generations use two independently
/// admitted owners in one packed Vector component. Each tier independently uses
/// exact retrieval or an owner-bound graph in the packed ANN component.
/// Partial quality requires the separate [`Self::admit_partial_quality`] contract
/// and an explicit partial-quality query. Sharded descriptors remain unsupported. Metadata
/// paths are descriptive names, never paths to open. Vector artifact
/// counts describe physical rows, including tombstones; the common document set
/// and metadata total count describe live documents. Empty generations retain
/// complete identity-bound zero-row vector images, an empty lexical image and
/// the canonical empty document set; absent or corrupt images are still refused.
///
/// Keeping this handle retains the original authority capability and all engine
/// bytes. Publishing or rolling back another generation does not change an
/// in-flight or intentionally retained old reader. Obtain a fresh admitted root
/// snapshot and construct another handle to observe a new generation.
/// Explicit historical selection also admits a queryable handle, carrying the
/// current authority that authorized that selection as separate provenance. Such
/// a handle may be a rollback target, but cannot act as the current head when
/// proposing a rollback or an update. Older ancestors retain their authenticated
/// manifest chain and current snapshot without inventing a physical slot.
pub struct NativeGenerationSnapshot {
    authority: Arc<OpenedGenerationSnapshotV1>,
    readers: CompositeReaders,
}

/// Queryable, immutable components of one exact unpublished candidate.
///
/// Only the source-owned complete generation builder can create this handle.
/// Admission and retrieval use the same component and provider checks as a
/// selected [`NativeGenerationSnapshot`], including both independent owners and
/// configured ANN graphs. Keeping it alive retains every admitted image without
/// reopening the build directory. It contains no qualified root or publication
/// capability, and its candidate authority is never proof of a selected head.
///
/// Preview queries require nonblank text, a positive limit and a nonempty live
/// cohort. This deliberately prevents an empty/no-op result from being used as
/// evidence that a replacement producer can execute a semantic query. Ordinary
/// selected-generation queries retain their existing empty-query behavior.
pub struct NativeGenerationPreview {
    candidate_authority: AuthorityRefV1,
    readers: CompositeReaders,
}

impl NativeGenerationPreview {
    pub(super) const fn new(
        candidate_authority: AuthorityRefV1,
        readers: CompositeReaders,
    ) -> Self {
        Self {
            candidate_authority,
            readers,
        }
    }

    /// Prospective authority identity, not an assertion that it is published.
    #[must_use]
    pub const fn candidate_authority(&self) -> AuthorityRefV1 {
        self.candidate_authority
    }

    /// Independently admitted shape of this candidate's vector owners.
    #[must_use]
    pub const fn topology(&self) -> NativeGenerationTopology {
        self.readers.topology()
    }

    /// Metadata authenticated by this candidate's exact component closure.
    #[must_use]
    pub const fn metadata(&self) -> &GenerationManifest {
        &self.readers.metadata
    }

    /// Authenticated primary-tier construction/search policy of this candidate.
    #[must_use]
    pub const fn retrieval_configuration(&self) -> NativeBuildRetrieval {
        self.readers.retrieval
    }

    /// Independent quality-tier policy; absent for a single-tier candidate.
    #[must_use]
    pub const fn quality_retrieval_configuration(&self) -> Option<NativeBuildRetrieval> {
        self.readers.quality_retrieval
    }

    /// Return the actual covered IDs of an admitted partial candidate.
    ///
    /// Membership comes from the candidate's retained quality vector owner in
    /// lexicographic order. Callers can compare it with the selected predecessor
    /// before publication without model inference or source-content export.
    ///
    /// # Errors
    /// Refuses nonpartial topology, cancellation or an invalid retained row.
    pub fn partial_quality_document_ids(&self, cx: &Cx) -> SearchResult<Vec<String>> {
        self.readers.partial_quality_document_ids(cx)
    }

    /// Lazily preview exactly one admitted vector tier and its lexical source.
    ///
    /// The other producer is neither required nor invoked. The Initial phase
    /// carries the selected tier's score and candidate provenance; attaching
    /// [`NativeProgressiveSearch::with_retained_reranker`] adds a lazy final
    /// cross-encoder stage over this candidate's own bounded stored source.
    /// An optional caller pool owns vector traversal, without inline fallback.
    ///
    /// # Errors
    /// Refuses no-op previews, an absent requested tier, producer disagreement,
    /// invalid budgets or cancellation. Validation must consume every required
    /// stage; constructing or abandoning the query proves no successful preview.
    pub fn search_tier_progressive<'a>(
        &'a self,
        cx: &'a Cx,
        tier: EmbedderTierTag,
        embedder: &'a dyn Embedder,
        text: &'a str,
        k: usize,
        pool: Option<BlockingPoolHandle>,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        Self::require_query(cx, text, k, self.readers.bounded_limit(k))?;
        require_query_tier(self.topology(), tier)?;
        self.readers.primary_query(
            cx,
            embedder,
            text,
            k,
            tier == EmbedderTierTag::Quality,
            pool,
        )
    }

    /// Execute the candidate's primary tier and hydrate its actual results.
    /// Quality-only candidates invoke only the supplied quality producer.
    ///
    /// # Errors
    /// Refuses no-op previews, producer drift, invalid query embeddings,
    /// retrieval/hydration failures and cancellation. No fallback is attempted.
    pub async fn search_text(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        Self::require_query(cx, text, k, self.readers.bounded_limit(k))?;
        self.readers.search_text(cx, embedder, text, k).await
    }

    /// Preview the primary tier with vector traversal on the caller's pool.
    /// Inference, lexical scoring and hydration retain their ordinary async
    /// ownership. The candidate's immutable vector owner is retained by the job.
    ///
    /// # Errors
    /// Has the admission and no-op refusal of [`Self::search_text`], plus
    /// blocking-pool failures. A stopped pool never falls back to inline work.
    pub async fn search_text_with_pool(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        text: &str,
        k: usize,
        pool: BlockingPoolHandle,
    ) -> SearchResult<Vec<ScoredResult>> {
        Self::require_query(cx, text, k, self.readers.bounded_limit(k))?;
        self.readers
            .search_text_with_pool(cx, embedder, text, k, pool)
            .await
    }

    /// Execute quality directly, including in a complete two-tier candidate.
    /// The fast producer is neither required nor invoked.
    ///
    /// # Errors
    /// Refuses no-op previews, missing quality, producer/embedding mismatch,
    /// retrieval/hydration failures and cancellation.
    pub async fn search_quality_text(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        Self::require_query(cx, text, k, self.readers.bounded_limit(k))?;
        self.readers
            .search_quality_text(cx, embedder, text, k)
            .await
    }

    /// Preview quality directly with retained vector traversal on the caller's
    /// pool. The fast provider is neither required nor invoked.
    ///
    /// # Errors
    /// Has the refusals of [`Self::search_quality_text`], plus blocking-pool
    /// failures, without inline traversal or an Initial-only fallback.
    pub async fn search_quality_text_with_pool(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        text: &str,
        k: usize,
        pool: BlockingPoolHandle,
    ) -> SearchResult<Vec<ScoredResult>> {
        Self::require_query(cx, text, k, self.readers.bounded_limit(k))?;
        self.readers
            .search_quality_text_with_pool(cx, embedder, text, k, pool)
            .await
    }

    /// Execute both independent producers and return their refined union.
    /// An Initial-only page never counts as successful complete validation.
    ///
    /// # Errors
    /// Refuses no-op previews, absent tiers, either producer's drift or failure,
    /// invalid embeddings, retrieval/hydration failures and cancellation.
    pub async fn search_refined_text(
        &self,
        cx: &Cx,
        fast: &dyn Embedder,
        quality: &dyn Embedder,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        Self::require_query(cx, text, k, self.readers.bounded_limit(k))?;
        self.readers
            .search_refined_text(cx, fast, quality, text, k)
            .await
    }

    /// Preview both independent tiers with their vector traversals on the
    /// caller's pool. Both arms must complete successfully before delivery.
    ///
    /// # Errors
    /// Has the refusals of [`Self::search_refined_text`], plus blocking-pool
    /// failures. Cancellation or a failed worker never returns a partial page.
    pub async fn search_refined_text_with_pool(
        &self,
        cx: &Cx,
        fast: &dyn Embedder,
        quality: &dyn Embedder,
        text: &str,
        k: usize,
        pool: BlockingPoolHandle,
    ) -> SearchResult<Vec<ScoredResult>> {
        Self::require_query(cx, text, k, self.readers.bounded_limit(k))?;
        self.readers
            .search_refined_text_with_pool(cx, fast, quality, text, k, pool)
            .await
    }

    /// Lazily preview both phases from this exact candidate.
    /// Callers using this as validation must consume refinement successfully;
    /// an unpolled or dropped stream proves no complete semantic execution.
    /// The returned query accepts fusion, candidate and beam tuning, and
    /// [`NativeProgressiveSearch::with_retained_reranker`] reads only its final
    /// candidate source from these same admitted images under a byte budget.
    ///
    /// # Errors
    /// Refuses no-op previews, absent tiers, producer drift and cancellation.
    pub fn search_progressive_with_quality<'a>(
        &'a self,
        cx: &'a Cx,
        fast: &'a dyn Embedder,
        quality: &'a dyn Embedder,
        text: &'a str,
        k: usize,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        Self::require_query(cx, text, k, self.readers.bounded_limit(k))?;
        self.readers
            .search_progressive_with_quality(cx, fast, quality, text, k)
    }

    /// Lazily preview both phases with retained vector traversal on the caller's
    /// pool. Construction starts no inference or worker; quality remains lazy.
    /// The returned query retains ordinary tuning and retained-source reranking.
    ///
    /// # Errors
    /// Has the admission refusals of [`Self::search_progressive_with_quality`].
    /// Polling a phase also propagates pool, retrieval and cancellation failures.
    pub fn search_progressive_with_quality_and_pool<'a>(
        &'a self,
        cx: &'a Cx,
        fast: &'a dyn Embedder,
        quality: &'a dyn Embedder,
        text: &'a str,
        k: usize,
        pool: BlockingPoolHandle,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        Self::require_query(cx, text, k, self.readers.bounded_limit(k))?;
        self.readers
            .search_progressive_with_quality_and_pool(cx, fast, quality, text, k, pool)
    }

    /// Explicitly preview refinement over this candidate's admitted quality subset.
    ///
    /// Initial searches the complete fast/lexical corpus; refinement retains
    /// uncovered fast candidates and adds independently retrieved quality rows.
    /// Coverage remains the exact stored topology returned by [`Self::topology`].
    ///
    /// # Errors
    /// Refuses no-op previews, a non-partial topology, producer disagreement,
    /// invalid query policy and cancellation. Quality work remains lazy.
    pub fn search_progressive_with_partial_quality<'a>(
        &'a self,
        cx: &'a Cx,
        fast: &'a dyn Embedder,
        quality: &'a dyn Embedder,
        text: &'a str,
        k: usize,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        Self::require_query(cx, text, k, self.readers.bounded_limit(k))?;
        self.readers
            .search_progressive_with_partial_quality(cx, fast, quality, text, k, None)
    }

    /// Preview explicit partial refinement with traversal on the caller's pool.
    ///
    /// # Errors
    /// Has the refusals of [`Self::search_progressive_with_partial_quality`],
    /// plus pool errors without inline fallback.
    pub fn search_progressive_with_partial_quality_and_pool<'a>(
        &'a self,
        cx: &'a Cx,
        fast: &'a dyn Embedder,
        quality: &'a dyn Embedder,
        text: &'a str,
        k: usize,
        pool: BlockingPoolHandle,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        Self::require_query(cx, text, k, self.readers.bounded_limit(k))?;
        self.readers
            .search_progressive_with_partial_quality(cx, fast, quality, text, k, Some(pool))
    }

    pub(in crate::native_ann) fn require_query(
        cx: &Cx,
        text: &str,
        k: usize,
        eligible: usize,
    ) -> SearchResult<()> {
        checkpoint(cx, "native_ann.generation.preview")?;
        let parsed = ParsedQuery::parse(text);
        if text.trim().is_empty()
            || (parsed.has_negations() && parsed.is_positive_empty())
            || k == 0
            || eligible == 0
        {
            return Err(refused("preview", "nonempty-semantic-query-required"));
        }
        Ok(())
    }

    /// Export this candidate's complete live stored source cohort.
    ///
    /// Uses the retained admitted images without reopening source paths or
    /// invoking models. Ordering, preserved fields and the UTF-8 payload budget
    /// match [`NativeGenerationSnapshot::source_documents`]. This does not
    /// establish that the candidate is published or selected.
    ///
    /// # Errors
    /// Refuses cancellation, budget exhaustion, unsupported schemas, invalid
    /// stored fields and disagreement with the admitted live cohort. Failure
    /// returns no partial collection.
    pub fn source_documents(
        &self,
        cx: &Cx,
        max_source_bytes: usize,
    ) -> SearchResult<Vec<IndexableDocument>> {
        self.readers.source_documents(cx, max_source_bytes)
    }

    pub(in crate::native_ann) fn scoped_lexical(&self) -> &dyn LexicalRead {
        &self.readers.lexical
    }

    pub(in crate::native_ann) fn scope_readers(&self) -> &CompositeReaders {
        &self.readers
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::native_ann) fn scoped_progressive<'a>(
        &'a self,
        cx: &'a Cx,
        primary: &'a dyn Embedder,
        quality: Option<&'a dyn Embedder>,
        quality_only: bool,
        lexical: &'a dyn LexicalRead,
        text: &'a str,
        k: usize,
        allowed: Arc<BTreeSet<String>>,
        pool: Option<BlockingPoolHandle>,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        Self::require_query(cx, text, k, allowed.len())?;
        self.readers.scoped_progressive(
            cx,
            primary,
            quality,
            quality_only,
            lexical,
            text,
            k,
            allowed,
            pool,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::native_ann) fn scoped_partial_quality_progressive<'a>(
        &'a self,
        cx: &'a Cx,
        fast: &'a dyn Embedder,
        quality: &'a dyn Embedder,
        lexical: &'a dyn LexicalRead,
        text: &'a str,
        k: usize,
        allowed: Arc<BTreeSet<String>>,
        pool: Option<BlockingPoolHandle>,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        Self::require_query(cx, text, k, allowed.len())?;
        self.readers.scoped_progressive_with_coverage(
            cx,
            fast,
            Some(quality),
            false,
            lexical,
            text,
            k,
            allowed,
            pool,
            VectorCohortCoverage::PartialQuality,
        )
    }
}

pub(super) struct CompositeReaders {
    vector: Arc<NativeAnnIndex>,
    quality: Option<Arc<NativeAnnIndex>>,
    lexical: QuillSnapshotIndex,
    metadata: GenerationManifest,
    tier: EmbedderTierTag,
    retrieval: NativeBuildRetrieval,
    quality_retrieval: Option<NativeBuildRetrieval>,
    components: ExactGenerationComponentsV1,
    topology: NativeGenerationTopology,
}

/// Display fields copied from one retained document. The snippet is a prefix
/// of the original content; the identifier, title and metadata are complete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeSourceSnippet {
    /// Retained, case-sensitive document identifier.
    pub id: String,
    /// At most the requested number of Unicode scalar values of content.
    pub snippet: String,
    /// Complete nonempty stored title, if present.
    pub title: Option<String>,
    /// Complete canonical source metadata.
    pub metadata: std::collections::HashMap<String, String>,
}

impl NativeGenerationSnapshot {
    /// Admit the actual search components retained by a qualified root snapshot.
    ///
    /// `expected_vector` is the caller's selected generation/embedding/storage
    /// binding. It must not be inferred from the untrusted component it admits.
    /// `schema` identifies the caller's lexical schema. Stored `metadata_json`
    /// is resolved by that descriptor for lexical and semantic-only winners;
    /// schemas without the column return no metadata. Complete source export
    /// remains restricted to the default schema's canonical document contract.
    /// `max_image_bytes` bounds
    /// the combined encoded component and activation-manifest images, not total
    /// process RSS; parsed search structures have additional memory costs.
    ///
    /// All operations are synchronous with cancellation checkpoints between
    /// parsing/validation stages. Large admissions belong on the caller's
    /// bounded blocking lane. No runtime or background task is created.
    ///
    /// # Errors
    ///
    /// Rejects cancellation, unsupported topology, budget exhaustion, corrupt
    /// component bytes, source/document-set/identity disagreement, unsupported
    /// activation predicates, and inconsistent metadata artifact declarations.
    /// Admission errors contain bounded reason codes rather than raw metadata.
    pub fn admit_exact(
        cx: &Cx,
        authority: Arc<OpenedGenerationSnapshotV1>,
        expected_vector: &FsviV2IdentityBinding,
        schema: SchemaDescriptor,
        config: QuillConfig,
        max_image_bytes: usize,
    ) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.generation.admit")?;
        let component = |role| {
            authority.closure().bytes(role).ok_or_else(|| {
                invalid(
                    "generation.component",
                    "missing",
                    "the selected authority must retain every mandatory component",
                )
            })
        };
        let readers = admit_components(
            cx,
            authority.manifest(),
            expected_vector,
            schema,
            config,
            [
                component(GenerationComponentRole::Vector)?,
                component(GenerationComponentRole::Lexical)?,
                component(GenerationComponentRole::Metadata)?,
            ],
            authority.closure().bytes(GenerationComponentRole::Ann),
            max_image_bytes,
        )?;
        checkpoint(cx, "native_ann.generation.admitted")?;
        Ok(Self { authority, readers })
    }

    /// Admit both independently selected tiers and their common Quill cohort.
    ///
    /// Both bindings come from the caller's trusted selection. Their complete
    /// document coverage, source checkpoint and artifact generation must agree;
    /// dimensions and producers may differ. The same aggregate image budget,
    /// synchronous cancellation and retained-owner rules as [`Self::admit_exact`]
    /// apply. Neither a missing tier nor an ANN is silently substituted.
    ///
    /// # Errors
    /// Refuses incomplete topology, identity/source/coverage disagreement,
    /// malformed or oversized components and cancellation before serving.
    #[allow(clippy::too_many_arguments)]
    pub fn admit_progressive(
        cx: &Cx,
        authority: Arc<OpenedGenerationSnapshotV1>,
        expected_fast: &FsviV2IdentityBinding,
        expected_quality: &FsviV2IdentityBinding,
        schema: SchemaDescriptor,
        config: QuillConfig,
        max_image_bytes: usize,
    ) -> SearchResult<Self> {
        Self::admit_with_quality_coverage(
            cx,
            authority,
            expected_fast,
            expected_quality,
            schema,
            config,
            VectorCohortCoverage::Complete,
            max_image_bytes,
        )
    }

    /// Admit a complete fast/lexical corpus and an explicitly selected quality subset.
    ///
    /// The partial wire contract must be selected by the caller; this method
    /// never downgrades a complete generation or infers permission from row
    /// counts. Every quality document must be live in the retained fast owner,
    /// and both producers, source checkpoint and artifact generation are joined
    /// before serving. [`Self::topology`] reports the actual retained coverage.
    /// Full-progressive search remains unavailable; use the explicit partial
    /// query surface, or search the complete fast/lexical corpus directly.
    ///
    /// # Errors
    /// Refuses complete or empty quality coverage, foreign quality IDs, changed
    /// bindings, malformed components, budget exhaustion and cancellation.
    #[allow(clippy::too_many_arguments)]
    pub fn admit_partial_quality(
        cx: &Cx,
        authority: Arc<OpenedGenerationSnapshotV1>,
        expected_fast: &FsviV2IdentityBinding,
        expected_quality: &FsviV2IdentityBinding,
        schema: SchemaDescriptor,
        config: QuillConfig,
        max_image_bytes: usize,
    ) -> SearchResult<Self> {
        Self::admit_with_quality_coverage(
            cx,
            authority,
            expected_fast,
            expected_quality,
            schema,
            config,
            VectorCohortCoverage::PartialQuality,
            max_image_bytes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn admit_with_quality_coverage(
        cx: &Cx,
        authority: Arc<OpenedGenerationSnapshotV1>,
        expected_fast: &FsviV2IdentityBinding,
        expected_quality: &FsviV2IdentityBinding,
        schema: SchemaDescriptor,
        config: QuillConfig,
        coverage: VectorCohortCoverage,
        max_image_bytes: usize,
    ) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.generation.progressive_admit")?;
        let component = |role| {
            authority
                .closure()
                .bytes(role)
                .ok_or_else(|| refused("component", "missing"))
        };
        let readers = admit_components_with_coverage(
            cx,
            authority.manifest(),
            expected_fast,
            Some(expected_quality),
            schema,
            config,
            [
                component(GenerationComponentRole::Vector)?,
                component(GenerationComponentRole::Lexical)?,
                component(GenerationComponentRole::Metadata)?,
            ],
            authority.closure().bytes(GenerationComponentRole::Ann),
            coverage,
            max_image_bytes,
        )?;
        checkpoint(cx, "native_ann.generation.admitted")?;
        Ok(Self { authority, readers })
    }

    /// The exact authority retained by this reader, including rollback sequence.
    #[must_use]
    pub fn authority(&self) -> AuthorityRefV1 {
        self.authority.head().authority
    }

    /// The original admitted root capability, never a fresh current-head lookup.
    #[must_use]
    pub const fn retained_snapshot(&self) -> &Arc<OpenedGenerationSnapshotV1> {
        &self.authority
    }

    /// The primary vector tier declared by the selected metadata.
    /// Complete progressive generations have a fast primary and a separate quality owner.
    #[must_use]
    pub const fn tier(&self) -> EmbedderTierTag {
        self.readers.tier
    }

    /// Stored independently admitted tier topology, without running a query.
    #[must_use]
    pub const fn topology(&self) -> NativeGenerationTopology {
        self.readers.topology()
    }

    /// The actual primary-tier retrieval mode of this handle.
    #[must_use]
    pub const fn retrieval_mode(&self) -> NativeRetrievalMode {
        match self.readers.retrieval {
            NativeBuildRetrieval::Exact => NativeRetrievalMode::Exact {
                reason: super::NativeExactReason::Requested,
            },
            NativeBuildRetrieval::Hnsw { .. } => NativeRetrievalMode::Ann,
        }
    }

    /// The authenticated primary-tier construction/search policy.
    /// Rebuilds can preserve its graph parameters and seed without reading paths.
    #[must_use]
    pub const fn retrieval_configuration(&self) -> NativeBuildRetrieval {
        self.readers.retrieval
    }

    /// Independent quality-tier policy in a complete fast/quality generation.
    /// A single-tier generation returns `None`, including quality-only topology.
    #[must_use]
    pub const fn quality_retrieval_configuration(&self) -> Option<NativeBuildRetrieval> {
        self.readers.quality_retrieval
    }

    /// The validated metadata image belonging to this exact authority.
    #[must_use]
    pub const fn metadata(&self) -> &GenerationManifest {
        &self.readers.metadata
    }

    /// Return the authenticated covered IDs of an explicitly partial generation.
    ///
    /// IDs are returned in lexicographic order from the retained quality vector
    /// owner. This does not export source content, load models, infer vectors or
    /// reopen paths. It supports mutation preflight before loading either model;
    /// counts or caller-provided IDs never substitute for this actual membership.
    ///
    /// # Errors
    /// Refuses nonpartial topology, cancellation or an invalid retained row.
    pub fn partial_quality_document_ids(&self, cx: &Cx) -> SearchResult<Vec<String>> {
        self.readers.partial_quality_document_ids(cx)
    }

    /// Recover the complete stored source cohort from this retained generation.
    ///
    /// Documents follow physical live FSVI row order; tombstones are omitted.
    /// Each identifier is resolved directly against the retained Quill image,
    /// and its stored identifier, full content, title and metadata must exist.
    /// This route supports the shipping [`DEFAULT_SCHEMA`] and canonical string
    /// metadata. It never reopens an artifact or original source pathname and
    /// does not load a model or perform inference.
    ///
    /// Quill stores both an absent title and an explicitly empty title as empty
    /// bytes. Both export as `None`; nonempty titles and document content are
    /// preserved exactly. The native builder embeds document content only.
    ///
    /// `max_source_bytes` bounds the sum of the raw stored UTF-8 byte lengths of
    /// `id`, `content`, `title` and `metadata_json` across all exported documents,
    /// including JSON punctuation and escaping. Each field is checked before
    /// its source bytes are copied or decoded. This is a payload budget, not a
    /// bound on process RSS: reader caches and decoded documents require
    /// additional storage. Run large exports on the caller's bounded blocking
    /// lane, as with admission.
    ///
    /// # Errors
    ///
    /// Refuses cancellation, budget exhaustion, unsupported schemas, missing or
    /// invalid stored fields, noncanonical metadata, and any disagreement with
    /// the admitted live cohort. Failure returns no partial source collection.
    pub fn source_documents(
        &self,
        cx: &Cx,
        max_source_bytes: usize,
    ) -> SearchResult<Vec<IndexableDocument>> {
        self.readers.source_documents(cx, max_source_bytes)
    }

    /// Recover one exact live source document from the retained generation.
    ///
    /// Only the requested document's stored fields are copied. The full
    /// identifier must belong to a live FSVI row and resolve to the same retained
    /// Quill document; absent or tombstoned IDs return `None`. No corpus export,
    /// pathname lookup, model load or inference is performed.
    ///
    /// The byte budget and lossless source checks match
    /// [`Self::source_documents`]. Missing documents consume no source bytes.
    /// This returns the complete document, so its full stored payload must fit
    /// the budget even if a caller later displays only a content window.
    ///
    /// # Errors
    /// Refuses cancellation, insufficient source budget, unsupported schemas,
    /// invalid stored fields or disagreement between the retained live readers.
    pub fn source_document(
        &self,
        cx: &Cx,
        doc_id: &str,
        max_source_bytes: usize,
    ) -> SearchResult<Option<IndexableDocument>> {
        Ok(self
            .readers
            .source_documents_by_ids(cx, &[doc_id], max_source_bytes)?
            .pop()
            .flatten())
    }

    /// Recover selected source documents without scanning or exporting the cohort.
    ///
    /// Results preserve input order and duplicate IDs; missing or tombstoned
    /// identifiers occupy `None` slots. Every returned occurrence is independently
    /// charged against the aggregate raw stored-field budget, including duplicate
    /// occurrences. An empty request returns an empty collection after ordinary
    /// schema/cancellation admission. The caller owns the request-count limit;
    /// the payload budget does not bound output slots for absent documents.
    ///
    /// Exact live FSVI membership is resolved before Quill fields are read. Each
    /// field's length is checked before copying, and the same strict decoder as
    /// [`Self::source_documents`] preserves content, title and canonical metadata.
    /// The selection stays on this retained reader throughout. No model or source
    /// pathname is accessed, and unrelated documents are never materialized.
    ///
    /// # Errors
    /// Refuses cancellation, insufficient aggregate budget, unsupported schemas,
    /// invalid stored fields or inconsistent live identity. Failure returns no
    /// partial collection, even when earlier requested documents were valid.
    pub fn source_documents_by_ids(
        &self,
        cx: &Cx,
        doc_ids: &[&str],
        max_source_bytes: usize,
    ) -> SearchResult<Vec<Option<IndexableDocument>>> {
        self.readers
            .source_documents_by_ids(cx, doc_ids, max_source_bytes)
    }

    /// Hydrate ranked document previews without copying entire content fields.
    ///
    /// ID order, duplicates and absent rows follow [`Self::source_documents_by_ids`].
    /// Each occurrence charges the complete stored ID, title, canonical metadata,
    /// and at most `max_chars` Unicode scalar values of content to one shared
    /// byte budget. The whole stored content must still be valid UTF-8.
    ///
    /// # Errors
    /// Refuses invalid source, insufficient budget and cancellation as for full export.
    pub fn source_snippets_by_ids(
        &self,
        cx: &Cx,
        doc_ids: &[&str],
        max_chars: usize,
        max_source_bytes: usize,
    ) -> SearchResult<Vec<Option<NativeSourceSnippet>>> {
        self.readers.read_selected_sources(
            cx,
            doc_ids,
            max_source_bytes,
            Some(max_chars),
            |document| NativeSourceSnippet {
                id: document.id,
                snippet: document.content,
                title: document.title,
                metadata: document.metadata,
            },
        )
    }

    /// Prepare incremental upserts and deletes against this exact retained cohort.
    ///
    /// The provider must reproduce the complete admitted embedding identity,
    /// including its persisted storage contract. Precision and native graph
    /// parameters/seed are inherited. Unchanged content reuses its exact stored
    /// vector; title/metadata edits still replace the retained source document.
    /// Models are never discovered or substituted, and no source/artifact path
    /// is reopened. The update retains the original authority pin and vector
    /// owner, including while a successor is finalized on a caller-owned pool.
    ///
    /// Source export uses [`Self::source_documents`]' payload budget. Preparation
    /// is synchronous and belongs on the caller's blocking lane for large
    /// cohorts. No directory, inference or publication starts during this call.
    /// Finish with [`NativeIndexUpdate::build_exact_generation`] and publish its
    /// complete result through the normal expected-authority publisher. The
    /// original snapshot remains independently searchable throughout.
    ///
    /// # Errors
    /// Refuses cancellation, non-newer artifact generations, changed producers,
    /// unsupported storage, source-budget exhaustion or inconsistent live rows.
    pub fn begin_update(
        &self,
        cx: &Cx,
        directory: impl AsRef<Path>,
        generation: ArtifactGenerationIdentityV1,
        provider: Arc<dyn Embedder>,
        max_source_bytes: usize,
    ) -> SearchResult<NativeIndexUpdate> {
        self.readers.require_mutable_topology()?;
        if self.readers.quality.is_some() {
            return Err(refused("update", "both-providers-required"));
        }
        NativeIndexUpdate::from_authority(
            cx,
            directory,
            generation,
            provider,
            self,
            max_source_bytes,
            Arc::clone(&self.readers.vector),
            None,
        )
    }

    /// Complete the quality tier of this selected fast-only or partial generation.
    ///
    /// The fast producer must match its full retained identity, but no fast
    /// document inference is performed. All quality inputs come from this
    /// snapshot's authenticated Quill source; original input files are unnecessary.
    /// For a fast-only source the new quality provider may have a different model
    /// and dimension while preserving the stored prepared-input contract and
    /// semantic/control kind. A partial source requires its existing complete
    /// quality identity: covered rows reuse their exact stored vectors and only
    /// missing rows are inferred. Its quality precision and retrieval policy are
    /// inherited unless explicitly overridden on the completion plan.
    ///
    /// The returned plan cannot edit sources, replace the fast model or change
    /// its precision. It builds a complete independently bound two-tier successor
    /// at the same source checkpoint. Publish it with this reader's expected
    /// authority pair; a competing writer is never adopted as a new update base.
    ///
    /// # Errors
    /// Refuses complete or quality-only topology, historical selection, non-newer
    /// generation, identity/input drift, source-budget exhaustion or cancellation
    /// before work. Ordinary partial mutation and model migration remain separate.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_quality_completion(
        &self,
        cx: &Cx,
        directory: impl AsRef<Path>,
        generation: ArtifactGenerationIdentityV1,
        fast_provider: Arc<dyn Embedder>,
        quality_provider: Arc<dyn Embedder>,
        max_source_bytes: usize,
    ) -> SearchResult<NativeQualityCompletion> {
        NativeQualityCompletion::from_authority(
            cx,
            directory,
            generation,
            fast_provider,
            quality_provider,
            self,
            max_source_bytes,
            Arc::clone(&self.readers.vector),
            self.readers.quality.as_ref().map(Arc::clone),
        )
    }

    /// Prepare complete two-tier edits with independent retained-vector reuse.
    ///
    /// Both providers are admitted before source export or inference. Each tier
    /// reuses only its own unchanged document vectors; the original authority
    /// and both owners remain pinned through successor construction. Source
    /// export has [`Self::source_documents`]' payload budget. Finish with
    /// [`NativeIndexUpdate::build_progressive_generation`].
    ///
    /// # Errors
    /// Refuses an absent tier, non-newer generation, changed provider, unsupported
    /// storage, inconsistent retained sources, cancellation or source-budget exhaustion.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_progressive_update(
        &self,
        cx: &Cx,
        directory: impl AsRef<Path>,
        generation: ArtifactGenerationIdentityV1,
        fast_provider: Arc<dyn Embedder>,
        quality_provider: Arc<dyn Embedder>,
        max_source_bytes: usize,
    ) -> SearchResult<NativeIndexUpdate> {
        checkpoint(cx, "native_ann.generation.progressive_update")?;
        let quality = self.readers.full_quality()?;
        NativeIndexUpdate::from_authority(
            cx,
            directory,
            generation,
            fast_provider,
            self,
            max_source_bytes,
            Arc::clone(&self.readers.vector),
            Some((quality_provider, Arc::clone(quality))),
        )
    }

    /// Prepare subset-preserving edits against this partial quality generation.
    ///
    /// Both providers must match their independent complete retained identities.
    /// Surviving covered IDs retain quality membership; changed covered content
    /// is re-embedded in both tiers, while new and uncovered IDs stay fast-only.
    /// Unchanged vectors retain their stored bits. Storage and graph policies
    /// are inherited independently and may be explicitly overridden on the plan.
    ///
    /// [`NativePartialQualityUpdate::build`] preserves partial topology and
    /// refuses deletion of all covered or all uncovered IDs before inference or
    /// work files. Explicitly select
    /// [`NativePartialQualityUpdate::build_with_realized_coverage`] to derive a
    /// fast-only or complete successor at those boundaries. Neither route adds
    /// previously uncovered IDs to quality; use the explicit quality-completion
    /// contract for that expansion. Source export uses
    /// [`Self::source_documents`]'s payload budget; the original authority and
    /// both owners remain pinned throughout construction. Publication still
    /// requires this snapshot's original expected authority pair.
    ///
    /// # Errors
    /// Refuses nonpartial topology, changed providers, historical selection,
    /// non-newer generations, source-budget exhaustion or cancellation.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_partial_quality_update(
        &self,
        cx: &Cx,
        directory: impl AsRef<Path>,
        generation: ArtifactGenerationIdentityV1,
        fast_provider: Arc<dyn Embedder>,
        quality_provider: Arc<dyn Embedder>,
        max_source_bytes: usize,
    ) -> SearchResult<NativePartialQualityUpdate> {
        checkpoint(cx, "native_ann.generation.partial_quality_update")?;
        let quality = self.readers.partial_quality()?;
        NativePartialQualityUpdate::from_authority(
            cx,
            directory,
            generation,
            fast_provider,
            quality_provider,
            self,
            max_source_bytes,
            Arc::clone(&self.readers.vector),
            Arc::clone(quality),
        )
    }

    /// Explicitly migrate models while preserving a partial quality subset.
    ///
    /// Each provider retains its own prepared-input contract. Its complete
    /// producer identity determines reuse independently: an unchanged tier
    /// copies its exact stored vectors, while a changed fast provider infers
    /// every surviving source and a changed quality provider infers only the
    /// surviving covered IDs. Equal dimensions or output values never permit
    /// reuse across producers. New and previously uncovered IDs stay fast-only.
    ///
    /// Storage and graph policies are inherited independently. The restricted
    /// plan supports ordinary source edits, monotone source boundaries, and
    /// explicit storage overrides while retaining the proper-subset contract.
    /// With no edits and the same source cut, it preserves the exact Quill image.
    /// Models are never discovered or substituted, and preparation creates no
    /// files or inference. Publish with this snapshot's original expected pair.
    /// Ordinary partial updates still require the original providers; the
    /// complete-generation migration API remains separate.
    ///
    /// # Errors
    /// Refuses nonpartial topology, changed prepared-input contracts, historical
    /// selection, non-newer generations, invalid or drifting producers,
    /// source-budget exhaustion or cancellation.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_partial_quality_model_migration(
        &self,
        cx: &Cx,
        directory: impl AsRef<Path>,
        generation: ArtifactGenerationIdentityV1,
        fast_provider: Arc<dyn Embedder>,
        quality_provider: Arc<dyn Embedder>,
        max_source_bytes: usize,
    ) -> SearchResult<NativePartialQualityUpdate> {
        checkpoint(cx, "native_ann.generation.partial_quality_model_migration")?;
        let quality = self.readers.partial_quality()?;
        NativePartialQualityUpdate::from_authority_model_migration(
            cx,
            directory,
            generation,
            fast_provider,
            quality_provider,
            self,
            max_source_bytes,
            Arc::clone(&self.readers.vector),
            Arc::clone(quality),
        )
    }

    /// Explicitly migrate the selected generation's models using retained source.
    ///
    /// Each supplied provider is compared with its own complete authenticated
    /// identity after normalizing the inherited storage encoding. An unchanged
    /// tier reuses its exact persisted vectors and compatible native graph;
    /// a changed tier embeds every surviving document under its new identity.
    /// Equal model names, dimensions or output values never authorize reuse.
    /// The primary provider corresponds to [`Self::tier`], including a
    /// quality-only generation. A full progressive generation requires both
    /// providers, and a singleton generation forbids an additional quality tier.
    ///
    /// Retained documents already carry prepared embedding input, so neither
    /// provider may change its input contract. Precision and graph policies
    /// are inherited independently; explicit overrides on the returned update
    /// retain the ordinary precision-change re-embedding rules. The selected
    /// authority and all owners stay pinned through complete successor building.
    /// Finish with [`NativeIndexUpdate::build_exact_generation`] or
    /// [`NativeIndexUpdate::build_progressive_generation`] and publish using
    /// this snapshot's expected authority pair. Preparation creates no files,
    /// starts no inference, and applies [`Self::source_documents`]' source budget.
    ///
    /// # Errors
    /// Refuses topology changes, changed prepared-input contracts, non-newer
    /// generations, cancelled work, invalid or drifting providers, inconsistent
    /// source membership, source-budget exhaustion or a reopened predecessor.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_model_migration(
        &self,
        cx: &Cx,
        directory: impl AsRef<Path>,
        generation: ArtifactGenerationIdentityV1,
        primary_provider: Arc<dyn Embedder>,
        quality_provider: Option<Arc<dyn Embedder>>,
        max_source_bytes: usize,
    ) -> SearchResult<NativeIndexUpdate> {
        checkpoint(cx, "native_ann.generation.model_migration")?;
        self.readers.require_mutable_topology()?;
        let quality = match (&self.readers.quality, quality_provider) {
            (Some(quality), Some(provider)) => Some((provider, Arc::clone(quality))),
            (None, None) => None,
            _ => return Err(refused("model_migration", "topology-change")),
        };
        NativeIndexUpdate::from_authority_model_migration(
            cx,
            directory,
            generation,
            primary_provider,
            self,
            max_source_bytes,
            Arc::clone(&self.readers.vector),
            quality,
        )
    }

    /// Propose this retained generation as a rollback of `current`.
    ///
    /// Both handles must have been independently admitted from the same logical
    /// root, physical root and persistent control anchors under the same
    /// mutation-authorized security profile. `self` must be older than
    /// `current`; an uncertain pending local successor cannot authorize a
    /// rollback. The caller supplies a fresh manifest object id and writer
    /// fence, just as for a newly built candidate.
    ///
    /// `current` must come from ordinary current-head admission. An explicitly
    /// selected ancestor is usable as `self`, including after the process
    /// that originally selected it has exited, but its selection provenance is
    /// not authority to act as the current head.
    ///
    /// The proposal advances the authority sequence beyond `current` while
    /// retaining this handle's exact artifact generation, producer identity,
    /// source checkpoint, document set and component receipts. It never copies
    /// an old authority number over the current one, loads a model, reads a
    /// pathname or changes either reader. Only the new activation manifest
    /// needs staging: the original immutable component objects stay retained.
    ///
    /// Publish through [`crate::GenerationPublisherV1`] with the exact pair from
    /// `current.retained_snapshot().authority_pair()`. That publisher re-admits
    /// the physical objects, checks the expected pair under its exclusive lock,
    /// and applies the configured external floor. This pure proposal does not
    /// reserve the head or prove publication; persist its prepared attempt and
    /// reconcile any unknown outcome through the normal publication protocol.
    ///
    /// # Errors
    ///
    /// Refuses cancellation, a foreign or replaced root/anchor, a security
    /// profile change, read-only or pending authority, an explicitly selected
    /// historical selection used as `current`, a target that is not older than `current`,
    /// sequence exhaustion, or invalid candidate ids.
    pub fn rollback_candidate(
        &self,
        cx: &Cx,
        current: &Self,
        object_id: [u8; 16],
        writer_fence_sha256: [u8; 32],
    ) -> SearchResult<GenerationCandidateV1> {
        checkpoint(cx, "native_ann.generation.rollback")?;
        let target = self.retained_snapshot();
        let expected = current.retained_snapshot();
        if expected.selected_history_of().is_some() {
            return Err(refused("rollback", "current-is-selected-predecessor"));
        }
        if target.root_id() != expected.root_id()
            || !same_root_security_identity(target.root_witness(), expected.root_witness())
            || !same_anchor_security_identity(
                target.anchor_witnesses().lock(),
                expected.anchor_witnesses().lock(),
            )
            || !same_anchor_security_identity(
                target.anchor_witnesses().authority(),
                expected.anchor_witnesses().authority(),
            )
        {
            return Err(refused("rollback", "different-root-or-anchors"));
        }
        if target.profile() != expected.profile() || !expected.profile().permits_mutation() {
            return Err(refused("rollback", "security-profile"));
        }
        if matches!(
            expected.health(),
            SnapshotHealthV1::PendingLocalSuccessor { .. }
        ) {
            return Err(refused("rollback", "pending-reconciliation"));
        }
        if self.authority().sequence >= current.authority().sequence {
            return Err(refused("rollback", "target-not-older"));
        }
        current
            .authority()
            .next_sequence()
            .map_err(|_| refused("rollback", "sequence-exhausted"))?;
        GenerationCandidateV1::new(
            object_id,
            Some(current.authority()),
            GenerationAuthorityActionV1::Rollback,
            target.manifest().generation(),
            writer_fence_sha256,
            &self.readers.components,
        )
        .map_err(|_| refused("rollback", "invalid-candidate"))
    }

    /// Propose a rollback from an explicitly admitted predecessor or ancestor.
    ///
    /// The retained snapshot must carry opaque historical-selection provenance
    /// from the qualified root. That admission authenticates the exact current
    /// authority and activation manifest, plus this target's complete healthy
    /// closure. The current generation's components may be unavailable; they
    /// never become query-capable through this operation. Ordinary current-head
    /// snapshots cannot use this route.
    ///
    /// This method derives the expected current authority from that provenance,
    /// rechecks the actual two-slot pair and retained ancestry provenance, and
    /// permits only the cooperative local profile without an external floor.
    /// It retains the target's exact images, source checkpoint, producer and
    /// artifact generation while proposing a higher authority sequence. It
    /// performs no inference, file access, repair, or publication.
    ///
    /// Publish using this target's `retained_snapshot().authority_pair()`, which
    /// captures the newer current head and its physical predecessor. An older
    /// selected target need not occupy either slot; its exact manifest chain
    /// remains retained by the admitted snapshot.
    /// Persist the prepared attempt before publishing and reconcile unknown
    /// outcomes through the ordinary publisher. A competing publication causes
    /// the existing expected-pair fence to refuse this proposal; it never
    /// authorizes rebasing onto a different current generation.
    ///
    /// # Errors
    ///
    /// Refuses cancellation, absent historical provenance, unsupported security
    /// posture, inconsistent physical slots or ancestry provenance, authority
    /// sequence exhaustion, and invalid candidate identities.
    pub fn rollback_selected_predecessor_candidate(
        &self,
        cx: &Cx,
        object_id: [u8; 16],
        writer_fence_sha256: [u8; 32],
    ) -> SearchResult<GenerationCandidateV1> {
        checkpoint(cx, "native_ann.generation.selected_rollback")?;
        let target = self.retained_snapshot();
        let current = target
            .selected_history_of()
            .ok_or_else(|| refused("selected_rollback", "missing-selection-provenance"))?;
        if target.profile() != GenerationRootSecurityProfileV1::CooperativeLocal
            || target.floor().is_some()
            || target.health()
                != (SnapshotHealthV1::Unanchored {
                    profile: GenerationRootSecurityProfileV1::CooperativeLocal,
                })
        {
            return Err(refused("selected_rollback", "security-profile"));
        }
        let pair = target.authority_pair();
        let (Some(first), Some(second)) = (pair.first, pair.second) else {
            return Err(refused("selected_rollback", "incomplete-authority-pair"));
        };
        if first.slot_index != 0
            || second.slot_index != 1
            || first.root_id != target.root_id()
            || second.root_id != target.root_id()
        {
            return Err(refused("selected_rollback", "invalid-authority-pair"));
        }
        let head = resolve_authority_slots_v1(pair.first, pair.second)
            .map_err(|_| refused("selected_rollback", "invalid-authority-pair"))?
            .ok_or_else(|| refused("selected_rollback", "incomplete-authority-pair"))?;
        if head.authority != current || self.authority().sequence >= current.sequence {
            return Err(refused("selected_rollback", "selection-pair-mismatch"));
        }
        if target.selected_predecessor_of().is_some() {
            let target_slot = target
                .head()
                .slot
                .ok_or_else(|| refused("selected_rollback", "predecessor-slot-missing"))?;
            if head.slot_index == target_slot.slot_index
                || ![first, second].contains(&target_slot)
                || self.authority().sequence.checked_add(1) != Some(current.sequence)
                || current.predecessor != Some(self.authority().fingerprint())
            {
                return Err(refused("selected_rollback", "selection-pair-mismatch"));
            }
        } else if target.head().slot.is_some()
            || (target.history_current().is_none() && target.recovery_current_manifest().is_none())
            || u64::try_from(target.history_manifests().len()).ok()
                != current.sequence.checked_sub(self.authority().sequence + 1)
        {
            return Err(refused("selected_rollback", "missing-ancestry-provenance"));
        }
        current
            .next_sequence()
            .map_err(|_| refused("selected_rollback", "sequence-exhausted"))?;
        GenerationCandidateV1::new(
            object_id,
            Some(current),
            GenerationAuthorityActionV1::Rollback,
            target.manifest().generation(),
            writer_fence_sha256,
            &self.readers.components,
        )
        .map_err(|_| refused("selected_rollback", "invalid-candidate"))
    }

    /// Search and hydrate against the selected vector and Quill images.
    ///
    /// The configured producer is checked before inference and its bound output
    /// is checked again before scoring. Quality-only generations run the quality
    /// lane directly; they do not require or invent a fast tier. In a complete
    /// progressive generation this explicitly queries the primary fast tier;
    /// use [`Self::search_refined_text`] to retrieve both tiers.
    ///
    /// # Errors
    ///
    /// Propagates typed producer/embedding refusal, search/hydration failure and
    /// cancellation. A failed semantic arm never becomes lexical-only success.
    pub async fn search_text(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        self.readers.search_text(cx, embedder, text, k).await
    }

    /// Lazily query one explicitly selected vector tier plus retained lexical
    /// data, with optional source-bound cross-encoder reranking.
    ///
    /// Only the requested owner and producer participate. Quality-only requests
    /// carry quality scores and candidate counts in both Initial and Reranked;
    /// no fast inference or synthetic refinement occurs. The optional caller
    /// pool owns vector traversal. Dropping after Initial starts no reranking,
    /// and later publication cannot change this query's source or owner.
    ///
    /// # Errors
    /// Refuses an absent tier, producer disagreement, invalid budgets and
    /// cancellation before retrieval, including empty and zero-limit queries.
    /// Phase execution can additionally fail source admission, inference,
    /// traversal, hydration or the configured reranker.
    pub fn search_tier_progressive<'a>(
        &'a self,
        cx: &'a Cx,
        tier: EmbedderTierTag,
        embedder: &'a dyn Embedder,
        text: &'a str,
        k: usize,
        pool: Option<BlockingPoolHandle>,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        checkpoint(cx, "native_ann.generation.tier_query")?;
        require_query_tier(self.topology(), tier)?;
        self.readers.primary_query(
            cx,
            embedder,
            text,
            k,
            tier == EmbedderTierTag::Quality,
            pool,
        )
    }

    /// Prepare a fast Initial page and subsequent quality refinement while the
    /// quality model is still cold. Both vector owners and the lexical reader
    /// come from this exact snapshot. On partial generations only covered
    /// quality rows participate in refinement. After Initial, bind the loaded
    /// provider with [`NativeProgressiveSearch::bind_deferred_quality`].
    ///
    /// # Errors
    /// Refuses a generation without both tiers, a mismatched fast provider,
    /// invalid budgets, or cancellation.
    pub fn search_progressive_with_deferred_quality<'a>(
        &'a self,
        cx: &'a Cx,
        fast: &'a dyn Embedder,
        text: &'a str,
        k: usize,
        pool: Option<BlockingPoolHandle>,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        let quality = self.deferred_quality_owner()?;
        Ok(self
            .search_tier_progressive(cx, EmbedderTierTag::Fast, fast, text, k, pool)?
            .with_deferred_quality(quality))
    }

    pub(super) fn deferred_quality_owner(&self) -> SearchResult<&Arc<NativeAnnIndex>> {
        match self.topology() {
            NativeGenerationTopology::PartialQuality { .. } => self.readers.partial_quality(),
            _ => self.readers.full_quality(),
        }
    }

    /// Search the retained primary tier with vector traversal on the caller's
    /// pool. The job owns the admitted vector image; later publications cannot
    /// alter it. Inference, lexical scoring and hydration stay on their ordinary
    /// async paths, and no pool or runtime is created by this handle.
    ///
    /// # Errors
    /// Has the producer, retrieval and cancellation refusals of
    /// [`Self::search_text`], plus pool failures without inline fallback.
    pub async fn search_text_with_pool(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        text: &str,
        k: usize,
        pool: BlockingPoolHandle,
    ) -> SearchResult<Vec<ScoredResult>> {
        self.readers
            .search_text_with_pool(cx, embedder, text, k, pool)
            .await
    }

    /// Retrieve and hydrate quality results without invoking the fast provider.
    ///
    /// # Errors
    /// Refuses missing quality, changed producer, invalid embeddings or cancellation.
    pub async fn search_quality_text(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        self.readers
            .search_quality_text(cx, embedder, text, k)
            .await
    }

    /// Query the independent quality owner with traversal on the caller's pool.
    /// Fast inference is never required, including in a full generation.
    ///
    /// # Errors
    /// Has the refusals of [`Self::search_quality_text`], plus pool failures.
    /// A failed worker never becomes inline work or a partial-success fallback.
    pub async fn search_quality_text_with_pool(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        text: &str,
        k: usize,
        pool: BlockingPoolHandle,
    ) -> SearchResult<Vec<ScoredResult>> {
        self.readers
            .search_quality_text_with_pool(cx, embedder, text, k, pool)
            .await
    }

    /// Retrieve each tier independently and fuse their candidate union.
    /// Quality may promote a document absent from the fast candidate window.
    ///
    /// # Errors
    /// Requires both admitted owners and providers, including for zero results;
    /// propagates inference, retrieval, hydration and cancellation errors.
    pub async fn search_refined_text(
        &self,
        cx: &Cx,
        fast: &dyn Embedder,
        quality: &dyn Embedder,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        self.readers
            .search_refined_text(cx, fast, quality, text, k)
            .await
    }

    /// Retrieve both independently admitted tiers on the caller's pool before
    /// fusing and hydrating their complete candidate union.
    ///
    /// # Errors
    /// Has the refusals of [`Self::search_refined_text`], plus pool failures.
    /// Both arms must succeed; cancellation never delivers a partial page.
    pub async fn search_refined_text_with_pool(
        &self,
        cx: &Cx,
        fast: &dyn Embedder,
        quality: &dyn Embedder,
        text: &str,
        k: usize,
        pool: BlockingPoolHandle,
    ) -> SearchResult<Vec<ScoredResult>> {
        self.readers
            .search_refined_text_with_pool(cx, fast, quality, text, k, pool)
            .await
    }

    /// Lazily emit Initial and Refined results from this same retained generation.
    /// Both provider identities are admitted before inference; quality work starts
    /// only when the caller requests refinement. Dropping the stream preserves
    /// the original immutable readers and starts no detached work.
    /// The returned query accepts fusion, candidate and beam tuning, and
    /// [`NativeProgressiveSearch::with_retained_reranker`] adds bounded, lazy
    /// reranking from this exact generation's stored source.
    ///
    /// # Errors
    /// Refuses absent tiers, producer disagreement, invalid budgets and cancellation.
    pub fn search_progressive_with_quality<'a>(
        &'a self,
        cx: &'a Cx,
        fast: &'a dyn Embedder,
        quality: &'a dyn Embedder,
        text: &'a str,
        k: usize,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        self.readers
            .search_progressive_with_quality(cx, fast, quality, text, k)
    }

    /// Lazily emit phases whose vector traversal runs on the caller's pool.
    /// Both admitted owners remain pinned to this generation, quality starts
    /// only on refinement, and dropping pending work abandons its queued job.
    /// Ordinary tuning and retained-source reranking remain available.
    ///
    /// # Errors
    /// Has the admission refusals of [`Self::search_progressive_with_quality`].
    /// Polling a phase also propagates pool, retrieval and cancellation failures.
    pub fn search_progressive_with_quality_and_pool<'a>(
        &'a self,
        cx: &'a Cx,
        fast: &'a dyn Embedder,
        quality: &'a dyn Embedder,
        text: &'a str,
        k: usize,
        pool: BlockingPoolHandle,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        self.readers
            .search_progressive_with_quality_and_pool(cx, fast, quality, text, k, pool)
    }

    /// Explicitly refine the complete fast corpus with its admitted quality subset.
    ///
    /// This is an opt-in partial-quality request. Existing full-progressive
    /// methods continue to require complete quality coverage. Both independent
    /// producers are admitted before inference, uncovered fast candidates remain
    /// eligible, and quality can contribute rows absent from the fast window.
    /// The returned query supports ordinary tuning and retained-source reranking.
    /// [`Self::topology`] reports exact stored counts and quality coverage.
    ///
    /// # Errors
    /// Refuses a non-partial topology, producer disagreement, invalid budgets
    /// and cancellation, including empty and zero-result queries.
    pub fn search_progressive_with_partial_quality<'a>(
        &'a self,
        cx: &'a Cx,
        fast: &'a dyn Embedder,
        quality: &'a dyn Embedder,
        text: &'a str,
        k: usize,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        self.readers
            .search_progressive_with_partial_quality(cx, fast, quality, text, k, None)
    }

    /// Run explicit partial refinement with vector traversal on the caller's pool.
    /// Both owners remain retained while a worker runs; quality stays lazy.
    ///
    /// # Errors
    /// Has the refusals of [`Self::search_progressive_with_partial_quality`],
    /// plus pool errors without inline or full-coverage fallback.
    pub fn search_progressive_with_partial_quality_and_pool<'a>(
        &'a self,
        cx: &'a Cx,
        fast: &'a dyn Embedder,
        quality: &'a dyn Embedder,
        text: &'a str,
        k: usize,
        pool: BlockingPoolHandle,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        self.readers
            .search_progressive_with_partial_quality(cx, fast, quality, text, k, Some(pool))
    }

    /// Search the retained Quill image without loading or invoking a model.
    ///
    /// This is an explicit keyword request over the already fully admitted
    /// authority, not a fallback from failed semantic retrieval. Every result
    /// belongs to the common live document cohort of all admitted vector owners.
    /// Scores and ordering are Quill's raw lexical results; no semantic fusion,
    /// vector address, or semantic score is reported. Only winning documents
    /// are hydrated, using the same lexical snapshot that produced them.
    ///
    /// Both single-tier and complete progressive generations are supported.
    /// A zero limit returns no results after checking cancellation. Larger
    /// limits are bounded by the admitted live cohort. No source corpus is
    /// exported, authority followed, or component pathname reopened.
    ///
    /// # Errors
    /// Propagates lexical query, hydration and cancellation failures, and
    /// refuses inconsistent scores, duplicate or nonlive IDs, or hydration
    /// that changes a winner's identity or score.
    pub async fn search_lexical_text(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        self.readers
            .search_lexical_with_reader(cx, &self.readers.lexical, text, k)
            .await
    }

    // Only the closed scope adapter may replace the lexical view; its reader
    // and membership retain this same immutable authority and live cohort.
    pub(in crate::native_ann) async fn scoped_lexical_text(
        &self,
        cx: &Cx,
        lexical: &dyn LexicalRead,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        self.readers
            .search_lexical_with_reader(cx, lexical, text, k)
            .await
    }

    // Only the closed source-scope adapter may wrap this immutable lexical
    // reader. Its source inventory is exported from these same admitted images.
    pub(in crate::native_ann) fn scoped_lexical(&self) -> &dyn LexicalRead {
        &self.readers.lexical
    }

    pub(in crate::native_ann) fn scope_readers(&self) -> &CompositeReaders {
        &self.readers
    }

    // Keep the raw vector owner private while using the ordinary phase engine's
    // identity admission, filtered traversal and winner hydration. Clamp before
    // candidate-budget multiplication, including for empty scopes and usize::MAX.
    pub(in crate::native_ann) fn scoped_progressive<'a>(
        &'a self,
        cx: &'a Cx,
        embedder: &'a dyn Embedder,
        lexical: &'a dyn LexicalRead,
        text: &'a str,
        k: usize,
        allowed: Arc<BTreeSet<String>>,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        self.readers
            .scoped_progressive(cx, embedder, None, false, lexical, text, k, allowed, None)
    }

    pub(in crate::native_ann) fn scoped_quality_progressive<'a>(
        &'a self,
        cx: &'a Cx,
        embedder: &'a dyn Embedder,
        lexical: &'a dyn LexicalRead,
        text: &'a str,
        k: usize,
        allowed: Arc<BTreeSet<String>>,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        self.readers
            .scoped_progressive(cx, embedder, None, true, lexical, text, k, allowed, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::native_ann) fn scoped_progressive_with_quality<'a>(
        &'a self,
        cx: &'a Cx,
        fast: &'a dyn Embedder,
        quality: &'a dyn Embedder,
        lexical: &'a dyn LexicalRead,
        text: &'a str,
        k: usize,
        allowed: Arc<BTreeSet<String>>,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        self.readers.scoped_progressive(
            cx,
            fast,
            Some(quality),
            false,
            lexical,
            text,
            k,
            allowed,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::native_ann) fn scoped_progressive_on_pool<'a>(
        &'a self,
        cx: &'a Cx,
        primary: &'a dyn Embedder,
        quality: Option<&'a dyn Embedder>,
        quality_only: bool,
        lexical: &'a dyn LexicalRead,
        text: &'a str,
        k: usize,
        allowed: Arc<BTreeSet<String>>,
        pool: BlockingPoolHandle,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        self.readers.scoped_progressive(
            cx,
            primary,
            quality,
            quality_only,
            lexical,
            text,
            k,
            allowed,
            Some(pool),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::native_ann) fn scoped_partial_quality_progressive<'a>(
        &'a self,
        cx: &'a Cx,
        fast: &'a dyn Embedder,
        quality: &'a dyn Embedder,
        lexical: &'a dyn LexicalRead,
        text: &'a str,
        k: usize,
        allowed: Arc<BTreeSet<String>>,
        pool: Option<BlockingPoolHandle>,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        self.readers.scoped_progressive_with_coverage(
            cx,
            fast,
            Some(quality),
            false,
            lexical,
            text,
            k,
            allowed,
            pool,
            VectorCohortCoverage::PartialQuality,
        )
    }
}

// Directory contents and control bytes legitimately change on publication.
// Compare only the descriptor-derived security identities, never derived
// witness equality (which also includes timestamps and directory length).
fn same_root_security_identity(
    first: GenerationRootObjectWitness,
    second: GenerationRootObjectWitness,
) -> bool {
    first.device() == second.device()
        && first.inode() == second.inode()
        && first.mode() == second.mode()
        && first.uid() == second.uid()
        && first.gid() == second.gid()
        && first.mount_identity() == second.mount_identity()
        && first.filesystem() == second.filesystem()
}

fn same_anchor_security_identity(
    first: GenerationRootObjectWitness,
    second: GenerationRootObjectWitness,
) -> bool {
    same_root_security_identity(first, second)
        && first.hard_links() == second.hard_links()
        && first.byte_len() == second.byte_len()
}

impl CompositeReaders {
    const fn topology(&self) -> NativeGenerationTopology {
        self.topology
    }

    fn require_mutable_topology(&self) -> SearchResult<()> {
        if matches!(
            self.topology(),
            NativeGenerationTopology::PartialQuality { .. }
        ) {
            return Err(refused("update", "partial-quality-mutation-unsupported"));
        }
        Ok(())
    }

    fn full_quality(&self) -> SearchResult<&Arc<NativeAnnIndex>> {
        if !matches!(self.topology(), NativeGenerationTopology::FullProgressive) {
            return Err(refused("topology", "complete-quality-required"));
        }
        self.quality
            .as_ref()
            .ok_or_else(|| refused("topology", "complete-quality-required"))
    }

    fn partial_quality(&self) -> SearchResult<&Arc<NativeAnnIndex>> {
        if !matches!(
            self.topology(),
            NativeGenerationTopology::PartialQuality { .. }
        ) {
            return Err(refused("topology", "explicit-partial-quality-required"));
        }
        self.quality
            .as_ref()
            .ok_or_else(|| refused("topology", "explicit-partial-quality-required"))
    }

    fn partial_quality_document_ids(&self, cx: &Cx) -> SearchResult<Vec<String>> {
        checkpoint(cx, "native_ann.generation.partial_quality_ids")?;
        let quality = self.partial_quality()?;
        let mut ids = Vec::with_capacity(quality.owner.live_count());
        for physical in 0..quality.owner.record_count() {
            checkpoint(cx, "native_ann.generation.partial_quality_id")?;
            let row = quality.owner.row(physical)?;
            if row.flags().is_live() {
                ids.push(row.doc_id().to_owned());
            }
        }
        ids.sort_unstable();
        checkpoint(cx, "native_ann.generation.partial_quality_ids_ready")?;
        Ok(ids)
    }

    fn bounded_limit(&self, requested: usize) -> usize {
        requested.min(usize::try_from(self.vector.owner_witness().live_count).unwrap_or(usize::MAX))
    }

    fn primary_query<'a>(
        &'a self,
        cx: &'a Cx,
        embedder: &'a dyn Embedder,
        text: &'a str,
        k: usize,
        quality_only: bool,
        pool: Option<BlockingPoolHandle>,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        let index = if quality_only && self.tier != EmbedderTierTag::Quality {
            self.full_quality()?
        } else {
            &self.vector
        };
        let search = match pool {
            Some(pool) => index.search_hybrid_progressive_with_pool(
                cx,
                embedder,
                None,
                &self.lexical,
                text,
                self.bounded_limit(k),
                pool,
            )?,
            None => index.search_hybrid_progressive(
                cx,
                embedder,
                None,
                &self.lexical,
                text,
                self.bounded_limit(k),
            )?,
        };
        let search = search.with_retained_metadata(self);
        Ok(if quality_only || self.tier == EmbedderTierTag::Quality {
            search.with_quality_primary()
        } else {
            search
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn scoped_progressive<'a>(
        &'a self,
        cx: &'a Cx,
        primary: &'a dyn Embedder,
        quality: Option<&'a dyn Embedder>,
        quality_only: bool,
        lexical: &'a dyn LexicalRead,
        text: &'a str,
        k: usize,
        allowed: Arc<BTreeSet<String>>,
        pool: Option<BlockingPoolHandle>,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        self.scoped_progressive_with_coverage(
            cx,
            primary,
            quality,
            quality_only,
            lexical,
            text,
            k,
            allowed,
            pool,
            VectorCohortCoverage::Complete,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn scoped_progressive_with_coverage<'a>(
        &'a self,
        cx: &'a Cx,
        primary: &'a dyn Embedder,
        quality: Option<&'a dyn Embedder>,
        quality_only: bool,
        lexical: &'a dyn LexicalRead,
        text: &'a str,
        k: usize,
        allowed: Arc<BTreeSet<String>>,
        pool: Option<BlockingPoolHandle>,
        coverage: VectorCohortCoverage,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        checkpoint(cx, "native_ann.generation.scoped_search")?;
        let index = if quality_only && self.tier != EmbedderTierTag::Quality {
            self.full_quality()?
        } else {
            &self.vector
        };
        let quality_index = match coverage {
            VectorCohortCoverage::Complete => quality.map(|_| self.full_quality()).transpose()?,
            VectorCohortCoverage::PartialQuality => {
                if quality.is_none() || quality_only {
                    return Err(refused("topology", "explicit-partial-quality-required"));
                }
                Some(self.partial_quality()?)
            }
        };
        let refinement = quality_index.zip(quality);
        let k = self.bounded_limit(k).min(allowed.len());
        let search = match pool {
            Some(pool) => index.search_hybrid_progressive_with_pool(
                cx, primary, refinement, lexical, text, k, pool,
            )?,
            None => index.search_hybrid_progressive(
                cx,
                primary,
                refinement.map(|(index, provider)| (index.as_ref(), provider)),
                lexical,
                text,
                k,
            )?,
        };
        let search = search.with_retained_metadata(self);
        let search = if quality_only || self.tier == EmbedderTierTag::Quality {
            search.with_quality_primary()
        } else {
            search
        };
        search.with_allowed_documents_arc(allowed)
    }

    async fn search_lexical_with_reader(
        &self,
        cx: &Cx,
        lexical: &dyn LexicalRead,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        checkpoint(cx, "native_ann.generation.lexical_search")?;
        let k = self.bounded_limit(k);
        if k == 0 {
            return Ok(Vec::new());
        }
        let response = with_cancellation(cx, "native_ann.generation.lexical_response", async {
            super::hybrid::immutable_candidates(cx, lexical, text, k).await
        })
        .await;
        checkpoint(cx, "native_ann.generation.lexical_response")?;
        let batch = response?;
        if batch.results().len() > k {
            return Err(refused("lexical-search", "overfull-response"));
        }
        let mut seen = BTreeSet::new();
        for hit in batch.results() {
            checkpoint(cx, "native_ann.generation.lexical_candidate")?;
            self.validate_lexical_hit(hit)?;
            if !seen.insert(hit.doc_id.as_str()) {
                return Err(refused("lexical-search", "duplicate-document"));
            }
        }
        drop(seen);
        let (mut results, context) = batch.into_parts();
        let identities: Vec<_> = results
            .iter()
            .map(|hit| (hit.doc_id.clone(), hit.score.to_bits()))
            .collect();
        let hydrated = with_cancellation(cx, "native_ann.generation.lexical_hydrated", async {
            lexical
                .hydrate_candidates(cx, context.as_ref(), &mut results)
                .await
        })
        .await;
        checkpoint(cx, "native_ann.generation.lexical_hydrated")?;
        hydrated?;
        for (hit, (id, score)) in results.iter_mut().zip(identities) {
            checkpoint(cx, "native_ann.generation.lexical_result")?;
            if hit.doc_id != id || hit.score.to_bits() != score {
                return Err(refused("lexical-search", "hydration-changed-winner"));
            }
            self.validate_lexical_hit(hit)?;
            // A lexical document coordinate is not a physical vector address.
            hit.index = None;
        }
        checkpoint(cx, "native_ann.generation.lexical_complete")?;
        Ok(results)
    }

    fn validate_lexical_hit(&self, hit: &ScoredResult) -> SearchResult<()> {
        if !hit.score.is_finite()
            || hit.lexical_score.map(f32::to_bits) != Some(hit.score.to_bits())
            || hit.source != ScoreSource::Lexical
            || hit.fast_score.is_some()
            || hit.quality_score.is_some()
            || hit.rerank_score.is_some()
        {
            return Err(refused("lexical-search", "invalid-score-provenance"));
        }
        if self
            .vector
            .owner
            .find_live_row_by_doc_id(hit.doc_id.as_str())
            .map_err(|_| refused("lexical-search", "invalid-vector-row"))?
            .is_none()
        {
            return Err(refused("lexical-search", "nonlive-document"));
        }
        Ok(())
    }

    fn source_context(&self, cx: &Cx) -> SearchResult<(Arc<KeeperSnapshot>, usize, [u16; 4])> {
        checkpoint(cx, "native_ann.generation.source")?;
        let snapshot = self
            .lexical
            .snapshot()
            .map_err(|_| refused("source", "lexical-snapshot"))?;
        if snapshot.schema() != DEFAULT_SCHEMA {
            return Err(refused("source", "unsupported-schema"));
        }
        let live_count = self.vector.owner_witness().live_count;
        if snapshot.doc_count() != live_count || self.metadata.total_documents != live_count {
            return Err(refused("source", "live-count-mismatch"));
        }
        let live_count =
            usize::try_from(live_count).map_err(|_| refused("source", "live-count-overflow"))?;
        let mut field_ids = [0; 4];
        for (field_id, name) in
            field_ids
                .iter_mut()
                .zip(["id", "content", "title", "metadata_json"])
        {
            *field_id = DEFAULT_SCHEMA
                .fields
                .iter()
                .find(|field| field.name == name && field.stored)
                .ok_or_else(|| refused("source", "unsupported-schema"))?
                .id;
        }
        Ok((snapshot, live_count, field_ids))
    }

    // Generic lexical hydration intentionally ignores vector-only winners.
    // This closed reader also owns the admitted common source cohort, so it
    // can recover their metadata without inventing a lexical rank or reopening
    // a component path. Read only the winning metadata column, not source bodies.
    // Search accepts the caller's admitted lexical schema; the default-schema
    // contract for exporting complete IndexableDocuments must not restrict it.
    pub(in crate::native_ann) fn hydrate_vector_metadata(
        &self,
        cx: &Cx,
        results: &mut [ScoredResult],
    ) -> SearchResult<()> {
        if results.iter().all(|result| result.lexical_score.is_some()) {
            return Ok(());
        }
        checkpoint(cx, "native_ann.generation.vector_metadata")?;
        let snapshot = self
            .lexical
            .snapshot()
            .map_err(|_| refused("source", "lexical-snapshot"))?;
        let live_count = self.vector.owner_witness().live_count;
        if snapshot.doc_count() != live_count || self.metadata.total_documents != live_count {
            return Err(refused("source", "live-count-mismatch"));
        }
        let schema = snapshot.schema();
        // Schema names are diagnostic, not part of persisted schema identity.
        // Renaming the default layout must not relax its source validation.
        let canonical_source_metadata = schema.fields == DEFAULT_SCHEMA.fields;
        let metadata_field = schema
            .fields
            .iter()
            .find(|field| field.name == "metadata_json" && field.stored)
            .map(|field| field.id);
        let mut staged = Vec::new();
        for (position, result) in results.iter().enumerate() {
            if result.lexical_score.is_some() {
                continue;
            }
            checkpoint(cx, "native_ann.generation.vector_metadata")?;
            if self
                .vector
                .owner
                .find_live_row_by_doc_id(result.doc_id.as_str())
                .map_err(|_| refused("source", "invalid-vector-row"))?
                .is_none()
            {
                return Err(refused("source", "missing-live-document"));
            }
            let document = snapshot
                .resolve_document_id(result.doc_id.as_str())
                .map_err(|_| refused("source", "invalid-lexical-identity"))?
                .ok_or_else(|| refused("source", "missing-live-document"))?;
            let bytes = metadata_field
                .map(|field| {
                    self.lexical
                        .stored_field_value(field, document.global_docid)
                        .map_err(|_| refused("source", "invalid-stored-field"))
                })
                .transpose()?
                .flatten();
            let metadata = if canonical_source_metadata {
                // Preserve the canonical source contract of the shipping
                // schema, including missing fields and duplicate JSON keys.
                let bytes = bytes.ok_or_else(|| refused("source", "missing-stored-field"))?;
                let metadata: BTreeMap<String, String> = serde_json::from_slice(&bytes)
                    .map_err(|_| refused("source", "invalid-metadata"))?;
                if serde_json::to_vec(&metadata)
                    .map_err(|_| refused("source", "invalid-metadata"))?
                    != bytes
                {
                    return Err(refused("source", "noncanonical-metadata"));
                }
                if metadata.is_empty() {
                    None
                } else {
                    Some(Arc::new(
                        serde_json::to_value(metadata)
                            .map_err(|_| refused("source", "invalid-metadata"))?,
                    ))
                }
            } else {
                // Schema-document ingestion may store arbitrary JSON and may
                // omit this column altogether. Match Quill lexical hydration
                // rather than interpreting an unrelated field as metadata.
                bytes
                    .map(|bytes| {
                        serde_json::from_slice::<serde_json::Value>(&bytes)
                            .map_err(|_| refused("source", "invalid-metadata"))
                    })
                    .transpose()?
                    .filter(|metadata| !metadata.as_object().is_some_and(serde_json::Map::is_empty))
                    .map(Arc::new)
            };
            staged.push((position, metadata));
        }
        checkpoint(cx, "native_ann.generation.vector_metadata_commit")?;
        for (position, metadata) in staged {
            results[position].metadata = metadata;
        }
        Ok(())
    }

    pub(super) fn source_documents(
        &self,
        cx: &Cx,
        max_source_bytes: usize,
    ) -> SearchResult<Vec<IndexableDocument>> {
        self.scan_source_documents(cx, None, max_source_bytes)
    }

    fn scan_source_documents(
        &self,
        cx: &Cx,
        id_prefix: Option<&str>,
        max_source_bytes: usize,
    ) -> SearchResult<Vec<IndexableDocument>> {
        let prefix = id_prefix.unwrap_or_default();
        let (snapshot, live_count, field_ids) = self.source_context(cx)?;
        // Every admitted FSVI identifier is nonempty. Reject an impossible
        // budget before reserving the output collection.
        let mut documents = Vec::new();
        if prefix.is_empty() {
            if live_count > max_source_bytes {
                return Err(refused("source", "stored-payload-limit"));
            }
            documents
                .try_reserve_exact(live_count)
                .map_err(|_| refused("source", "allocation-failed"))?;
        }
        let mut remaining_bytes = max_source_bytes;
        self.visit_live_ids(cx, |id| {
            if id.starts_with(prefix) {
                documents.push(self.read_source_document(
                    cx,
                    &snapshot,
                    field_ids,
                    id,
                    &mut remaining_bytes,
                )?);
            }
            Ok(())
        })?;
        Ok(documents)
    }

    pub(super) fn visit_live_ids(
        &self,
        cx: &Cx,
        mut visit: impl FnMut(&str) -> SearchResult<()>,
    ) -> SearchResult<()> {
        let mut seen_live = 0;
        for physical_row in 0..self.vector.owner.record_count() {
            checkpoint(cx, "native_ann.generation.source_row")?;
            let row = self
                .vector
                .owner
                .row(physical_row)
                .map_err(|_| refused("source", "invalid-vector-row"))?;
            if row.flags().is_live() {
                seen_live += 1;
                visit(row.doc_id())?;
            }
        }
        if seen_live != self.metadata.total_documents {
            return Err(refused("source", "live-count-mismatch"));
        }
        checkpoint(cx, "native_ann.generation.source_complete")
    }

    pub(super) fn has_live_document(&self, id: &str) -> SearchResult<bool> {
        self.vector
            .owner
            .find_live_row_by_doc_id(id)
            .map(|row| row.is_some())
            .map_err(|_| refused("source", "invalid-vector-row"))
    }

    /// Resolve exact IDs directly when available, otherwise inspect live vector
    /// row IDs. Only the intersection with the literal prefix may read source.
    pub(super) fn scope_documents(
        &self,
        cx: &Cx,
        ids: Option<&BTreeSet<String>>,
        id_prefix: Option<&str>,
        max_source_bytes: usize,
    ) -> SearchResult<Vec<IndexableDocument>> {
        let Some(ids) = ids else {
            return self.scan_source_documents(cx, id_prefix, max_source_bytes);
        };
        let (snapshot, _, fields) = self.source_context(cx)?;
        let prefix = id_prefix.unwrap_or_default();
        let mut remaining = max_source_bytes;
        let mut documents = Vec::new();
        for id in ids {
            checkpoint(cx, "native_ann.generation.scope_source_id")?;
            if id.starts_with(prefix) && self.has_live_document(id)? {
                documents.push(self.read_source_document(
                    cx,
                    &snapshot,
                    fields,
                    id,
                    &mut remaining,
                )?);
            }
        }
        checkpoint(cx, "native_ann.generation.scope_source_complete")?;
        Ok(documents)
    }

    fn source_documents_by_ids(
        &self,
        cx: &Cx,
        doc_ids: &[&str],
        max_source_bytes: usize,
    ) -> SearchResult<Vec<Option<IndexableDocument>>> {
        self.read_selected_sources(cx, doc_ids, max_source_bytes, None, std::convert::identity)
    }

    fn read_selected_sources<T>(
        &self,
        cx: &Cx,
        doc_ids: &[&str],
        max_source_bytes: usize,
        content_chars: Option<usize>,
        project: impl Fn(IndexableDocument) -> T,
    ) -> SearchResult<Vec<Option<T>>> {
        let (snapshot, _, field_ids) = self.source_context(cx)?;
        let mut documents = Vec::new();
        documents
            .try_reserve_exact(doc_ids.len())
            .map_err(|_| refused("source", "allocation-failed"))?;
        let mut remaining_bytes = max_source_bytes;
        for doc_id in doc_ids {
            checkpoint(cx, "native_ann.generation.source_row")?;
            if !self.has_live_document(doc_id)? {
                documents.push(None);
                continue;
            }
            let document = self.read_source_fields(
                cx,
                &snapshot,
                field_ids,
                doc_id,
                &mut remaining_bytes,
                content_chars,
            )?;
            documents.push(Some(project(document)));
        }
        checkpoint(cx, "native_ann.generation.source_complete")?;
        Ok(documents)
    }

    fn read_source_document(
        &self,
        cx: &Cx,
        snapshot: &KeeperSnapshot,
        field_ids: [u16; 4],
        doc_id: &str,
        remaining_bytes: &mut usize,
    ) -> SearchResult<IndexableDocument> {
        self.read_source_fields(cx, snapshot, field_ids, doc_id, remaining_bytes, None)
    }

    fn read_source_fields(
        &self,
        cx: &Cx,
        snapshot: &KeeperSnapshot,
        field_ids: [u16; 4],
        doc_id: &str,
        remaining_bytes: &mut usize,
        content_chars: Option<usize>,
    ) -> SearchResult<IndexableDocument> {
        let resolved = snapshot
            .resolve_document_id(doc_id)
            .map_err(|_| refused("source", "invalid-lexical-identity"))?
            .ok_or_else(|| refused("source", "missing-live-document"))?;
        let mut values: [Vec<u8>; 4] = std::array::from_fn(|_| Vec::new());
        for (value, field_id) in values.iter_mut().zip(field_ids) {
            checkpoint(cx, "native_ann.generation.source_field")?;
            let bytes = match content_chars {
                Some(chars) if field_id == field_ids[1] => self.lexical.stored_text_prefix_bounded(
                    field_id,
                    resolved.global_docid,
                    chars,
                    *remaining_bytes,
                ),
                _ => self.lexical.stored_field_value_bounded(
                    field_id,
                    resolved.global_docid,
                    *remaining_bytes,
                ),
            }
            .map_err(|error| match error {
                QuillIndexError::StoredFieldBudgetExceeded { .. } => {
                    refused("source", "stored-payload-limit")
                }
                _ => refused("source", "invalid-stored-field"),
            })?
            .ok_or_else(|| refused("source", "missing-stored-field"))?;
            *remaining_bytes = remaining_bytes
                .checked_sub(bytes.len())
                .ok_or_else(|| refused("source", "stored-payload-limit"))?;
            *value = bytes;
        }
        checkpoint(cx, "native_ann.generation.source_decode")?;
        let [id, content, title, metadata_json] = values;
        let id = String::from_utf8(id).map_err(|_| refused("source", "invalid-stored-utf8"))?;
        if id != doc_id {
            return Err(refused("source", "stored-id-mismatch"));
        }
        let content =
            String::from_utf8(content).map_err(|_| refused("source", "invalid-stored-utf8"))?;
        let title =
            String::from_utf8(title).map_err(|_| refused("source", "invalid-stored-utf8"))?;
        let metadata: BTreeMap<String, String> = serde_json::from_slice(&metadata_json)
            .map_err(|_| refused("source", "invalid-metadata"))?;
        // Compare the canonical encoding to reject duplicate keys, non-string
        // values and any lossy interpretation of the retained metadata source.
        if serde_json::to_vec(&metadata).map_err(|_| refused("source", "invalid-metadata"))?
            != metadata_json
        {
            return Err(refused("source", "noncanonical-metadata"));
        }
        Ok(IndexableDocument {
            id,
            content,
            title: (!title.is_empty()).then_some(title),
            metadata: metadata.into_iter().collect(),
        })
    }

    // Only a closed generation query may request this source. Validate one
    // candidate at a time against the retained common cohort and charge all
    // stored fields to the request's aggregate budget before copying them.
    // An absent or malformed document is an error, never a missing-text skip.
    pub(in crate::native_ann) fn retained_rerank_text(
        &self,
        cx: &Cx,
        doc_id: &str,
        remaining_bytes: &mut usize,
    ) -> SearchResult<String> {
        let (snapshot, _, field_ids) = self.source_context(cx)?;
        if self
            .vector
            .owner
            .find_live_row_by_doc_id(doc_id)
            .map_err(|_| refused("source", "invalid-vector-row"))?
            .is_none()
        {
            return Err(refused("source", "missing-live-document"));
        }
        let document =
            self.read_source_document(cx, &snapshot, field_ids, doc_id, remaining_bytes)?;
        checkpoint(cx, "native_ann.generation.rerank_source_complete")?;
        Ok(document.content)
    }

    async fn search_text(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        checkpoint(cx, "native_ann.generation.search")?;
        let k = self.bounded_limit(k);
        if ParsedQuery::parse(text).has_negations() {
            let mut results = self
                .primary_query(cx, embedder, text, k, false, None)?
                .collect()
                .await?;
            if self.tier == EmbedderTierTag::Quality {
                super::builder::scope::quality_provenance(cx, &mut results)?;
            }
            return Ok(results);
        }
        let mut results = match self.tier {
            EmbedderTierTag::Fast => {
                self.vector
                    .search_hybrid_text(cx, embedder, &self.lexical, text, k)
                    .await
            }
            EmbedderTierTag::Quality => {
                self.vector
                    .search_hybrid_quality_text(cx, embedder, &self.lexical, text, k)
                    .await
            }
        }?;
        self.hydrate_vector_metadata(cx, &mut results)?;
        Ok(results)
    }

    async fn search_text_with_pool(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        text: &str,
        k: usize,
        pool: BlockingPoolHandle,
    ) -> SearchResult<Vec<ScoredResult>> {
        checkpoint(cx, "native_ann.generation.search")?;
        let k = self.bounded_limit(k);
        if ParsedQuery::parse(text).has_negations() {
            let mut results = self
                .primary_query(cx, embedder, text, k, false, Some(pool))?
                .collect()
                .await?;
            if self.tier == EmbedderTierTag::Quality {
                super::builder::scope::quality_provenance(cx, &mut results)?;
            }
            return Ok(results);
        }
        let mut results = match self.tier {
            EmbedderTierTag::Fast => {
                self.vector
                    .search_hybrid_text_with_pool(cx, embedder, &self.lexical, text, k, pool)
                    .await
            }
            EmbedderTierTag::Quality => {
                self.vector
                    .search_hybrid_quality_text_with_pool(
                        cx,
                        embedder,
                        &self.lexical,
                        text,
                        k,
                        pool,
                    )
                    .await
            }
        }?;
        self.hydrate_vector_metadata(cx, &mut results)?;
        Ok(results)
    }

    async fn search_quality_text(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        checkpoint(cx, "native_ann.generation.quality_search")?;
        if ParsedQuery::parse(text).has_negations() {
            let mut results = self
                .primary_query(cx, embedder, text, k, true, None)?
                .collect()
                .await?;
            super::builder::scope::quality_provenance(cx, &mut results)?;
            return Ok(results);
        }
        let quality = if self.tier == EmbedderTierTag::Quality {
            &self.vector
        } else {
            self.full_quality()?
        };
        let mut results = quality
            .search_hybrid_quality_text(cx, embedder, &self.lexical, text, self.bounded_limit(k))
            .await?;
        self.hydrate_vector_metadata(cx, &mut results)?;
        Ok(results)
    }

    async fn search_quality_text_with_pool(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        text: &str,
        k: usize,
        pool: BlockingPoolHandle,
    ) -> SearchResult<Vec<ScoredResult>> {
        checkpoint(cx, "native_ann.generation.quality_search")?;
        if ParsedQuery::parse(text).has_negations() {
            let mut results = self
                .primary_query(cx, embedder, text, k, true, Some(pool))?
                .collect()
                .await?;
            super::builder::scope::quality_provenance(cx, &mut results)?;
            return Ok(results);
        }
        let quality = if self.tier == EmbedderTierTag::Quality {
            &self.vector
        } else {
            self.full_quality()?
        };
        let mut results = quality
            .search_hybrid_quality_text_with_pool(
                cx,
                embedder,
                &self.lexical,
                text,
                self.bounded_limit(k),
                pool,
            )
            .await?;
        self.hydrate_vector_metadata(cx, &mut results)?;
        Ok(results)
    }

    async fn search_refined_text(
        &self,
        cx: &Cx,
        fast: &dyn Embedder,
        quality: &dyn Embedder,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        checkpoint(cx, "native_ann.generation.refined_search")?;
        if ParsedQuery::parse(text).has_negations() {
            return self
                .search_progressive_with_quality(cx, fast, quality, text, k)?
                .collect()
                .await;
        }
        let mut results = self
            .vector
            .search_hybrid_refined_text(
                cx,
                fast,
                (self.full_quality()?.as_ref(), quality),
                &self.lexical,
                text,
                self.bounded_limit(k),
            )
            .await?;
        self.hydrate_vector_metadata(cx, &mut results)?;
        Ok(results)
    }

    async fn search_refined_text_with_pool(
        &self,
        cx: &Cx,
        fast: &dyn Embedder,
        quality: &dyn Embedder,
        text: &str,
        k: usize,
        pool: BlockingPoolHandle,
    ) -> SearchResult<Vec<ScoredResult>> {
        checkpoint(cx, "native_ann.generation.refined_search")?;
        if ParsedQuery::parse(text).has_negations() {
            return self
                .search_progressive_with_quality_and_pool(cx, fast, quality, text, k, pool)?
                .collect()
                .await;
        }
        let mut results = self
            .vector
            .search_hybrid_refined_text_with_pool(
                cx,
                fast,
                (self.full_quality()?, quality),
                &self.lexical,
                text,
                self.bounded_limit(k),
                pool,
            )
            .await?;
        self.hydrate_vector_metadata(cx, &mut results)?;
        Ok(results)
    }

    fn search_progressive_with_quality<'a>(
        &'a self,
        cx: &'a Cx,
        fast: &'a dyn Embedder,
        quality: &'a dyn Embedder,
        text: &'a str,
        k: usize,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        checkpoint(cx, "native_ann.generation.progressive_search")?;
        Ok(self
            .vector
            .search_hybrid_progressive(
                cx,
                fast,
                Some((self.full_quality()?.as_ref(), quality)),
                &self.lexical,
                text,
                self.bounded_limit(k),
            )?
            .with_retained_metadata(self))
    }

    fn search_progressive_with_quality_and_pool<'a>(
        &'a self,
        cx: &'a Cx,
        fast: &'a dyn Embedder,
        quality: &'a dyn Embedder,
        text: &'a str,
        k: usize,
        pool: BlockingPoolHandle,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        checkpoint(cx, "native_ann.generation.progressive_search")?;
        Ok(self
            .vector
            .search_hybrid_progressive_with_pool(
                cx,
                fast,
                Some((self.full_quality()?, quality)),
                &self.lexical,
                text,
                self.bounded_limit(k),
                pool,
            )?
            .with_retained_metadata(self))
    }

    fn search_progressive_with_partial_quality<'a>(
        &'a self,
        cx: &'a Cx,
        fast: &'a dyn Embedder,
        quality: &'a dyn Embedder,
        text: &'a str,
        k: usize,
        pool: Option<BlockingPoolHandle>,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        checkpoint(cx, "native_ann.generation.partial_quality_search")?;
        let quality_index = self.partial_quality()?;
        let search = match pool {
            Some(pool) => self.vector.search_hybrid_progressive_with_pool(
                cx,
                fast,
                Some((quality_index, quality)),
                &self.lexical,
                text,
                self.bounded_limit(k),
                pool,
            )?,
            None => self.vector.search_hybrid_progressive(
                cx,
                fast,
                Some((quality_index.as_ref(), quality)),
                &self.lexical,
                text,
                self.bounded_limit(k),
            )?,
        };
        Ok(search.with_retained_metadata(self))
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn admit_components(
    cx: &Cx,
    activation: &ActivationManifest,
    expected_vector: &FsviV2IdentityBinding,
    schema: SchemaDescriptor,
    config: QuillConfig,
    images: [Arc<[u8]>; 3],
    ann_image: Option<Arc<[u8]>>,
    max_image_bytes: usize,
) -> SearchResult<CompositeReaders> {
    admit_components_with_coverage(
        cx,
        activation,
        expected_vector,
        None,
        schema,
        config,
        images,
        ann_image,
        VectorCohortCoverage::Complete,
        max_image_bytes,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn admit_progressive_components(
    cx: &Cx,
    activation: &ActivationManifest,
    expected_fast: &FsviV2IdentityBinding,
    expected_quality: &FsviV2IdentityBinding,
    schema: SchemaDescriptor,
    config: QuillConfig,
    images: [Arc<[u8]>; 3],
    ann_image: Option<Arc<[u8]>>,
    max_image_bytes: usize,
) -> SearchResult<CompositeReaders> {
    admit_components_with_coverage(
        cx,
        activation,
        expected_fast,
        Some(expected_quality),
        schema,
        config,
        images,
        ann_image,
        VectorCohortCoverage::Complete,
        max_image_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn admit_components_with_coverage(
    cx: &Cx,
    activation: &ActivationManifest,
    expected_vector: &FsviV2IdentityBinding,
    expected_quality: Option<&FsviV2IdentityBinding>,
    schema: SchemaDescriptor,
    config: QuillConfig,
    images: [Arc<[u8]>; 3],
    ann_image: Option<Arc<[u8]>>,
    coverage: VectorCohortCoverage,
    max_image_bytes: usize,
) -> SearchResult<CompositeReaders> {
    checkpoint(cx, "native_ann.generation.components")?;
    if coverage == VectorCohortCoverage::PartialQuality && expected_quality.is_none() {
        return Err(refused(
            "topology",
            "partial-quality-requires-both-producers",
        ));
    }
    let declared = activation.components();
    if declared.ann.is_some() != ann_image.is_some() {
        return Err(refused("ann", "component-presence-mismatch"));
    }
    if activation.generation() != expected_vector.generation()
        || expected_quality.is_some_and(|quality| quality.generation() != activation.generation())
    {
        return Err(refused("generation", "binding-mismatch"));
    }
    let image_bytes = images
        .iter()
        .chain(ann_image.iter())
        .try_fold(activation.canonical_bytes().len(), |total, image| {
            total.checked_add(image.len())
        });
    if image_bytes.is_none_or(|total| total > max_image_bytes) {
        return Err(refused("budget", "encoded-image-limit"));
    }
    let [vector_bytes, lexical_bytes, metadata_bytes] = images;
    for (image, receipt) in [
        (&vector_bytes, declared.vector),
        (&lexical_bytes, declared.lexical),
        (&metadata_bytes, declared.metadata),
    ]
    .into_iter()
    .chain(ann_image.as_ref().zip(declared.ann))
    {
        let mut hasher = sha2::Sha256::new();
        for chunk in image.chunks(1024 * 1024) {
            checkpoint(cx, "native_ann.generation.component_hash")?;
            hasher.update(chunk);
        }
        if u64::try_from(image.len()).ok() != Some(receipt.byte_len)
            || <[u8; 32]>::from(hasher.finalize()) != receipt.sha256
        {
            return Err(refused("component", "image-mismatch"));
        }
    }
    checkpoint(cx, "native_ann.generation.metadata")?;
    let metadata: GenerationManifest =
        serde_json::from_slice(&metadata_bytes).map_err(|_| refused("metadata", "invalid-json"))?;
    let tier_count = if expected_quality.is_some() { 2 } else { 1 };
    if metadata.vector_artifacts.len() != tier_count
        || metadata.lexical_artifacts.len() != 1
        || metadata.embedders.len() != tier_count
        || !metadata.repair_descriptors.is_empty()
    {
        return Err(refused("metadata", "unsupported-component-shape"));
    }
    if !validate_manifest(&metadata).is_valid() {
        return Err(refused("metadata", "invalid-manifest"));
    }
    let declared_coverage = metadata
        .metadata
        .get(NATIVE_QUALITY_COVERAGE_KEY)
        .map(String::as_str);
    if !matches!(
        (coverage, declared_coverage),
        (VectorCohortCoverage::Complete, None)
            | (
                VectorCohortCoverage::PartialQuality,
                Some("partial_quality")
            )
    ) {
        return Err(refused("metadata", "quality-coverage-contract-mismatch"));
    }
    let declared_quality_ids = declared_partial_quality_document_ids(cx, &metadata)?;
    let decode_receipt = |key: &str| {
        metadata
            .metadata
            .get(key)
            .map(|encoded| {
                serde_json::from_str::<NativeHnswGenerationReceiptV2>(encoded)
                    .map_err(|_| refused("ann", "invalid-native-receipt"))
            })
            .transpose()
    };
    let native_receipt = decode_receipt(NATIVE_ANN_RECEIPT_KEY)?;
    let quality_native_receipt = decode_receipt(NATIVE_QUALITY_ANN_RECEIPT_KEY)?;
    if (expected_quality.is_none() && quality_native_receipt.is_some())
        || declared.ann.is_some() != (native_receipt.is_some() || quality_native_receipt.is_some())
    {
        return Err(refused("ann", "native-receipt-presence-mismatch"));
    }
    let checkpoint_value = SourceCheckpointV1::derive(&metadata.commit_range);
    if checkpoint_value.to_bytes() != activation.source_checkpoint_sha256() {
        return Err(refused("checkpoint", "source-mismatch"));
    }
    checkpoint(cx, "native_ann.generation.vector")?;
    let cohort = expected_quality
        .map(|quality| {
            ExactVectorCohortV1::admit_with_coverage(
                cx,
                Arc::clone(&vector_bytes),
                expected_vector,
                quality,
                checkpoint_value,
                coverage,
                max_image_bytes,
            )
        })
        .transpose()?;
    if let Some(identifiers) = declared_quality_ids {
        let quality = cohort
            .as_ref()
            .ok_or_else(|| refused("metadata", "quality-membership-contract-mismatch"))?
            .quality();
        if u64::try_from(identifiers.len()).ok() != Some(quality.witness().live_count) {
            return Err(refused("metadata", "quality-membership-owner-mismatch"));
        }
        for identifier in identifiers {
            checkpoint(cx, "native_ann.generation.quality_membership_owner")?;
            if quality.find_live_row_by_doc_id(&identifier)?.is_none() {
                return Err(refused("metadata", "quality-membership-owner-mismatch"));
            }
        }
    }
    let owner = if let Some(cohort) = &cohort {
        Arc::clone(cohort.fast())
    } else {
        Arc::new(
            ValidatedFsviBytes::from_arc(vector_bytes, expected_vector)
                .map_err(|_| refused("vector", "admission-failed"))?,
        )
    };
    let witness = owner.witness();
    let mut identifiers = Vec::new();
    for row in 0..owner.record_count() {
        checkpoint(cx, "native_ann.generation.document")?;
        let row = owner.row(row)?;
        if row.flags().is_live() {
            identifiers.push(row.doc_id().to_owned());
        }
    }
    let primary_receipt = vector_component_receipt(
        witness,
        identifiers.iter().map(String::as_str),
        checkpoint_value,
    )
    .map_err(|_| refused("vector", "invalid-document-set"))?;
    let vector_receipt = cohort.as_ref().map_or_else(
        || primary_receipt.clone(),
        |cohort| cohort.receipt().clone(),
    );
    if vector_receipt.bytes != declared.vector
        || vector_receipt.docset_digest != activation.document_set_sha256()
    {
        return Err(refused("vector", "selected-document-set-mismatch"));
    }
    let ann_cohort = cohort
        .as_ref()
        .zip(ann_image.as_ref())
        .map(|(cohort, bytes)| {
            NativeAnnCohortV1::admit(
                cx,
                Arc::clone(bytes),
                cohort,
                native_receipt.as_ref(),
                quality_native_receipt.as_ref(),
                max_image_bytes,
            )
        })
        .transpose()?;
    let ann_receipt = if let Some(cohort) = &ann_cohort {
        Some(cohort.receipt().clone())
    } else {
        native_receipt
            .as_ref()
            .map(|receipt| {
                ann_component_receipt(
                    receipt,
                    identifiers.iter().map(String::as_str),
                    checkpoint_value,
                )
                .map_err(|_| refused("ann", "invalid-document-set"))
            })
            .transpose()?
    };
    if ann_receipt.as_ref().map(|receipt| receipt.bytes) != declared.ann {
        return Err(refused("ann", "selected-receipt-mismatch"));
    }
    let documents = CanonicalDocsetV1::from_ordered_live_documents(identifiers)
        .map_err(|_| refused("metadata", "invalid-document-set"))?;
    let metadata_receipt =
        ExactComponentReceiptV1::for_metadata_manifest(&metadata, &metadata_bytes, &documents)
            .map_err(|_| refused("metadata", "document-count-mismatch"))?;
    metadata_receipt
        .verify_metadata_manifest(&metadata, &metadata_bytes)
        .map_err(|_| refused("metadata", "receipt-mismatch"))?;
    validate_artifact_metadata(
        &metadata,
        expected_vector,
        activation,
        &owner,
        &primary_receipt,
        expected_quality
            .zip(cohort.as_ref())
            .map(|(binding, cohort)| {
                (binding, cohort.quality().as_ref(), cohort.quality_receipt())
            }),
    )?;
    let lexical_receipt = ExactComponentReceiptV1 {
        role: GenerationComponentRole::Lexical,
        bytes: declared.lexical,
        docset_digest: activation.document_set_sha256(),
        live_document_count: witness.live_count,
        source_checkpoint: checkpoint_value.to_bytes(),
    };
    checkpoint(cx, "native_ann.generation.lexical")?;
    let lexical_result = KeeperSnapshot::from_exact_image(
        cx,
        lexical_bytes,
        schema,
        checkpoint_value,
        &lexical_receipt,
        max_image_bytes,
    );
    checkpoint(cx, "native_ann.generation.lexical_complete")?;
    let lexical = lexical_result.map_err(|_| refused("lexical", "admission-failed"))?;
    let components = ExactGenerationComponentsV1::admit(
        vector_receipt,
        lexical_receipt,
        ann_receipt,
        metadata_receipt,
    )
    .map_err(|_| refused("components", "incoherent-generation"))?;
    checkpoint(cx, "native_ann.generation.readers")?;
    let (fast_graph, quality_graph) = if let Some(cohort) = ann_cohort {
        cohort.into_graphs()
    } else {
        let graph = match (ann_image, native_receipt.as_ref()) {
            (Some(bytes), Some(receipt)) => Some(
                ValidatedNativeHnsw::from_bytes(Arc::clone(&owner), bytes, receipt)
                    .map_err(|_| refused("ann", "native-admission-failed"))?,
            ),
            (None, None) => None,
            _ => return Err(refused("ann", "native-receipt-presence-mismatch")),
        };
        (graph, None)
    };
    let (vector, retrieval) = retained_tier(cx, owner, fast_graph, native_receipt.as_ref())?;
    let tier = metadata.vector_artifacts[0].embedder_tier;
    let topology = match (cohort.as_ref(), coverage, tier) {
        (Some(cohort), VectorCohortCoverage::PartialQuality, _) => {
            let quality_documents = cohort.quality_receipt().live_document_count;
            let total_documents = cohort.fast_receipt().live_document_count;
            // The admitted subset is strict and nonempty. Keep exact counts
            // alongside the quantized metric, including a sub-one-ppm subset.
            let coverage_ppm = u32::try_from(
                u128::from(quality_documents) * 1_000_000 / u128::from(total_documents),
            )
            .map_err(|_| refused("topology", "quality-coverage-overflow"))?;
            NativeGenerationTopology::PartialQuality {
                quality_documents,
                total_documents,
                coverage_ppm,
            }
        }
        (Some(_), VectorCohortCoverage::Complete, _) => NativeGenerationTopology::FullProgressive,
        (None, _, EmbedderTierTag::Quality) => NativeGenerationTopology::QualityOnly,
        (None, _, EmbedderTierTag::Fast) => NativeGenerationTopology::FastOnly,
    };
    let (quality, quality_retrieval) = if let Some(cohort) = cohort {
        let (index, policy) = retained_tier(
            cx,
            Arc::clone(cohort.quality()),
            quality_graph,
            quality_native_receipt.as_ref(),
        )?;
        (Some(Arc::new(index)), Some(policy))
    } else {
        (None, None)
    };
    let lexical_result = QuillSnapshotIndex::new(cx, lexical, config);
    checkpoint(cx, "native_ann.generation.ready")?;
    let lexical = lexical_result.map_err(|_| refused("lexical", "invalid-search-configuration"))?;
    Ok(CompositeReaders {
        vector: Arc::new(vector),
        quality,
        lexical,
        metadata,
        tier,
        retrieval,
        quality_retrieval,
        components,
        topology,
    })
}

fn retained_tier(
    cx: &Cx,
    owner: Arc<ValidatedFsviBytes>,
    graph: Option<ValidatedNativeHnsw>,
    receipt: Option<&NativeHnswGenerationReceiptV2>,
) -> SearchResult<(NativeAnnIndex, NativeBuildRetrieval)> {
    checkpoint(cx, "native_ann.generation.ann_admitted")?;
    match (graph, receipt) {
        (Some(graph), Some(receipt)) => {
            let dimension =
                |value| usize::try_from(value).map_err(|_| refused("ann", "invalid-parameters"));
            let params = HnswParams {
                m: dimension(receipt.params.m)?,
                m0: dimension(receipt.params.m0)?,
                ef_construction: dimension(receipt.params.ef_construction)?,
                ef_search: dimension(receipt.params.ef_search)?,
            };
            Ok((
                NativeAnnIndex {
                    owner,
                    graph: NativeBackend::Ann(Box::new(graph)),
                    default_ef_search: params.ef_search,
                },
                NativeBuildRetrieval::Hnsw {
                    params,
                    seed: receipt.seed,
                },
            ))
        }
        (None, None) => Ok((
            NativeAnnIndex::exact(cx, owner)?,
            NativeBuildRetrieval::Exact,
        )),
        _ => Err(refused("ann", "native-receipt-presence-mismatch")),
    }
}

fn validate_artifact_metadata(
    metadata: &GenerationManifest,
    binding: &FsviV2IdentityBinding,
    activation: &ActivationManifest,
    owner: &ValidatedFsviBytes,
    vector_receipt: &ExactComponentReceiptV1,
    quality: Option<(
        &FsviV2IdentityBinding,
        &ValidatedFsviBytes,
        &ExactComponentReceiptV1,
    )>,
) -> SearchResult<()> {
    let declared = activation.components();
    let lexical = &metadata.lexical_artifacts[0];
    let live_count = owner.witness().live_count;
    if lexical.document_count != live_count
        || lexical.size_bytes != declared.lexical.byte_len
        || lexical.checksum != lower_hex(&declared.lexical.sha256)
        || metadata.total_documents != live_count
    {
        return Err(refused("metadata", "artifact-binding-mismatch"));
    }
    let mut record_count = 0u64;
    for (position, (binding, owner, receipt)) in std::iter::once((binding, owner, vector_receipt))
        .chain(quality)
        .enumerate()
    {
        let vector = &metadata.vector_artifacts[position];
        let tier = match vector.embedder_tier {
            EmbedderTierTag::Fast => "fast",
            EmbedderTierTag::Quality => "quality",
        };
        if quality.is_some()
            && (vector.embedder_tier
                != if position == 0 {
                    EmbedderTierTag::Fast
                } else {
                    EmbedderTierTag::Quality
                }
                || vector.path != format!("{tier}/{}.vector", lower_hex(&receipt.bytes.sha256)))
        {
            return Err(refused("metadata", "progressive-tier-order"));
        }
        if metadata.embedders.get(tier) != Some(binding.frozen_identity())
            || usize::try_from(vector.dimension).ok() != Some(binding.dimension())
            || vector.vector_count != owner.witness().record_count
            || vector.size_bytes != receipt.bytes.byte_len
            || vector.checksum != lower_hex(&receipt.bytes.sha256)
        {
            return Err(refused("metadata", "artifact-binding-mismatch"));
        }
        record_count = record_count
            .checked_add(owner.witness().record_count)
            .ok_or_else(|| refused("metadata", "record-count-overflow"))?;
    }
    for invariant in &metadata.activation_invariants {
        match &invariant.kind {
            InvariantKind::AllArtifactsVerified | InvariantKind::EmbedderRevisionMatch => {}
            InvariantKind::VectorCountConsistency { expected_total }
                if *expected_total == record_count => {}
            _ => return Err(refused("metadata", "unsupported-activation-predicate")),
        }
    }
    Ok(())
}

fn refused(field: &str, code: &str) -> crate::SearchError {
    invalid(
        &format!("generation.{field}"),
        code,
        "the retained components do not satisfy the selected exact-generation contract",
    )
}
