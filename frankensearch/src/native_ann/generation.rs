//! Native retrieval over one authority-admitted composite generation.
//!
//! All engine readers consume the retained component images. No component is
//! rediscovered through a pathname, and a later authority activation cannot
//! change the bytes used by an already-admitted reader.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use frankensearch_core::generation::{
    ActivationManifest, ArtifactGenerationIdentityV1, AuthorityRefV1, CanonicalDocsetV1,
    EmbedderTierTag, ExactComponentReceiptV1, ExactGenerationComponentsV1,
    GenerationAuthorityActionV1, GenerationComponentRole, GenerationManifest, InvariantKind,
    SourceCheckpointV1, validate_manifest,
};
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
use frankensearch_index::{FsviV2IdentityBinding, ValidatedFsviBytes};
use frankensearch_quill::{
    DEFAULT_SCHEMA, KeeperSnapshot, QuillConfig, QuillSnapshotIndex, SchemaDescriptor,
};
use sha2::Digest as _;

use super::builder::{NativeBuildRetrieval, NativeIndexUpdate};
use super::{
    NativeAnnIndex, NativeBackend, NativeProgressiveSearch, NativeRetrievalMode, checkpoint,
    invalid, lower_hex,
};
use crate::{
    Cx, Embedder, GenerationCandidateV1, IndexableDocument, LexicalRead, ScoredResult, SearchResult,
};

#[cfg(test)]
mod tests;

// The selected metadata image authenticates the full owner-bound native receipt.
// The separately selected ANN component remains the raw graph image, so its
// byte receipt has exactly the existing ann_component_receipt semantics.
pub(super) const NATIVE_ANN_RECEIPT_KEY: &str = "native_hnsw_receipt_v2";

/// Query-capable readers for one fixed-authority, single-tier native generation.
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
/// exact search. Multi-tier/sharded descriptors remain unsupported. Metadata
/// paths are descriptive names, never paths to open. Vector artifact
/// counts describe physical rows, including tombstones; the common document set
/// and metadata total count describe live documents. The existing metadata
/// readiness policy requires at least one live document in this route.
///
/// Keeping this handle retains the original authority capability and all engine
/// bytes. Publishing or rolling back another generation does not change an
/// in-flight or intentionally retained old reader. Obtain a fresh admitted root
/// snapshot and construct another handle to observe a new generation.
/// Explicit predecessor selection also admits a queryable handle, carrying the
/// current authority that authorized that selection as separate provenance. Such
/// a handle may be a rollback target, but cannot act as the current head when
/// proposing a rollback.
pub struct NativeGenerationSnapshot {
    authority: Arc<OpenedGenerationSnapshotV1>,
    readers: CompositeReaders,
}

pub(super) struct CompositeReaders {
    vector: Arc<NativeAnnIndex>,
    lexical: QuillSnapshotIndex,
    metadata: GenerationManifest,
    tier: EmbedderTierTag,
    retrieval: NativeBuildRetrieval,
    components: ExactGenerationComponentsV1,
}

impl NativeGenerationSnapshot {
    /// Admit the actual search components retained by a qualified root snapshot.
    ///
    /// `expected_vector` is the caller's selected generation/embedding/storage
    /// binding. It must not be inferred from the untrusted component it admits.
    /// `schema` identifies the caller's lexical schema. `max_image_bytes` bounds
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

    /// The single vector tier independently declared by the selected metadata.
    #[must_use]
    pub const fn tier(&self) -> EmbedderTierTag {
        self.readers.tier
    }

    /// The actual retrieval mode of this handle.
    #[must_use]
    pub const fn retrieval_mode(&self) -> NativeRetrievalMode {
        match self.readers.retrieval {
            NativeBuildRetrieval::Exact => NativeRetrievalMode::Exact {
                reason: super::NativeExactReason::Requested,
            },
            NativeBuildRetrieval::Hnsw { .. } => NativeRetrievalMode::Ann,
        }
    }

    /// The authenticated construction/search policy retained by this generation.
    /// Rebuilds can preserve its graph parameters and seed without reading paths.
    #[must_use]
    pub const fn retrieval_configuration(&self) -> NativeBuildRetrieval {
        self.readers.retrieval
    }

    /// The validated metadata image belonging to this exact authority.
    #[must_use]
    pub const fn metadata(&self) -> &GenerationManifest {
        &self.readers.metadata
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
    /// including JSON punctuation and escaping. Each field is charged before
    /// decoding. This is a payload budget, not a bound on process RSS or peak
    /// allocation: Quill materializes a field before its size can be checked,
    /// and decoded documents require additional storage. Run large exports on
    /// the caller's bounded blocking lane, as with admission.
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
        NativeIndexUpdate::from_authority(
            cx,
            directory,
            generation,
            provider,
            self,
            max_source_bytes,
            Arc::clone(&self.readers.vector),
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
    /// selected predecessor is usable as `self`, including after the process
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
    /// predecessor used as `current`, a target that is not older than `current`,
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
        if expected.selected_predecessor_of().is_some() {
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

    /// Search and hydrate against the selected vector and Quill images.
    ///
    /// The configured producer is checked before inference and its bound output
    /// is checked again before scoring. Quality-only generations run the quality
    /// lane directly; they do not require or invent a fast tier.
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

    // Only the closed source-scope adapter may wrap this immutable lexical
    // reader. Its source inventory is exported from these same admitted images.
    pub(in crate::native_ann) fn scoped_lexical(&self) -> &dyn LexicalRead {
        &self.readers.lexical
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
        checkpoint(cx, "native_ann.generation.scoped_search")?;
        let live_count =
            usize::try_from(self.readers.vector.owner_witness().live_count).unwrap_or(usize::MAX);
        let k = k.min(live_count).min(allowed.len());
        self.readers
            .vector
            .search_hybrid_progressive(cx, embedder, None, lexical, text, k)?
            .with_allowed_documents_arc(allowed)
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
    pub(super) fn source_documents(
        &self,
        cx: &Cx,
        max_source_bytes: usize,
    ) -> SearchResult<Vec<IndexableDocument>> {
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
        // Every admitted FSVI identifier is nonempty. Reject an impossible
        // budget before reserving the output collection.
        if live_count > max_source_bytes {
            return Err(refused("source", "stored-payload-limit"));
        }
        let mut documents = Vec::new();
        documents
            .try_reserve_exact(live_count)
            .map_err(|_| refused("source", "allocation-failed"))?;
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
        let mut source_bytes = 0usize;
        for physical_row in 0..self.vector.owner.record_count() {
            checkpoint(cx, "native_ann.generation.source_row")?;
            let row = self
                .vector
                .owner
                .row(physical_row)
                .map_err(|_| refused("source", "invalid-vector-row"))?;
            if !row.flags().is_live() {
                continue;
            }
            let resolved = snapshot
                .resolve_document_id(row.doc_id())
                .map_err(|_| refused("source", "invalid-lexical-identity"))?
                .ok_or_else(|| refused("source", "missing-live-document"))?;
            let mut values: [Vec<u8>; 4] = std::array::from_fn(|_| Vec::new());
            for (value, field_id) in values.iter_mut().zip(field_ids) {
                checkpoint(cx, "native_ann.generation.source_field")?;
                let bytes = self
                    .lexical
                    .stored_field_value(field_id, resolved.global_docid)
                    .map_err(|_| refused("source", "invalid-stored-field"))?
                    .ok_or_else(|| refused("source", "missing-stored-field"))?;
                source_bytes = source_bytes
                    .checked_add(bytes.len())
                    .filter(|total| *total <= max_source_bytes)
                    .ok_or_else(|| refused("source", "stored-payload-limit"))?;
                *value = bytes;
            }
            checkpoint(cx, "native_ann.generation.source_decode")?;
            let [id, content, title, metadata_json] = values;
            let id = String::from_utf8(id).map_err(|_| refused("source", "invalid-stored-utf8"))?;
            if id != row.doc_id() {
                return Err(refused("source", "stored-id-mismatch"));
            }
            let content =
                String::from_utf8(content).map_err(|_| refused("source", "invalid-stored-utf8"))?;
            let title =
                String::from_utf8(title).map_err(|_| refused("source", "invalid-stored-utf8"))?;
            let metadata: BTreeMap<String, String> = serde_json::from_slice(&metadata_json)
                .map_err(|_| refused("source", "invalid-metadata"))?;
            // The shipping writer emits sorted canonical string maps. Comparing
            // their encoding also refuses duplicate JSON keys instead of
            // silently discarding source values during deserialization.
            if serde_json::to_vec(&metadata).map_err(|_| refused("source", "invalid-metadata"))?
                != metadata_json
            {
                return Err(refused("source", "noncanonical-metadata"));
            }
            documents.push(IndexableDocument {
                id,
                content,
                title: (!title.is_empty()).then_some(title),
                metadata: metadata.into_iter().collect(),
            });
        }
        if documents.len() != live_count {
            return Err(refused("source", "live-count-mismatch"));
        }
        checkpoint(cx, "native_ann.generation.source_complete")?;
        Ok(documents)
    }

    async fn search_text(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        checkpoint(cx, "native_ann.generation.search")?;
        let live_count =
            usize::try_from(self.vector.owner_witness().live_count).unwrap_or(usize::MAX);
        let k = k.min(live_count);
        match self.tier {
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
        }
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
    checkpoint(cx, "native_ann.generation.components")?;
    let declared = activation.components();
    if declared.ann.is_some() != ann_image.is_some() {
        return Err(refused("ann", "component-presence-mismatch"));
    }
    if activation.generation() != expected_vector.generation() {
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
    if metadata.vector_artifacts.len() != 1
        || metadata.lexical_artifacts.len() != 1
        || metadata.embedders.len() != 1
        || !metadata.repair_descriptors.is_empty()
    {
        return Err(refused("metadata", "unsupported-component-shape"));
    }
    if !validate_manifest(&metadata).is_valid() {
        return Err(refused("metadata", "invalid-manifest"));
    }
    let native_receipt = match (declared.ann, metadata.metadata.get(NATIVE_ANN_RECEIPT_KEY)) {
        (Some(_), Some(encoded)) => Some(
            serde_json::from_str::<NativeHnswGenerationReceiptV2>(encoded)
                .map_err(|_| refused("ann", "invalid-native-receipt"))?,
        ),
        (None, None) => None,
        _ => return Err(refused("ann", "native-receipt-presence-mismatch")),
    };
    let checkpoint_value = SourceCheckpointV1::derive(&metadata.commit_range);
    if checkpoint_value.to_bytes() != activation.source_checkpoint_sha256() {
        return Err(refused("checkpoint", "source-mismatch"));
    }
    checkpoint(cx, "native_ann.generation.vector")?;
    let owner = Arc::new(
        ValidatedFsviBytes::from_arc(vector_bytes, expected_vector)
            .map_err(|_| refused("vector", "admission-failed"))?,
    );
    let witness = owner.witness();
    let mut identifiers = Vec::new();
    for row in 0..owner.record_count() {
        checkpoint(cx, "native_ann.generation.document")?;
        let row = owner.row(row)?;
        if row.flags().is_live() {
            identifiers.push(row.doc_id().to_owned());
        }
    }
    let vector_receipt = vector_component_receipt(
        witness,
        identifiers.iter().map(String::as_str),
        checkpoint_value,
    )
    .map_err(|_| refused("vector", "invalid-document-set"))?;
    if vector_receipt.bytes != declared.vector
        || vector_receipt.docset_digest != activation.document_set_sha256()
    {
        return Err(refused("vector", "selected-document-set-mismatch"));
    }
    let ann_receipt = native_receipt
        .as_ref()
        .map(|receipt| {
            ann_component_receipt(
                receipt,
                identifiers.iter().map(String::as_str),
                checkpoint_value,
            )
            .map_err(|_| refused("ann", "invalid-document-set"))
        })
        .transpose()?;
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
        owner.record_count(),
        witness.live_count,
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
    let (vector, retrieval) = match (ann_image, native_receipt) {
        (Some(bytes), Some(receipt)) => {
            let graph = ValidatedNativeHnsw::from_bytes(Arc::clone(&owner), bytes, &receipt)
                .map_err(|_| refused("ann", "native-admission-failed"))?;
            checkpoint(cx, "native_ann.generation.ann_admitted")?;
            let dimension =
                |value| usize::try_from(value).map_err(|_| refused("ann", "invalid-parameters"));
            let params = HnswParams {
                m: dimension(receipt.params.m)?,
                m0: dimension(receipt.params.m0)?,
                ef_construction: dimension(receipt.params.ef_construction)?,
                ef_search: dimension(receipt.params.ef_search)?,
            };
            let vector = NativeAnnIndex {
                owner,
                graph: NativeBackend::Ann(Box::new(graph)),
                default_ef_search: params.ef_search,
            };
            (
                vector,
                NativeBuildRetrieval::Hnsw {
                    params,
                    seed: receipt.seed,
                },
            )
        }
        (None, None) => (
            NativeAnnIndex::exact(cx, owner)?,
            NativeBuildRetrieval::Exact,
        ),
        _ => return Err(refused("ann", "native-receipt-presence-mismatch")),
    };
    let lexical_result = QuillSnapshotIndex::new(cx, lexical, config);
    checkpoint(cx, "native_ann.generation.ready")?;
    let lexical = lexical_result.map_err(|_| refused("lexical", "invalid-search-configuration"))?;
    let tier = metadata.vector_artifacts[0].embedder_tier;
    Ok(CompositeReaders {
        vector: Arc::new(vector),
        lexical,
        metadata,
        tier,
        retrieval,
        components,
    })
}

fn validate_artifact_metadata(
    metadata: &GenerationManifest,
    binding: &FsviV2IdentityBinding,
    activation: &ActivationManifest,
    record_count: usize,
    live_count: u64,
) -> SearchResult<()> {
    let declared = activation.components();
    let vector = &metadata.vector_artifacts[0];
    let lexical = &metadata.lexical_artifacts[0];
    let tier = match vector.embedder_tier {
        EmbedderTierTag::Fast => "fast",
        EmbedderTierTag::Quality => "quality",
    };
    if metadata.embedders.get(tier) != Some(binding.frozen_identity())
        || usize::try_from(vector.dimension).ok() != Some(binding.dimension())
        || usize::try_from(vector.vector_count).ok() != Some(record_count)
        || vector.size_bytes != declared.vector.byte_len
        || vector.checksum != lower_hex(&declared.vector.sha256)
        || lexical.document_count != live_count
        || lexical.size_bytes != declared.lexical.byte_len
        || lexical.checksum != lower_hex(&declared.lexical.sha256)
        || metadata.total_documents != live_count
    {
        return Err(refused("metadata", "artifact-binding-mismatch"));
    }
    for invariant in &metadata.activation_invariants {
        match &invariant.kind {
            InvariantKind::AllArtifactsVerified | InvariantKind::EmbedderRevisionMatch => {}
            InvariantKind::VectorCountConsistency { expected_total }
                if usize::try_from(*expected_total).ok() == Some(record_count) => {}
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
