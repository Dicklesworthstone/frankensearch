//! Incremental inference into a new, complete native generation.
//!
//! Source edits never mutate the retained predecessor. Reuse is by exact source
//! ID, content, producer identity and storage precision, not by row number. New
//! FSVI rows and optional graphs are always built for the successor's membership.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use super::{
    NativeBuildPrecision, NativeBuildRetrieval, NativeBuiltIndex, NativeBuiltTier,
    NativeIndexBuilder, TierPlan, checkpoint, invalid,
};
use crate::{Cx, IndexableDocument, SearchResult};
use frankensearch_core::generation::ArtifactGenerationIdentityV1;
use frankensearch_index::native_hnsw::HnswParams;
use frankensearch_index::{ValidatedFsviBytes, VectorIndexWriter};

#[cfg(feature = "quill")]
use super::NativeBuiltHybridIndex;

/// Pending upserts/deletes against one immutable source/vector cohort.
///
/// The last edit for an ID wins; deleting an absent ID is a no-op. A successful
/// build contains every surviving document in every configured tier. Unchanged
/// content reuses its admitted vector; title/metadata edits still replace the
/// retained source document. Prepared `content` is the complete embedding input,
/// just as in [`NativeIndexBuilder`]. A rename is a delete plus an upsert and
/// deliberately re-embeds the new ID rather than assuming cross-ID equivalence.
///
/// Producers and tier presence cannot change through this API. Precision and
/// graph policies are inherited but may be explicitly overridden. A precision
/// change re-embeds that entire tier: decoded F16 values are not original F32
/// model outputs. Reuse does not skip producer admission, even for empty builds.
///
/// This stages a successor, not a CURRENT switch or a durable rollback floor.
/// The predecessor remains searchable throughout a failed, cancelled or dropped
/// update. A caller must seal/select the completed successor through its existing
/// publication policy. Source, vector and graph rebuilding still have their
/// ordinary full-cohort memory and I/O costs; only inference is incremental.
/// Construct with [`NativeBuiltIndex::begin_update`].
pub struct NativeIndexUpdate {
    builder: NativeIndexBuilder,
    edits: BTreeMap<String, Option<IndexableDocument>>,
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
        checkpoint(cx, "native_ann.update.prepare")?;
        if generation.sequence <= self.fast.index.owner_witness().generation.sequence {
            return Err(invalid(
                "update.generation",
                "not-newer",
                "an update requires a strictly higher generation sequence",
            ));
        }
        let mut builder =
            NativeIndexBuilder::new(directory, generation, Arc::clone(&self.fast.embedder))?;
        builder.fast = retained_plan(&self.fast)?;
        builder.quality = self.quality.as_ref().map(retained_plan).transpose()?;
        // Admit the new generation binding even when no changed documents will
        // need inference. An empty successor is not an identity bypass.
        builder.fast.binding(&generation)?;
        if let Some(quality) = &builder.quality {
            quality.binding(&generation)?;
        }
        builder.reuse = Some(ReuseSource {
            documents: Arc::clone(&self.documents),
            fast: ReuseTier::new(cx, &self.fast, &self.documents)?,
            quality: self
                .quality
                .as_ref()
                .map(|tier| ReuseTier::new(cx, tier, &self.documents))
                .transpose()?,
        });
        checkpoint(cx, "native_ann.update.prepared")?;
        Ok(NativeIndexUpdate {
            builder,
            edits: BTreeMap::new(),
        })
    }
}

impl NativeIndexUpdate {
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

    fn prepare(mut self, cx: &Cx) -> SearchResult<NativeIndexBuilder> {
        checkpoint(cx, "native_ann.update.edits")?;
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
        for previous in source.documents.iter() {
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

pub(super) struct ReuseSource {
    documents: Arc<[IndexableDocument]>,
    pub(super) fast: ReuseTier,
    pub(super) quality: Option<ReuseTier>,
}

pub(super) struct ReuseTier {
    owner: Arc<ValidatedFsviBytes>,
    physical_rows: Vec<usize>,
    producer_fingerprint: String,
    precision: NativeBuildPrecision,
}

impl ReuseTier {
    fn new(cx: &Cx, tier: &NativeBuiltTier, documents: &[IndexableDocument]) -> SearchResult<Self> {
        let owner = Arc::clone(&tier.index.owner);
        if owner.record_count() != documents.len() || owner.live_count() != documents.len() {
            return Err(invalid(
                "update.source_join",
                "cardinality",
                "source/tier membership differs",
            ));
        }
        let mut physical_rows = vec![usize::MAX; documents.len()];
        for physical in 0..owner.record_count() {
            checkpoint(cx, "native_ann.update.row_map")?;
            let row = owner.row(physical)?;
            let source = documents
                .binary_search_by(|document| document.id.as_str().cmp(row.doc_id()))
                .map_err(|_| {
                    invalid("update.source_join", "unknown-id", "row has no source document")
                })?;
            if !row.flags().is_live()
                || std::mem::replace(&mut physical_rows[source], physical) != usize::MAX
            {
                return Err(invalid(
                    "update.source_join",
                    "duplicate-or-deleted",
                    "source rows must be live and unique",
                ));
            }
        }
        Ok(Self {
            owner,
            physical_rows,
            producer_fingerprint: tier.producer_identity.fingerprint(),
            precision: tier.precision,
        })
    }
}

impl TierPlan {
    pub(super) async fn write_batch_reusing(
        &self,
        cx: &Cx,
        writer: &mut VectorIndexWriter,
        documents: &[IndexableDocument],
        reuse: Option<(&ReuseSource, &ReuseTier)>,
    ) -> SearchResult<()> {
        let Some((source, tier)) = reuse else {
            return self.write_batch(cx, writer, documents).await;
        };
        checkpoint(cx, "native_ann.update.before_batch")?;
        self.admit(self.embedder.identity()?)?;
        if self.identity.fingerprint() != tier.producer_fingerprint {
            return Err(invalid(
                "update.producer",
                "changed",
                "reused vectors require the original complete producer identity",
            ));
        }
        if self.precision != tier.precision {
            return self.write_batch(cx, writer, documents).await;
        }
        let mut changed = Vec::new();
        for document in documents {
            checkpoint(cx, "native_ann.update.reuse_row")?;
            let previous = source
                .documents
                .binary_search_by(|previous| previous.id.cmp(&document.id))
                .ok()
                .filter(|&position| source.documents[position].content == document.content);
            if let Some(position) = previous {
                let physical = tier.physical_rows[position];
                let vector = tier.owner.vector_at_f32(physical)?;
                // F16 -> F32 -> the SAME F16 storage is exact. F32 rows also
                // retain their bits; no normalized or rank-derived score is used.
                writer.write_record(document.id.as_str(), &vector)?;
            } else {
                changed.push(document.clone());
            }
        }
        if !changed.is_empty() {
            self.write_batch(cx, writer, &changed).await?;
        }
        checkpoint(cx, "native_ann.update.after_batch")?;
        self.admit(self.embedder.identity()?)
    }
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
            let identity = EmbeddingIdentityBundleV1::explicit_test_model(name, dimension);
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
                    output.last_mut().unwrap().identity.clone_from(&self.foreign);
                }
                if fault == CANCEL {
                    cx.set_cancel_requested(true);
                }
                Ok(output)
            })
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
                actual.index.owner.vector_at_f32(row).unwrap()
                    .iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                expected.index.owner.vector_at_f32(row).unwrap()
                    .iter().map(|value| value.to_bits()).collect::<Vec<_>>()
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
                        .build(&cx).await.unwrap();
                    let old_fast_bytes = std::fs::read(old.fast.vector_path()).unwrap();
                    let old_quality_bytes = std::fs::read(old.quality().unwrap().vector_path()).unwrap();
                    assert_eq!(fast.submitted.load(Ordering::SeqCst), 5);
                    assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
                    let revised_a = IndexableDocument::new("a", "fractional")
                        .with_title("new title")
                        .with_metadata("revision", "2");
                    let next = old.begin_update(&cx, dir.path().join("next"), generation(2))
                        .unwrap()
                        .upsert_document(IndexableDocument::new("b", "horizontal"))
                        .delete_document("c")
                        .upsert_document(revised_a.clone())
                        .upsert_document(IndexableDocument::new("new", "vertical"))
                        .with_batch_size(2).unwrap()
                        .build(&cx).await.unwrap();
                    assert_eq!(fast.submitted.load(Ordering::SeqCst), 7);
                    assert_eq!(quality.submitted.load(Ordering::SeqCst), 7);
                    assert_eq!(next.fast.graph_path.is_some(), ann);
                    assert_eq!(next.quality().unwrap().graph_path.is_some(), !ann);
                    assert!(next.document("c").is_none());
                    assert_eq!(next.document("a").unwrap().metadata["revision"], "2");
                    assert_eq!(next.document("a").unwrap().title.as_deref(), Some("new title"));
                    // The baseline re-embeds the explicitly specified final cohort.
                    // It does not derive expected vectors or rows from the update.
                    let rebuilt = NativeIndexBuilder::new(
                        dir.path().join("rebuilt"), generation(2), fast.clone(),
                    ).unwrap()
                        .with_quality_embedder(quality.clone()).unwrap()
                        .with_fast_storage(precision, graph(ann))
                        .with_quality_storage(other_precision, graph(!ann)).unwrap()
                        .add_documents([
                            revised_a,
                            IndexableDocument::new("b", "horizontal"),
                            IndexableDocument::new("x", "diagonal"),
                            IndexableDocument::new("z-fast", "horizontal"),
                            IndexableDocument::new("new", "vertical"),
                        ])
                        .build(&cx).await.unwrap();
                    assert_same_tier(next.fast(), rebuilt.fast());
                    assert_same_tier(next.quality().unwrap(), rebuilt.quality().unwrap());
                    assert_eq!(old.document("b").unwrap().content, "vertical");
                    assert!(old.document("c").is_some() && old.document("new").is_none());
                    assert_eq!(std::fs::read(old.fast.vector_path()).unwrap(), old_fast_bytes);
                    assert_eq!(std::fs::read(old.quality().unwrap().vector_path()).unwrap(), old_quality_bytes);
                    assert_eq!(old.fast().search(&cx, "vertical", 1).await.unwrap()[0].doc_id, "b");
                    assert_eq!(next.fast().search(&cx, "vertical", 1).await.unwrap()[0].doc_id, "new");
                }
            }
        });
    }

    #[test]
    fn metadata_only_noop_and_delete_all_updates_do_not_start_inference() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let old = builder(&dir.path().join("old"), &fast, &quality).build(&cx).await.unwrap();
            // Both providers would fail if any batch were submitted.
            fast.fault.store(FAIL, Ordering::SeqCst);
            quality.fault.store(FAIL, Ordering::SeqCst);
            let metadata = old.begin_update(&cx, dir.path().join("metadata"), generation(2))
                .unwrap()
                .upsert_document(IndexableDocument::new("b", "vertical").with_metadata("v", "new"))
                .delete_document("absent")
                .build(&cx).await.unwrap();
            assert_eq!(metadata.document("b").unwrap().metadata["v"], "new");
            let noop = metadata.begin_update(&cx, dir.path().join("noop"), generation(3))
                .unwrap().build(&cx).await.unwrap();
            let mut update = noop.begin_update(&cx, dir.path().join("empty"), generation(4)).unwrap();
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

    #[test]
    fn precision_change_reembeds_only_that_tier_instead_of_promoting_rounded_vectors() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let old = builder(&dir.path().join("old"), &fast, &quality)
                .with_fast_storage(NativeBuildPrecision::F16, graph(false))
                .build(&cx).await.unwrap();
            let next = old.begin_update(&cx, dir.path().join("next"), generation(2))
                .unwrap()
                .with_fast_storage(NativeBuildPrecision::F32, graph(true))
                .build(&cx).await.unwrap();
            assert_eq!(fast.submitted.load(Ordering::SeqCst), 10);
            assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
            let owner = &next.fast.index.owner;
            let row = (0..owner.record_count()).find(|&row| owner.doc_id_at(row).unwrap() == "a").unwrap();
            let actual = owner.vector_at_f32(row).unwrap();
            assert_eq!(actual, fast.values("fractional"));
            let old_owner = &old.fast.index.owner;
            let old_row = (0..old_owner.record_count()).find(|&row| old_owner.doc_id_at(row).unwrap() == "a").unwrap();
            assert_ne!(actual, old_owner.vector_at_f32(old_row).unwrap());
        });
    }

    #[test]
    fn ordered_edits_do_not_resurrect_deletions_and_new_ids_are_embedded() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let fast = Provider::new("fast", 2);
            let old = NativeIndexBuilder::new(dir.path().join("old"), generation(1), fast.clone())
                .unwrap().add_document(IndexableDocument::new("old-id", "vertical"))
                .build(&cx).await.unwrap();
            let next = old.begin_update(&cx, dir.path().join("next"), generation(2)).unwrap()
                .delete_document("old-id")
                .upsert_document(IndexableDocument::new("new-id", "horizontal"))
                .delete_document("new-id")
                .upsert_document(IndexableDocument::new("new-id", "vertical"))
                .upsert_document(IndexableDocument::new("temporary", "horizontal"))
                .delete_document("temporary")
                .build(&cx).await.unwrap();
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
                let old = builder(&dir.path().join("old"), &fast, &quality).build(&cx).await.unwrap();
                let old_bytes = std::fs::read(old.fast.vector_path()).unwrap();
                quality.fault.store(fault, Ordering::SeqCst);
                let path = dir.path().join("failed");
                let result = old.begin_update(&cx, &path, generation(2)).unwrap()
                    .upsert_document(IndexableDocument::new("new", "vertical"))
                    .build(&cx).await;
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
                assert_eq!(old.fast().search(&cx, "vertical", 1).await.unwrap()[0].doc_id, "b");
                assert_eq!(old.quality().unwrap().search(&cx, "vertical", 1).await.unwrap()[0].doc_id, "b");
            }
        });
    }

    #[test]
    fn pending_update_keeps_old_readers_live_and_drop_does_not_publish_a_partial_generation() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let old = builder(&dir.path().join("old"), &fast, &quality).build(&cx).await.unwrap();
            quality.fault.store(PENDING, Ordering::SeqCst);
            let path = dir.path().join("abandoned");
            let update = old.begin_update(&cx, &path, generation(2)).unwrap()
                .upsert_document(IndexableDocument::new("new", "vertical"));
            let mut future = Box::pin(update.build(&cx));
            assert!(future.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
            assert_eq!(old.fast().search(&cx, "vertical", 1).await.unwrap()[0].doc_id, "b");
            assert_eq!(old.quality().unwrap().search(&cx, "vertical", 1).await.unwrap()[0].doc_id, "b");
            drop(future);
            assert_eq!(quality.drops.load(Ordering::SeqCst), 2);
            assert!(!path.join("fast.fsvi").exists());
            quality.fault.store(OK, Ordering::SeqCst);
            let retry = old.begin_update(&cx, dir.path().join("retry"), generation(2)).unwrap()
                .upsert_document(IndexableDocument::new("new", "vertical"))
                .build(&cx).await.unwrap();
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
            let old = builder(&dir.path().join("old"), &fast, &quality).build(&cx).await.unwrap();
            let path = dir.path().join("rejected");
            assert!(old.begin_update(&cx, &path, generation(1)).is_err());
            assert!(old.begin_update(
                &cx, &path, ArtifactGenerationIdentityV1::new(1, [0x56; 16]).unwrap(),
            ).is_err());
            quality.fault.store(IDENTITY_DRIFT, Ordering::SeqCst);
            assert!(old.begin_update(&cx, &path, generation(2)).is_err());
            quality.fault.store(OK, Ordering::SeqCst);
            let update = old.begin_update(&cx, &path, generation(2)).unwrap();
            quality.fault.store(IDENTITY_DRIFT, Ordering::SeqCst);
            assert!(update.build(&cx).await.is_err());
            quality.fault.store(OK, Ordering::SeqCst);
            assert!(old.begin_update(&cx, &path, generation(2)).unwrap()
                .delete_document("").build(&cx).await.is_err());
            assert!(old.begin_update(&cx, &path, generation(2)).unwrap()
                .upsert_document(IndexableDocument::new("", "horizontal"))
                .build(&cx).await.is_err());
            assert!(!path.exists());
            assert_eq!(fast.submitted.load(Ordering::SeqCst), 5);
            assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
            let result = old.begin_update(&cx, old.directory(), generation(2)).unwrap()
                .build(&cx).await;
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
            let old = builder(&old_path, &fast, &quality).build(&cx).await.unwrap();
            let update = old.begin_update(&cx, dir.path().join("next"), generation(2)).unwrap();
            std::fs::rename(&old_path, dir.path().join("archived")).unwrap();
            assert!(!old_path.exists());
            drop(old);
            let next = update.build(&cx).await.unwrap();
            assert_eq!(next.documents().len(), 5);
            assert_eq!(fast.submitted.load(Ordering::SeqCst), 5);
            assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
            assert_eq!(next.fast().search(&cx, "vertical", 1).await.unwrap()[0].doc_id, "b");
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
                .build(&cx).await.unwrap();
            let receipt = old.seal_for_reopen(&cx).unwrap();
            drop(old);
            let reopened = NativeBuiltIndex::open_selected(
                &cx, &path, &receipt, fast.clone(), Some(quality.clone()),
            ).unwrap();
            let next_path = dir.path().join("next");
            let next = reopened.begin_update(&cx, &next_path, generation(2)).unwrap()
                .delete_document("b")
                .build(&cx).await.unwrap();
            assert_eq!(fast.submitted.load(Ordering::SeqCst), 5);
            assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
            assert!(next.fast.graph_path().is_some());
            let next_receipt = next.seal_for_reopen(&cx).unwrap();
            drop(next);
            let selected = NativeBuiltIndex::open_selected(
                &cx, &next_path, &next_receipt, fast.clone(), Some(quality.clone()),
            ).unwrap();
            assert!(selected.document("b").is_none());
            assert!(reopened.document("b").is_some());
            assert_eq!(selected.fast.index.owner_witness().generation, generation(2));
            assert_eq!(selected.quality().unwrap().index.owner_witness().generation, generation(2));
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
                .with_quality_embedder(quality.clone()).unwrap()
                .with_fast_storage(NativeBuildPrecision::F16, graph(true))
                .add_documents([
                    IndexableDocument::new("a", "retained common"),
                    IndexableDocument::new("b", "obsolete common").with_metadata("revision", "old"),
                    IndexableDocument::new("c", "removed common"),
                ])
                .build_hybrid(&cx).await.unwrap();
            let mut old_stream = old.progressive(&cx, "obsolete", 1).unwrap();
            let Some(NativeSearchPhase::Initial { results, .. }) =
                old_stream.next_phase().await.unwrap()
            else {
                panic!("initial");
            };
            assert_eq!(results[0].doc_id, "b");
            assert!(results[0].lexical_score.is_some());

            let path = dir.path().join("next");
            let next = old.begin_update(&cx, &path, generation(2)).unwrap()
                .delete_document("c")
                .upsert_documents([
                    IndexableDocument::new("b", "replacement common").with_metadata("revision", "new"),
                    IndexableDocument::new("d", "arrival common"),
                ])
                .build_hybrid(&cx).await.unwrap();
            assert_eq!(fast.submitted.load(Ordering::SeqCst), 5);
            assert_eq!(quality.submitted.load(Ordering::SeqCst), 5);
            assert_eq!(old.lexical().doc_count().unwrap(), 3);
            assert_eq!(next.lexical().doc_count().unwrap(), 3);
            assert!(next.lexical().search(&cx, "obsolete", 10).await.unwrap().is_empty());
            assert!(next.lexical().search(&cx, "removed", 10).await.unwrap().is_empty());
            assert_eq!(next.lexical().search(&cx, "replacement", 10).await.unwrap()[0].doc_id, "b");
            assert_eq!(next.lexical().search(&cx, "arrival", 10).await.unwrap()[0].doc_id, "d");
            assert_eq!(old.lexical().search(&cx, "removed", 10).await.unwrap()[0].doc_id, "c");
            let Some(NativeSearchPhase::Refined { results, .. }) =
                old_stream.next_phase().await.unwrap()
            else {
                panic!("old stream retains its admitted generation");
            };
            assert_eq!(results[0].doc_id, "b");
            assert!(results[0].lexical_score.is_some());
            assert_eq!(old.vectors().document("b").unwrap().metadata["revision"], "old");
            assert_eq!(next.vectors().document("b").unwrap().metadata["revision"], "new");

            let receipt = next.seal_for_reopen(&cx).unwrap();
            drop(next);
            let reopened = NativeBuiltHybridIndex::open_selected(
                &cx, &path, &receipt, fast.clone(), Some(quality.clone()),
            ).await.unwrap();
            assert!(reopened.vectors().document("c").is_none());
            assert_eq!(reopened.lexical().search(&cx, "arrival", 10).await.unwrap()[0].doc_id, "d");
            assert_eq!(reopened.vectors().document("b").unwrap().content, "replacement common");
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
                .build_hybrid(&cx).await.unwrap();
            quality.fault.store(PENDING, Ordering::SeqCst);
            let path = dir.path().join("pending");
            let update = old.begin_update(&cx, &path, generation(2)).unwrap()
                .delete_document("b")
                .upsert_document(IndexableDocument::new("new", "vertical"));
            let mut future = Box::pin(update.build_hybrid(&cx));
            assert!(future.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
            assert_eq!(old.lexical().search(&cx, "vertical", 10).await.unwrap()[0].doc_id, "b");
            assert_eq!(old.search(&cx, "vertical", 1).await.unwrap()[0].doc_id, "b");
            drop(future);
            assert_eq!(quality.drops.load(Ordering::SeqCst), 2);
            assert!(!path.join("native.hybrid.json").exists());
            assert!(!path.join("lexical").exists());
            assert_eq!(old.lexical().doc_count().unwrap(), 5);
        });
    }
}
