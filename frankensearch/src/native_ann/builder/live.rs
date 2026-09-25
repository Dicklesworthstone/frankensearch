//! Live, process-local selection of complete native hybrid generations.
//!
//! Queries retain one source/vector/Quill object, not independently refreshed
//! readers. Building or reopening a successor happens outside the selection lock.
//! Installation compares the exact predecessor pin, not just a sequence number,
//! so competing updates cannot silently discard each other's source edits.
//!
//! This is NOT a durable publisher or rollback floor. A selected-reopen receipt
//! still comes from the caller's trusted catalog, and CURRENT, writer fencing,
//! filesystem immutability and disk retention remain the existing publisher's
//! responsibility. In particular, keep every pinned Quill generation's mapped
//! files immutable and present while its readers live; this API does not run GC.

use std::fmt;
use std::path::Path;
use std::sync::Arc;

use asupersync::sync::{RwLock, RwLockError};
use frankensearch_core::generation::{ArtifactGenerationIdentityV1, GenerationComponentReceiptV1};

use super::{
    NativeBuildPrecision, NativeBuildRetrieval, NativeBuiltHybridIndex, NativeHybridReopenLimits,
    NativeIndexUpdate, checkpoint, invalid,
};
use crate::{Cx, IndexableDocument, ScoredResult, SearchError, SearchResult};

/// Shared serving selection for a complete, read-only native hybrid generation.
///
/// Share this handle with `Arc`. The cancel-aware lock protects only a short
/// snapshot clone or whole-generation swap. It is never held during inference,
/// retrieval, hydration, reranking, filesystem access, or generation destruction.
/// A slow query cannot hold the selection lock while an update is installed.
///
/// For progressive queries, obtain [`Self::snapshot`] once and call
/// `snapshot.index().progressive(...)` or `progressive_with_reranker(...)`.
/// Keep that snapshot through all phases and result-row interpretation. Never
/// reacquire the current snapshot between phases of the same query.
#[derive(Debug)]
pub struct NativeLiveHybridIndex {
    current: RwLock<NativeHybridSnapshot>,
}

/// Owned pin of the exact source documents, vector owners, models and Quill view.
///
/// Cloning shares the complete object, not vector slabs or source strings.
/// Replacing the live selection does not change this snapshot or invalidate
/// its physical row coordinates. Pins from another live handle are not valid
/// predecessors here, even when their generation numbers and bytes are equal.
#[derive(Clone)]
pub struct NativeHybridSnapshot {
    index: Arc<NativeBuiltHybridIndex>,
}

impl fmt::Debug for NativeHybridSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeHybridSnapshot")
            .field("generation", &self.generation())
            .field("documents", &self.index.vectors().documents().len())
            .finish_non_exhaustive()
    }
}

/// Query results accompanied by the owners needed to interpret their rows.
///
/// A result index belongs to this snapshot, not to the live handle's newest
/// generation. The existing native fast/quality row-selection policy is unchanged.
#[derive(Debug)]
pub struct NativeHybridResults {
    /// Retained complete generation that actually executed this query.
    pub snapshot: NativeHybridSnapshot,
    /// Ranked and hydrated results from that snapshot.
    pub results: Vec<ScoredResult>,
}

/// Complete successor paired with the exact predecessor it was prepared against.
///
/// Only successful hybrid builds or selected hybrid reopens create candidates.
/// Private fields prevent pairing an arbitrary vector-only result or an unrelated
/// update with a newer predecessor after a stale-install refusal.
#[derive(Debug)]
pub struct NativeHybridCandidate {
    base: NativeHybridSnapshot,
    next: NativeHybridSnapshot,
}

/// Source edits tied to one retained live predecessor.
///
/// This delegates incremental inference and complete-cohort rebuilding to
/// [`NativeIndexUpdate`]. It can only build a hybrid candidate: a required
/// quality/lexical failure cannot be installed as vector-only success.
pub struct NativeLiveHybridUpdate {
    base: NativeHybridSnapshot,
    update: NativeIndexUpdate,
}

impl NativeLiveHybridIndex {
    /// Install an already-built or explicitly reopened complete initial cohort.
    /// No inference, file writes, runtime creation or generation discovery occurs.
    ///
    /// # Errors
    /// Returns cancellation before creating the live handle.
    pub fn new(cx: &Cx, initial: NativeBuiltHybridIndex) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.live.new")?;
        Ok(Self {
            current: RwLock::new(NativeHybridSnapshot {
                index: Arc::new(initial),
            }),
        })
    }

    /// Pin one complete current generation without holding a lock during queries.
    ///
    /// # Errors
    /// Returns cancellation or a poisoned selection-lock error. A dropped pending
    /// acquisition owns no snapshot and cannot change the live selection.
    pub async fn snapshot(&self, cx: &Cx) -> SearchResult<NativeHybridSnapshot> {
        checkpoint(cx, "native_ann.live.snapshot")?;
        let current = self.current.read(cx).await.map_err(lock_error)?;
        checkpoint(cx, "native_ann.live.snapshot_acquired")?;
        let snapshot = (*current).clone();
        drop(current);
        Ok(snapshot)
    }

    /// Atomically select a prepared successor only if its predecessor is current.
    ///
    /// A higher candidate sequence does NOT override stale predecessor evidence.
    /// On refusal, cancellation or a dropped pending acquisition, selection stays
    /// unchanged and the borrowed candidate remains available for inspection.
    /// Retrying that same candidate after another install still fails; rebuild
    /// edits against a fresh pin instead of rebasing them by changing a number.
    ///
    /// The final cancellation checkpoint precedes the swap. A successful swap is
    /// not subsequently reported cancelled. Old owners are released only after
    /// unlocking, and snapshots held by queries continue serving the old cohort.
    /// This does not publish a durable pointer or authorize deleting old artifacts.
    ///
    /// # Errors
    /// Returns stale/foreign predecessor, cancellation or selection-lock errors.
    pub async fn install(
        &self,
        cx: &Cx,
        candidate: &NativeHybridCandidate,
    ) -> SearchResult<NativeHybridSnapshot> {
        checkpoint(cx, "native_ann.live.install")?;
        let mut current = self.current.write(cx).await.map_err(lock_error)?;
        checkpoint(cx, "native_ann.live.install_acquired")?;
        if !Arc::ptr_eq(&current.index, &candidate.base.index) {
            return Err(invalid(
                "live.expected_current",
                "stale-or-foreign",
                "the candidate must retain this live handle's exact current predecessor",
            ));
        }
        checkpoint(cx, "native_ann.live.install_commit")?;
        let installed = candidate.next.clone();
        let previous = std::mem::replace(&mut *current, installed.clone());
        drop(current);
        // Destructors can release mapped readers or caller-owned models. Never
        // run the last owner's destructor inside the serving-selection lock.
        drop(previous);
        Ok(installed)
    }

    /// Fast-plus-lexical query returning its originating snapshot with the page.
    ///
    /// # Errors
    /// Propagates pin acquisition and the complete native hybrid search errors.
    pub async fn search(&self, cx: &Cx, text: &str, k: usize) -> SearchResult<NativeHybridResults> {
        let snapshot = self.snapshot(cx).await?;
        let results = snapshot.index.search(cx, text, k).await?;
        Ok(NativeHybridResults { snapshot, results })
    }

    /// Independently refined query with the same pin for every configured arm.
    ///
    /// # Errors
    /// Propagates acquisition, required-tier, lexical and hydration failures.
    pub async fn search_refined(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
    ) -> SearchResult<NativeHybridResults> {
        let snapshot = self.snapshot(cx).await?;
        let results = snapshot.index.search_refined(cx, text, k).await?;
        Ok(NativeHybridResults { snapshot, results })
    }

    /// Quality-primary query without starting the fast provider.
    ///
    /// # Errors
    /// Refuses an absent quality tier and propagates pin/search failures.
    pub async fn search_quality(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
    ) -> SearchResult<NativeHybridResults> {
        let snapshot = self.snapshot(cx).await?;
        let results = snapshot.index.search_quality(cx, text, k).await?;
        Ok(NativeHybridResults { snapshot, results })
    }
}

impl NativeHybridSnapshot {
    /// Complete read-only cohort for queries, progressive phases and row lookup.
    #[must_use]
    pub fn index(&self) -> &NativeBuiltHybridIndex {
        &self.index
    }

    /// Complete artifact generation identity, including its nonce.
    #[must_use]
    pub fn generation(&self) -> ArtifactGenerationIdentityV1 {
        self.index
            .vectors()
            .fast()
            .index()
            .owner_witness()
            .generation
    }

    /// Stage incremental source edits against this exact predecessor.
    ///
    /// No live selection lock is held and no inference or filesystem write starts.
    /// Changing model identity or adding/removing a quality tier requires a new
    /// live handle, rather than silently migrating this handle's query contract.
    ///
    /// # Errors
    /// Has the generation, provider, membership and cancellation errors of
    /// [`NativeBuiltHybridIndex::begin_update`].
    pub fn begin_update(
        &self,
        cx: &Cx,
        directory: impl AsRef<Path>,
        generation: ArtifactGenerationIdentityV1,
    ) -> SearchResult<NativeLiveHybridUpdate> {
        let update = self.index.begin_update(cx, directory, generation)?;
        Ok(NativeLiveHybridUpdate {
            base: self.clone(),
            update,
        })
    }

    /// Prepare an externally selected complete successor with the retained models.
    ///
    /// The receipt must come from the caller's trusted selection, not from hashing
    /// an unknown directory. Reuses the existing exact hybrid reopen and resource
    /// limits, including required quality topology and all source/vector/Quill
    /// bytes. No inference, discovery, repair, file mutation or fallback occurs.
    /// The old generation remains selected while reopening suspends or fails.
    ///
    /// This records the expected in-process predecessor, not external publication
    /// lineage. The caller's durable authority must authorize the selected cohort.
    ///
    /// # Errors
    /// Propagates selected-reopen errors and refuses non-newer generations or
    /// changes to either retained producer. Installation still checks staleness.
    pub async fn prepare_selected(
        &self,
        cx: &Cx,
        directory: impl AsRef<Path> + Send,
        expected: &GenerationComponentReceiptV1,
        limits: NativeHybridReopenLimits,
    ) -> SearchResult<NativeHybridCandidate> {
        checkpoint(cx, "native_ann.live.prepare_selected")?;
        let vectors = self.index.vectors();
        let next = NativeBuiltHybridIndex::open_selected_with_limits(
            cx,
            directory,
            expected,
            Arc::clone(&vectors.fast.embedder),
            vectors
                .quality
                .as_ref()
                .map(|tier| Arc::clone(&tier.embedder)),
            limits,
        )
        .await?;
        NativeHybridCandidate::new(cx, self.clone(), next)
    }
}

impl NativeHybridCandidate {
    fn new(
        cx: &Cx,
        base: NativeHybridSnapshot,
        next: NativeBuiltHybridIndex,
    ) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.live.admit_successor")?;
        let before = base.index.vectors();
        let after = next.vectors();
        if after.fast.index.owner_witness().generation.sequence <= base.generation().sequence {
            return Err(invalid(
                "live.generation",
                "not-newer",
                "a live successor must have a strictly higher artifact generation sequence",
            ));
        }
        if before.quality.is_some() != after.quality.is_some() {
            return Err(invalid(
                "live.topology",
                "changed",
                "live replacement must preserve required vector-tier topology",
            ));
        }
        for (old, new) in std::iter::once(&before.fast)
            .chain(before.quality.iter())
            .zip(std::iter::once(&after.fast).chain(after.quality.iter()))
        {
            if old.producer_identity.fingerprint() != new.producer_identity.fingerprint() {
                return Err(invalid(
                    "live.producer",
                    "changed",
                    "live replacement must preserve each tier's complete producing identity",
                ));
            }
        }
        checkpoint(cx, "native_ann.live.successor_ready")?;
        Ok(Self {
            base,
            next: NativeHybridSnapshot {
                index: Arc::new(next),
            },
        })
    }

    /// Inspect, query or seal the completed candidate without selecting it.
    /// `seal_for_reopen` retains its existing trusted-directory contract and
    /// returns a receipt; sealing alone never changes the live selection.
    #[must_use]
    pub fn index(&self) -> &NativeBuiltHybridIndex {
        self.next.index()
    }
}

impl NativeLiveHybridUpdate {
    /// Replace the complete source document for this ID; the last edit wins.
    #[must_use]
    pub fn upsert_document(mut self, document: IndexableDocument) -> Self {
        self.update = self.update.upsert_document(document);
        self
    }

    /// Apply complete-document upserts in iteration order.
    #[must_use]
    pub fn upsert_documents(
        mut self,
        documents: impl IntoIterator<Item = IndexableDocument>,
    ) -> Self {
        self.update = self.update.upsert_documents(documents);
        self
    }

    /// Remove an ID from every successor arm. Old snapshots are unchanged.
    #[must_use]
    pub fn delete_document(mut self, id: impl Into<String>) -> Self {
        self.update = self.update.delete_document(id);
        self
    }

    /// Bound changed-document inference batches.
    ///
    /// # Errors
    /// Refuses zero before building or installing anything.
    pub fn with_batch_size(mut self, size: usize) -> SearchResult<Self> {
        self.update = self.update.with_batch_size(size)?;
        Ok(self)
    }

    /// Bound successor inference input bytes without holding a selection lock.
    ///
    /// This per-update policy has the scope and source-preflight behavior of
    /// [`NativeIndexUpdate::with_max_batch_input_bytes`]. A rejected oversized
    /// source cannot produce an installable candidate or change live selection.
    ///
    /// # Errors
    /// Refuses zero without building or installing anything.
    pub fn with_max_batch_input_bytes(mut self, max_bytes: usize) -> SearchResult<Self> {
        self.update = self.update.with_max_batch_input_bytes(max_bytes)?;
        Ok(self)
    }

    /// Override fast precision/retrieval; a precision change re-embeds that tier.
    #[must_use]
    pub fn with_fast_storage(
        mut self,
        precision: NativeBuildPrecision,
        retrieval: NativeBuildRetrieval,
    ) -> Self {
        self.update = self.update.with_fast_storage(precision, retrieval);
        self
    }

    /// Override quality precision/retrieval without changing tier presence/model.
    ///
    /// # Errors
    /// Refuses an absent quality tier.
    pub fn with_quality_storage(
        mut self,
        precision: NativeBuildPrecision,
        retrieval: NativeBuildRetrieval,
    ) -> SearchResult<Self> {
        self.update = self.update.with_quality_storage(precision, retrieval)?;
        Ok(self)
    }

    /// Build a complete hybrid candidate without holding a serving-selection lock.
    ///
    /// Inference is incremental; vector/graph and lexical rebuilding still have
    /// full-cohort costs. Dropping this pending future cannot install a partial
    /// candidate. Finish with [`NativeLiveHybridIndex::install`], which refuses
    /// this candidate if another update has won in the meantime.
    ///
    /// # Errors
    /// Propagates all required source/vector/Quill build and admission failures.
    pub async fn build(self, cx: &Cx) -> SearchResult<NativeHybridCandidate> {
        let next = Box::pin(self.update.build_hybrid(cx)).await?;
        NativeHybridCandidate::new(cx, self.base, next)
    }
}

fn lock_error(error: RwLockError) -> SearchError {
    match error {
        RwLockError::Cancelled => SearchError::Cancelled {
            phase: "native_ann.live.lock".to_owned(),
            reason: "cancelled while acquiring the live generation selection".to_owned(),
        },
        error => invalid("live.lock", "acquisition", &error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::task::{Context, Poll, Waker};

    use frankensearch_core::generation::{
        EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1, EmbeddingSpaceKindV1,
    };
    use frankensearch_core::traits::{IdentityBoundEmbedding, ModelCategory, SearchFuture};
    use frankensearch_index::native_hnsw::HnswParams;

    use crate::Embedder;
    use crate::native_ann::NativeSearchPhase;
    use crate::native_ann::builder::NativeIndexBuilder;

    struct Provider {
        identity: EmbeddingIdentityBundleV1,
        documents: AtomicUsize,
        queries: AtomicUsize,
        query_drops: AtomicUsize,
        batch_drops: AtomicUsize,
        hold_queries: AtomicBool,
        hold_batches: AtomicBool,
        fail_batches: AtomicBool,
    }

    struct CountDrop<'a>(&'a AtomicUsize);
    impl Drop for CountDrop<'_> {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl Provider {
        fn new(name: &str, dimension: u32) -> Arc<Self> {
            let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(name, dimension);
            // Synthetic semantic-shaped identities, not real-model quality evidence.
            identity.space.kind = EmbeddingSpaceKindV1::Semantic;
            identity.space.hash_control = None;
            identity.space.artifact_manifest_fingerprint = "a".repeat(64);
            identity.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
                role: "weights".to_owned(),
                sha256: "b".repeat(64),
                size: 1,
            }];
            identity.producer.space_fingerprint = identity.space.fingerprint();
            identity.validate().unwrap();
            Arc::new(Self {
                identity,
                documents: AtomicUsize::new(0),
                queries: AtomicUsize::new(0),
                query_drops: AtomicUsize::new(0),
                batch_drops: AtomicUsize::new(0),
                hold_queries: AtomicBool::new(false),
                hold_batches: AtomicBool::new(false),
                fail_batches: AtomicBool::new(false),
            })
        }

        fn values(&self, text: &str) -> Vec<f32> {
            let mut values = vec![0.0; self.dimension()];
            values[usize::from(text.contains("vertical"))] = 1.0;
            values
        }
    }

    impl Embedder for Provider {
        fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async move {
                self.queries.fetch_add(1, Ordering::SeqCst);
                let _guard = CountDrop(&self.query_drops);
                if self.hold_queries.load(Ordering::SeqCst) {
                    std::future::pending::<()>().await;
                }
                Ok(self.values(text))
            })
        }

        fn embed_batch_bound<'a>(
            &'a self,
            _cx: &'a Cx,
            texts: &'a [&'a str],
        ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
            Box::pin(async move {
                self.documents.fetch_add(texts.len(), Ordering::SeqCst);
                let _guard = CountDrop(&self.batch_drops);
                if self.hold_batches.load(Ordering::SeqCst) {
                    std::future::pending::<()>().await;
                }
                if self.fail_batches.load(Ordering::SeqCst) {
                    return Err(invalid(
                        "live.test_provider",
                        "failed",
                        "required tier failed",
                    ));
                }
                Ok(texts
                    .iter()
                    .map(|text| IdentityBoundEmbedding {
                        identity: self.identity.clone(),
                        values: self.values(text),
                    })
                    .collect())
            })
        }

        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(&self.identity)
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
            true
        }
        fn category(&self) -> ModelCategory {
            ModelCategory::TransformerEmbedder
        }
    }

    fn generation(sequence: u64) -> ArtifactGenerationIdentityV1 {
        ArtifactGenerationIdentityV1::new(sequence, [0x71; 16]).unwrap()
    }

    async fn build(
        cx: &Cx,
        path: &Path,
        sequence: u64,
        fast: &Arc<Provider>,
        quality: Option<&Arc<Provider>>,
    ) -> NativeBuiltHybridIndex {
        let mut builder = NativeIndexBuilder::new(path, generation(sequence), fast.clone())
            .unwrap()
            .add_documents([
                IndexableDocument::new("a", "legacyterm horizontal")
                    .with_metadata("version", "old"),
                IndexableDocument::new("b", "vertical"),
                IndexableDocument::new("c", "horizontal"),
            ])
            .with_fast_storage(
                NativeBuildPrecision::F16,
                NativeBuildRetrieval::Hnsw {
                    params: HnswParams::default(),
                    seed: 7,
                },
            );
        if let Some(quality) = quality {
            builder = builder.with_quality_embedder(quality.clone()).unwrap();
        }
        builder.build_hybrid(cx).await.unwrap()
    }

    async fn fixture(
        cx: &Cx,
        root: &Path,
    ) -> (NativeLiveHybridIndex, Arc<Provider>, Arc<Provider>) {
        let fast = Provider::new("live-fast", 2);
        let quality = Provider::new("live-quality", 3);
        let initial = build(cx, &root.join("initial"), 1, &fast, Some(&quality)).await;
        (
            NativeLiveHybridIndex::new(cx, initial).unwrap(),
            fast,
            quality,
        )
    }

    fn assert_refusal(error: SearchError, field: &str) {
        assert!(
            matches!(error, SearchError::InvalidConfig { field: actual, .. }
            if actual == field)
        );
    }

    fn assert_rows(page: &NativeHybridResults) {
        for result in &page.results {
            if let Some(row) = result.index {
                let vectors = page.snapshot.index().vectors();
                let tier = if result.fast_score.is_some() {
                    vectors.fast()
                } else {
                    assert!(result.quality_score.is_some());
                    vectors.quality().unwrap()
                };
                assert_eq!(
                    tier.index
                        .owner
                        .doc_id_at(usize::try_from(row).unwrap())
                        .unwrap(),
                    result.doc_id
                );
            }
        }
    }

    #[test]
    fn live_install_keeps_old_phases_and_pages_on_their_complete_source_cohort() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            fn send_sync<T: Send + Sync>() {}
            send_sync::<NativeLiveHybridIndex>();
            send_sync::<NativeHybridSnapshot>();
            let dir = tempfile::tempdir().unwrap();
            let (live, fast, quality) = fixture(&cx, dir.path()).await;
            let old_page = live.search(&cx, "vertical", 1).await.unwrap();
            let old = live.snapshot(&cx).await.unwrap();
            let mut stream = old.index().progressive(&cx, "vertical", 1).unwrap();
            let Some(NativeSearchPhase::Initial { results, .. }) =
                stream.next_phase().await.unwrap()
            else {
                panic!("initial phase");
            };
            assert_eq!(results[0].doc_id, "b");
            let candidate = old
                .begin_update(&cx, dir.path().join("successor"), generation(2))
                .unwrap()
                .upsert_documents([
                    IndexableDocument::new("a", "newterm vertical").with_metadata("version", "new"),
                    IndexableDocument::new("d", "novelterm horizontal"),
                ])
                .delete_document("b")
                .with_batch_size(1)
                .unwrap()
                .build(&cx)
                .await
                .unwrap();
            assert_eq!(fast.documents.load(Ordering::SeqCst), 5);
            assert_eq!(quality.documents.load(Ordering::SeqCst), 5);
            assert!(Arc::ptr_eq(
                &live.snapshot(&cx).await.unwrap().index,
                &old.index
            ));
            let installed = live.install(&cx, &candidate).await.unwrap();
            assert_eq!(installed.generation(), generation(2));
            assert!(installed.index().vectors().document("b").is_none());
            assert!(old.index().vectors().document("b").is_some());
            assert_eq!(
                old.index().vectors().document("a").unwrap().metadata["version"],
                "old"
            );
            assert_eq!(
                installed.index().vectors().document("a").unwrap().metadata["version"],
                "new"
            );
            assert_eq!(
                old.index()
                    .lexical()
                    .search(&cx, "legacyterm", 10)
                    .await
                    .unwrap()
                    .len(),
                1
            );
            assert!(
                installed
                    .index()
                    .lexical()
                    .search(&cx, "legacyterm", 10)
                    .await
                    .unwrap()
                    .is_empty()
            );
            for page in [
                live.search(&cx, "vertical", 3).await.unwrap(),
                live.search_refined(&cx, "vertical", 3).await.unwrap(),
                live.search_quality(&cx, "vertical", 3).await.unwrap(),
            ] {
                assert!(Arc::ptr_eq(&page.snapshot.index, &installed.index));
                assert_eq!(page.results[0].doc_id, "a");
                assert!(page.results.iter().all(|hit| hit.doc_id != "b"));
                assert_rows(&page);
            }
            let Some(NativeSearchPhase::Refined { results, .. }) =
                stream.next_phase().await.unwrap()
            else {
                panic!("old retained refinement");
            };
            assert_eq!(results[0].doc_id, "b");
            assert_eq!(old_page.snapshot.generation(), generation(1));
            assert_eq!(old_page.results[0].doc_id, "b");
            assert_rows(&old_page);
            assert!(stream.next_phase().await.unwrap().is_none());
        });
    }

    #[test]
    fn higher_sequence_stale_update_cannot_discard_the_winning_edits() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let (live, _, _) = fixture(&cx, dir.path()).await;
            let base = live.snapshot(&cx).await.unwrap();
            let first = base
                .begin_update(&cx, dir.path().join("first"), generation(2))
                .unwrap()
                .upsert_document(IndexableDocument::new("winner", "horizontal"))
                .build(&cx)
                .await
                .unwrap();
            let late = base
                .begin_update(&cx, dir.path().join("late"), generation(9))
                .unwrap()
                .upsert_document(IndexableDocument::new("loser", "vertical"))
                .build(&cx)
                .await
                .unwrap();
            let current = live.install(&cx, &first).await.unwrap();
            for candidate in [&late, &first] {
                assert_refusal(
                    live.install(&cx, candidate).await.unwrap_err(),
                    "native_ann.live.expected_current",
                );
            }
            let after = live.snapshot(&cx).await.unwrap();
            assert!(Arc::ptr_eq(&after.index, &current.index));
            assert!(after.index().vectors().document("winner").is_some());
            assert!(after.index().vectors().document("loser").is_none());
            // The caller can still inspect a refused, fully built candidate.
            assert!(late.index().vectors().document("loser").is_some());
            let rebased = after
                .begin_update(&cx, dir.path().join("rebased"), generation(10))
                .unwrap()
                .upsert_document(IndexableDocument::new("loser", "vertical"))
                .build(&cx)
                .await
                .unwrap();
            let final_view = live.install(&cx, &rebased).await.unwrap();
            assert!(final_view.index().vectors().document("winner").is_some());
            assert!(final_view.index().vectors().document("loser").is_some());
        });
    }

    #[test]
    fn equal_generation_and_vector_bytes_do_not_authorize_another_live_handle() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let (live, fast, quality) = fixture(&cx, dir.path()).await;
            let other = NativeLiveHybridIndex::new(
                &cx,
                build(&cx, &dir.path().join("other"), 1, &fast, Some(&quality)).await,
            )
            .unwrap();
            let base = live.snapshot(&cx).await.unwrap();
            let foreign = other.snapshot(&cx).await.unwrap();
            assert_eq!(base.generation(), foreign.generation());
            assert_eq!(
                base.index().vectors().fast().index().owner_witness(),
                foreign.index().vectors().fast().index().owner_witness()
            );
            let candidate = base
                .begin_update(&cx, dir.path().join("next"), generation(2))
                .unwrap()
                .delete_document("b")
                .build(&cx)
                .await
                .unwrap();
            assert_refusal(
                other.install(&cx, &candidate).await.unwrap_err(),
                "native_ann.live.expected_current",
            );
            assert!(Arc::ptr_eq(
                &other.snapshot(&cx).await.unwrap().index,
                &foreign.index
            ));
            live.install(&cx, &candidate).await.unwrap();
        });
    }

    #[test]
    fn pending_query_does_not_block_installation_or_a_new_query() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let (live, fast, _) = fixture(&cx, dir.path()).await;
            let base = live.snapshot(&cx).await.unwrap();
            let candidate = base
                .begin_update(&cx, dir.path().join("next"), generation(2))
                .unwrap()
                .delete_document("b")
                .build(&cx)
                .await
                .unwrap();
            fast.hold_queries.store(true, Ordering::SeqCst);
            let mut query = Box::pin(live.search(&cx, "vertical", 3));
            assert!(
                query
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            assert_eq!(fast.queries.load(Ordering::SeqCst), 1);
            let mut install = Box::pin(live.install(&cx, &candidate));
            let Poll::Ready(Ok(installed)) = install
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
            else {
                panic!("inference must not retain the selection lock");
            };
            drop(install);
            fast.hold_queries.store(false, Ordering::SeqCst);
            // The first query is still pending inside its provider, but a fresh
            // query must use the complete new cohort without waiting for it.
            let page = live.search_refined(&cx, "horizontal", 3).await.unwrap();
            assert!(Arc::ptr_eq(&page.snapshot.index, &installed.index));
            assert!(page.results.iter().all(|result| result.doc_id != "b"));
            let drops = fast.query_drops.load(Ordering::SeqCst);
            drop(query);
            assert_eq!(fast.query_drops.load(Ordering::SeqCst), drops + 1);
            assert!(base.index().vectors().document("b").is_some());
        });
    }

    #[test]
    fn cancelled_and_dropped_lock_waiters_never_replace_the_current_generation() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let (live, _, _) = fixture(&cx, dir.path()).await;
            let base = live.snapshot(&cx).await.unwrap();
            let candidate = base
                .begin_update(&cx, dir.path().join("next"), generation(2))
                .unwrap()
                .delete_document("b")
                .build(&cx)
                .await
                .unwrap();
            let held = live.current.write(&cx).await.unwrap();
            let mut reader = Box::pin(live.snapshot(&cx));
            assert!(
                reader
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            drop(reader);
            let mut abandoned = Box::pin(live.install(&cx, &candidate));
            assert!(
                abandoned
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            drop(abandoned);
            let mut cancelled = Box::pin(live.install(&cx, &candidate));
            assert!(
                cancelled
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            cx.set_cancel_requested(true);
            assert!(matches!(
                cancelled
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop())),
                Poll::Ready(Err(SearchError::Cancelled { .. }))
            ));
            drop(cancelled);
            drop(held);
            cx.set_cancel_requested(false);
            assert!(Arc::ptr_eq(
                &live.snapshot(&cx).await.unwrap().index,
                &base.index
            ));
            // No failed attempt consumed or partially installed the candidate.
            live.install(&cx, &candidate).await.unwrap();
            cx.set_cancel_requested(true);
            assert!(matches!(
                live.snapshot(&cx).await,
                Err(SearchError::Cancelled { .. })
            ));
            assert!(matches!(
                live.install(&cx, &candidate).await,
                Err(SearchError::Cancelled { .. })
            ));
            cx.set_cancel_requested(false);
            assert_eq!(
                live.snapshot(&cx).await.unwrap().generation(),
                generation(2)
            );
        });
    }

    #[test]
    fn pending_and_failed_required_builds_leave_live_queries_on_the_predecessor() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let (live, fast, quality) = fixture(&cx, dir.path()).await;
            let base = live.snapshot(&cx).await.unwrap();
            let update = base
                .begin_update(&cx, dir.path().join("abandoned"), generation(2))
                .unwrap()
                .upsert_document(IndexableDocument::new("new", "vertical"));
            fast.hold_batches.store(true, Ordering::SeqCst);
            let before = fast.batch_drops.load(Ordering::SeqCst);
            let mut building = Box::pin(update.build(&cx));
            assert!(
                building
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            let page = live.search(&cx, "vertical", 3).await.unwrap();
            assert_eq!(page.results[0].doc_id, "b");
            assert!(Arc::ptr_eq(&page.snapshot.index, &base.index));
            drop(building);
            assert_eq!(fast.batch_drops.load(Ordering::SeqCst), before + 1);
            fast.hold_batches.store(false, Ordering::SeqCst);
            quality.fail_batches.store(true, Ordering::SeqCst);
            let error = base
                .begin_update(&cx, dir.path().join("failed-quality"), generation(2))
                .unwrap()
                .upsert_document(IndexableDocument::new("new", "vertical"))
                .build(&cx)
                .await
                .unwrap_err();
            assert_refusal(error, "native_ann.live.test_provider");
            quality.fail_batches.store(false, Ordering::SeqCst);
            assert!(Arc::ptr_eq(
                &live.snapshot(&cx).await.unwrap().index,
                &base.index
            ));
            assert!(base.index().vectors().document("new").is_none());
            assert_eq!(
                live.search_quality(&cx, "vertical", 1)
                    .await
                    .unwrap()
                    .results[0]
                    .doc_id,
                "b"
            );
        });
    }

    #[test]
    fn selected_successor_reopens_without_inference_and_rejects_bad_receipts_and_rollback() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let (live, fast, quality) = fixture(&cx, dir.path()).await;
            let base = live.snapshot(&cx).await.unwrap();
            let old_receipt = base.index().seal_for_reopen(&cx).unwrap();
            let next_path = dir.path().join("sealed-next");
            let candidate = base
                .begin_update(&cx, &next_path, generation(2))
                .unwrap()
                .upsert_document(IndexableDocument::new("a", "newterm vertical"))
                .build(&cx)
                .await
                .unwrap();
            let receipt = candidate.index().seal_for_reopen(&cx).unwrap();
            drop(candidate);
            let before = (
                fast.documents.load(Ordering::SeqCst),
                quality.documents.load(Ordering::SeqCst),
            );
            let mut bad = receipt;
            bad.sha256[0] ^= 1;
            let limits = NativeHybridReopenLimits::default();
            assert!(
                base.prepare_selected(&cx, &next_path, &bad, limits)
                    .await
                    .is_err()
            );
            assert!(Arc::ptr_eq(
                &live.snapshot(&cx).await.unwrap().index,
                &base.index
            ));
            let reopened = base
                .prepare_selected(&cx, &next_path, &receipt, limits)
                .await
                .unwrap();
            assert_eq!(
                (
                    fast.documents.load(Ordering::SeqCst),
                    quality.documents.load(Ordering::SeqCst)
                ),
                before
            );
            assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
            assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
            let installed = live.install(&cx, &reopened).await.unwrap();
            let page = live.search_refined(&cx, "newterm", 1).await.unwrap();
            assert_eq!(page.results[0].doc_id, "a");
            assert!(Arc::ptr_eq(&page.snapshot.index, &installed.index));
            assert_rows(&page);
            let rollback = installed
                .prepare_selected(
                    &cx,
                    base.index().vectors().directory(),
                    &old_receipt,
                    limits,
                )
                .await
                .unwrap_err();
            assert_refusal(rollback, "native_ann.live.generation");
            assert!(Arc::ptr_eq(
                &live.snapshot(&cx).await.unwrap().index,
                &installed.index
            ));
        });
    }

    #[test]
    fn selected_reopen_cannot_drop_quality_or_substitute_a_same_dimension_producer() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let (live, fast, quality) = fixture(&cx, dir.path()).await;
            let base = live.snapshot(&cx).await.unwrap();
            let foreign = Provider::new("different-live-fast", 2);
            for (name, provider, quality) in [
                ("no-quality", &fast, None),
                ("foreign-producer", &foreign, Some(&quality)),
            ] {
                let path = dir.path().join(name);
                let candidate = build(&cx, &path, 2, provider, quality).await;
                let receipt = candidate.seal_for_reopen(&cx).unwrap();
                drop(candidate);
                assert!(
                    base.prepare_selected(
                        &cx,
                        &path,
                        &receipt,
                        NativeHybridReopenLimits::default()
                    )
                    .await
                    .is_err()
                );
                assert!(Arc::ptr_eq(
                    &live.snapshot(&cx).await.unwrap().index,
                    &base.index
                ));
            }
        });
    }
}
