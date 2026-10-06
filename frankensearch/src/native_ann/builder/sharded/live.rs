//! Process-local selection of complete partitioned native hybrid generations.
//!
//! A query owns one source/vector/graph/global-Quill snapshot through every
//! phase. Repartitioning is prepared through the existing update builder, never
//! by independently swapping shard sets or a lexical reader. Installation is a
//! short, cancel-aware exact-predecessor comparison; preparation holds no lock.
//!
//! This is not durable publication, a rollback floor, or disk retention. Keep
//! mapped Quill files immutable and present while any snapshot lives. Sealing
//! and externally selecting a receipt remain the caller's responsibility. The
//! library creates no runtime or tasks; blocking work stays on the caller's lane.
//!
//! ```ignore
//! use frankensearch::native_ann::builder::sharded::live::NativeLiveShardedHybridIndex;
//!
//! // initial is a complete built or explicitly reopened sharded hybrid cohort.
//! let live = NativeLiveShardedHybridIndex::new(&cx, initial)?;
//! let old = live.snapshot(&cx).await?;
//! let candidate = old.begin_update(&cx, &new_directory, next_generation)?
//!     .upsert_document(revised_document)
//!     .delete_document("obsolete-id")
//!     .with_batch_size(16)?
//!     .build(&cx, 5_000).await?;
//! // Optionally seal and persist its receipt using the caller's authority.
//! // That persistence alone does not change this live handle.
//! let installed = live.install(&cx, &candidate).await?;
//! // old still owns its original sources, models, graphs, Quill and row maps.
//! let page = live.search_refined(&cx, "retry failed requests", 10).await?;
//! // Interpret both physical coordinates through page.snapshot, not live.
//! ```

use std::fmt;
use std::path::Path;
use std::sync::Arc;

use asupersync::sync::{RwLock, RwLockError};
use frankensearch_core::generation::ArtifactGenerationIdentityV1;

use super::super::{NativeBuildPrecision, NativeBuildRetrieval, NativeIndexUpdate};
use super::{NativeBuiltShardedHybridIndex, NativeBuiltShardedIndex};
use crate::native_ann::{NativeShardedResult, checkpoint, invalid};
use crate::{Cx, IndexableDocument, SearchError, SearchResult};

/// Live selection of an entire admitted source/vector/Quill partition inventory.
///
/// Share this handle with `Arc`. Snapshot acquisition clones only one owner;
/// inference, graph work, hydration, building and destruction never hold the
/// selection lock. The caller schedules concurrent work through its own runtime.
#[derive(Debug)]
pub struct NativeLiveShardedHybridIndex {
    current: RwLock<NativeShardedSnapshot>,
}

/// Owned pin of the complete generation, including both physical shard maps.
///
/// Obtain this once for progressive/reranked queries and keep it through all
/// phases and row interpretation. A replacement may change every shard ordinal;
/// a result's `fast_row`/`quality_row` belongs to THIS snapshot, not the live head.
#[derive(Clone)]
pub struct NativeShardedSnapshot {
    index: Arc<NativeBuiltShardedHybridIndex>,
}

impl fmt::Debug for NativeShardedSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeShardedSnapshot")
            .field("generation", &self.generation())
            .field("partitions", &self.index.vectors().partitions().len())
            .field("documents", &self.index.vectors().document_count())
            .finish_non_exhaustive()
    }
}

/// Ranked results and the retained owners needed to resolve their shard rows.
#[derive(Debug)]
pub struct NativeShardedResults {
    /// The complete generation that actually ran this query.
    pub snapshot: NativeShardedSnapshot,
    /// Global rankings with independently scoped fast/quality shard coordinates.
    pub results: Vec<NativeShardedResult>,
}

/// Fully built successor bound to the exact live predecessor used to prepare it.
///
/// Fields are private. A caller cannot install a vector-only build, replace a
/// candidate's predecessor, or turn a rejected candidate into a newer transaction
/// by editing its generation number. Refusal does not consume the candidate.
#[derive(Debug)]
pub struct NativeShardedCandidate {
    base: NativeShardedSnapshot,
    next: NativeShardedSnapshot,
}

/// Source edits against one retained complete sharded predecessor.
///
/// This delegates to `NativeIndexUpdate`: reuse follows source ID/content and
/// each original producer/precision, not the previous partition or physical row.
/// Only a complete hybrid build can become an installable candidate.
pub struct NativeLiveShardedUpdate {
    base: NativeShardedSnapshot,
    update: NativeIndexUpdate,
}

impl NativeLiveShardedHybridIndex {
    /// Select an already built or explicitly reopened complete initial cohort.
    /// No discovery, inference, file writes or background work occurs.
    ///
    /// # Errors
    /// Returns cancellation before the live handle exists.
    pub fn new(cx: &Cx, initial: NativeBuiltShardedHybridIndex) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.sharded_live.new")?;
        Ok(Self {
            current: RwLock::new(NativeShardedSnapshot {
                index: Arc::new(initial),
            }),
        })
    }

    /// Pin one complete generation, releasing the lock before returning.
    ///
    /// # Errors
    /// Returns cancellation or selection-lock errors. Dropped pending reads
    /// neither retain a lock nor modify the live selection.
    pub async fn snapshot(&self, cx: &Cx) -> SearchResult<NativeShardedSnapshot> {
        checkpoint(cx, "native_ann.sharded_live.snapshot")?;
        let current = self.current.read(cx).await.map_err(lock_error)?;
        checkpoint(cx, "native_ann.sharded_live.snapshot_acquired")?;
        let snapshot = (*current).clone();
        drop(current);
        Ok(snapshot)
    }

    /// Install the complete candidate only if its exact predecessor is current.
    ///
    /// Sequence ordering alone never authorizes discarding another writer's
    /// edits. A candidate from another live handle is refused even if both heads
    /// have identical generations and bytes. Build again from a fresh snapshot
    /// after a stale refusal; do not relabel the old candidate.
    ///
    /// The final checkpoint precedes the swap. There is no cancellation point
    /// after visible installation, and the old owner is dropped after unlocking.
    /// Existing snapshots keep serving their original files and row coordinates.
    /// This does not publish a durable pointer or permit deleting old files.
    ///
    /// # Errors
    /// Returns cancellation, lock failure or stale/foreign predecessor refusal.
    pub async fn install(
        &self,
        cx: &Cx,
        candidate: &NativeShardedCandidate,
    ) -> SearchResult<NativeShardedSnapshot> {
        checkpoint(cx, "native_ann.sharded_live.install")?;
        let mut current = self.current.write(cx).await.map_err(lock_error)?;
        checkpoint(cx, "native_ann.sharded_live.install_acquired")?;
        if !Arc::ptr_eq(&current.index, &candidate.base.index) {
            return Err(invalid(
                "sharded_live.expected_current",
                "stale-or-foreign",
                "the candidate must retain this live handle's exact current predecessor",
            ));
        }
        checkpoint(cx, "native_ann.sharded_live.install_commit")?;
        let installed = candidate.next.clone();
        let previous = std::mem::replace(&mut *current, installed.clone());
        drop(current);
        drop(previous);
        Ok(installed)
    }

    /// Fast-plus-global-lexical search with its originating snapshot.
    ///
    /// # Errors
    /// Propagates snapshot acquisition and complete sharded search errors.
    pub async fn search(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
    ) -> SearchResult<NativeShardedResults> {
        let snapshot = self.snapshot(cx).await?;
        let results = snapshot.index.search(cx, text, k).await?;
        Ok(NativeShardedResults { snapshot, results })
    }

    /// Independent fast/quality retrieval and global lexical fusion on one pin.
    ///
    /// # Errors
    /// Propagates all configured tier, lexical, hydration and cancellation errors.
    pub async fn search_refined(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
    ) -> SearchResult<NativeShardedResults> {
        let snapshot = self.snapshot(cx).await?;
        let results = snapshot.index.search_refined(cx, text, k).await?;
        Ok(NativeShardedResults { snapshot, results })
    }

    /// Quality-primary search, without fast inference, on one retained cohort.
    ///
    /// # Errors
    /// Refuses absent quality and propagates acquisition and query failures.
    pub async fn search_quality(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
    ) -> SearchResult<NativeShardedResults> {
        let snapshot = self.snapshot(cx).await?;
        let results = snapshot.index.search_quality(cx, text, k).await?;
        Ok(NativeShardedResults { snapshot, results })
    }
}

impl NativeShardedSnapshot {
    /// The complete read-only cohort for progressive phases and row resolution.
    #[must_use]
    pub fn index(&self) -> &NativeBuiltShardedHybridIndex {
        &self.index
    }

    /// Full generation identity shared by all partitions, including its nonce.
    #[must_use]
    pub fn generation(&self) -> ArtifactGenerationIdentityV1 {
        self.index.vectors().generation()
    }

    /// Stage source edits against this exact predecessor without locking selection.
    ///
    /// Uses the existing sharded update admission, including every partition's
    /// original producer, membership and inherited storage policy. Changing the
    /// model or tier topology requires a separately opened live handle.
    ///
    /// # Errors
    /// Propagates generation, producer, membership, policy and cancellation errors.
    pub fn begin_update(
        &self,
        cx: &Cx,
        directory: impl AsRef<Path>,
        generation: ArtifactGenerationIdentityV1,
    ) -> SearchResult<NativeLiveShardedUpdate> {
        let update = self.index.begin_update(cx, directory, generation)?;
        Ok(NativeLiveShardedUpdate {
            base: self.clone(),
            update,
        })
    }
}

impl NativeShardedCandidate {
    fn new(
        cx: &Cx,
        base: NativeShardedSnapshot,
        next: NativeBuiltShardedHybridIndex,
    ) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.sharded_live.admit_successor")?;
        let before = base.index.vectors();
        let after = next.vectors();
        if after.generation().sequence <= base.generation().sequence {
            return Err(invalid(
                "sharded_live.generation",
                "not-newer",
                "a live successor requires a strictly higher generation sequence",
            ));
        }
        if before.quality().is_some() != after.quality().is_some() {
            return Err(invalid(
                "sharded_live.topology",
                "changed",
                "live replacement must preserve required vector-tier topology",
            ));
        }
        validate_producers(cx, before, after)?;
        checkpoint(cx, "native_ann.sharded_live.successor_ready")?;
        Ok(Self {
            base,
            next: NativeShardedSnapshot {
                index: Arc::new(next),
            },
        })
    }

    /// Query or seal the complete candidate without selecting it.
    /// Sealing returns the existing hybrid receipt; it does not activate this pin.
    #[must_use]
    pub fn index(&self) -> &NativeBuiltShardedHybridIndex {
        self.next.index()
    }
}

fn validate_producers(
    cx: &Cx,
    before: &NativeBuiltShardedIndex,
    after: &NativeBuiltShardedIndex,
) -> SearchResult<()> {
    // Complete typed cohorts already require one identity-bearing partition
    // and matching per-tier identities across their inventories. Compare the
    // captured producing identity, never a new same-dimensional model's name.
    let first = &before.partitions()[0];
    let fast = first.fast.producer_identity.fingerprint();
    let quality = first
        .quality
        .as_ref()
        .map(|tier| tier.producer_identity.fingerprint());
    for partition in after.partitions() {
        checkpoint(cx, "native_ann.sharded_live.producer_join")?;
        if partition.fast.producer_identity.fingerprint() != fast
            || partition
                .quality
                .as_ref()
                .map(|tier| tier.producer_identity.fingerprint())
                != quality
        {
            return Err(invalid(
                "sharded_live.producer",
                "changed",
                "every successor partition must preserve both original producing identities",
            ));
        }
    }
    Ok(())
}

impl NativeLiveShardedUpdate {
    /// Replace a complete source record; the last edit for an ID wins.
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

    /// Remove an ID from every successor component, leaving old snapshots intact.
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

    /// Bound prepared input bytes under the existing update preflight policy.
    ///
    /// # Errors
    /// Refuses zero. Oversized surviving sources fail before candidate creation.
    pub fn with_max_batch_input_bytes(mut self, max_bytes: usize) -> SearchResult<Self> {
        self.update = self.update.with_max_batch_input_bytes(max_bytes)?;
        Ok(self)
    }

    /// Change fast storage/retrieval, not its model. Precision changes re-embed it.
    #[must_use]
    pub fn with_fast_storage(
        mut self,
        precision: NativeBuildPrecision,
        retrieval: NativeBuildRetrieval,
    ) -> Self {
        self.update = self.update.with_fast_storage(precision, retrieval);
        self
    }

    /// Change quality storage/retrieval while preserving required tier presence.
    ///
    /// # Errors
    /// Refuses a quality policy on a fast-only cohort.
    pub fn with_quality_storage(
        mut self,
        precision: NativeBuildPrecision,
        retrieval: NativeBuildRetrieval,
    ) -> SearchResult<Self> {
        self.update = self.update.with_quality_storage(precision, retrieval)?;
        Ok(self)
    }

    /// Build every successor partition and one global Quill index before admission.
    ///
    /// The explicit positive partition size may differ from the predecessor's.
    /// Reuse follows source IDs across changed boundaries. Inference is incremental;
    /// artifact writes and lexical construction remain full-cohort work. A late
    /// failure or dropped future leaves no installable prefix and never selects it.
    /// Finish with `NativeLiveShardedHybridIndex::install`, which rechecks staleness.
    ///
    /// # Errors
    /// Propagates partition limits, source, required-tier, lexical and cancellation
    /// errors. Failed new directories remain for inspection and are not overwritten.
    pub async fn build(
        self,
        cx: &Cx,
        max_documents_per_shard: usize,
    ) -> SearchResult<NativeShardedCandidate> {
        let next = Box::pin(self.update.build_sharded_hybrid(cx, max_documents_per_shard)).await?;
        NativeShardedCandidate::new(cx, self.base, next)
    }
}

fn lock_error(error: RwLockError) -> SearchError {
    match error {
        RwLockError::Cancelled => SearchError::Cancelled {
            phase: "native_ann.sharded_live.lock".to_owned(),
            reason: "cancelled while acquiring the sharded generation selection".to_owned(),
        },
        error => invalid("sharded_live.lock", "acquisition", &error.to_string()),
    }
}

#[cfg(test)]
mod tests;
