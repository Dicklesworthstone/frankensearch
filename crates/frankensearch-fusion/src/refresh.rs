//! Index refresh worker (asupersync background task).
//!
//! [`RefreshWorker`] periodically drains the [`EmbeddingQueue`],
//! embeds documents in batches, and rebuilds the vector index. It runs as an
//! asupersync task within a structured concurrency region.
//!
//! # Single-writer guarantee
//!
//! The worker is the **only** component that writes to vector indices. All
//! reads go through the [`IndexCache`] which provides
//! atomic snapshot replacement.
//!
//! # Lifecycle
//!
//! The worker loops until the parent `Cx` is cancelled. On cancellation it
//! finishes the current batch before exiting.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use asupersync::Cx;
use tracing::{debug, error, info, warn};

use frankensearch_core::config::TwoTierConfig;
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::traits::Embedder;
use frankensearch_index::{
    TwoTierIndex, VECTOR_INDEX_FALLBACK_FILENAME, VECTOR_INDEX_FAST_FILENAME,
    VECTOR_INDEX_QUALITY_FILENAME, VectorIndex,
};

use crate::cache::IndexCache;
use crate::queue::{EmbeddingJob, EmbeddingQueue, JobOutcome};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the index refresh worker.
#[derive(Debug, Clone)]
pub struct RefreshWorkerConfig {
    /// How often to poll the queue for new jobs. Default: 1000ms.
    pub poll_interval: Duration,
    /// Maximum documents to embed per refresh cycle. Default: 1000.
    pub max_docs_per_cycle: usize,
    /// Directory where vector indices are written. Required.
    pub index_dir: PathBuf,
    /// `TwoTierConfig` for newly built indices.
    pub index_config: TwoTierConfig,
}

impl RefreshWorkerConfig {
    /// Create a config with the given index directory and defaults.
    #[must_use]
    pub fn new(index_dir: impl Into<PathBuf>) -> Self {
        Self {
            poll_interval: Duration::from_secs(1),
            max_docs_per_cycle: 1000,
            index_dir: index_dir.into(),
            index_config: TwoTierConfig::default(),
        }
    }

    /// Override the poll interval.
    #[must_use]
    pub const fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Override the max docs per cycle.
    #[must_use]
    pub const fn with_max_docs_per_cycle(mut self, max: usize) -> Self {
        self.max_docs_per_cycle = max;
        self
    }

    /// Override the index config.
    #[must_use]
    pub fn with_index_config(mut self, config: TwoTierConfig) -> Self {
        self.index_config = config;
        self
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Lock-free counters for refresh worker telemetry.
#[derive(Debug, Default)]
pub struct RefreshMetrics {
    /// Total refresh cycles executed.
    pub cycles: AtomicU64,
    /// Total documents embedded.
    pub docs_embedded: AtomicU64,
    /// Total documents that failed embedding.
    pub docs_failed: AtomicU64,
    /// Total index rebuilds (successful).
    pub index_rebuilds: AtomicU64,
    /// Total index rebuild failures.
    pub rebuild_failures: AtomicU64,
    /// Total embedding time in microseconds.
    pub embed_time_us: AtomicU64,
    /// Total rebuild time in microseconds.
    pub rebuild_time_us: AtomicU64,
}

impl RefreshMetrics {
    /// Snapshot of the current metrics.
    #[must_use]
    pub fn snapshot(&self) -> RefreshMetricsSnapshot {
        RefreshMetricsSnapshot {
            cycles: self.cycles.load(Ordering::Relaxed),
            docs_embedded: self.docs_embedded.load(Ordering::Relaxed),
            docs_failed: self.docs_failed.load(Ordering::Relaxed),
            index_rebuilds: self.index_rebuilds.load(Ordering::Relaxed),
            rebuild_failures: self.rebuild_failures.load(Ordering::Relaxed),
            embed_time_us: self.embed_time_us.load(Ordering::Relaxed),
            rebuild_time_us: self.rebuild_time_us.load(Ordering::Relaxed),
        }
    }
}

/// Point-in-time snapshot of refresh metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefreshMetricsSnapshot {
    /// Total refresh cycles executed.
    pub cycles: u64,
    /// Total documents embedded.
    pub docs_embedded: u64,
    /// Total documents that failed embedding.
    pub docs_failed: u64,
    /// Total index rebuilds (successful).
    pub index_rebuilds: u64,
    /// Total index rebuild failures.
    pub rebuild_failures: u64,
    /// Total embedding time in microseconds.
    pub embed_time_us: u64,
    /// Total rebuild time in microseconds.
    pub rebuild_time_us: u64,
}

// ---------------------------------------------------------------------------
// Embedded record (intermediate)
// ---------------------------------------------------------------------------

/// A document with its computed embedding, ready for index insertion.
#[derive(Debug)]
struct EmbeddedRecord {
    doc_id: String,
    fast_embedding: Vec<f32>,
    quality_embedding: Option<Vec<f32>>,
    content_hash: String,
}

/// Explain why the current refresh writer must not merge an existing generation.
///
/// Legacy FSVI v1 carries only a display id, revision string, and dimension.
/// None proves vector-space compatibility, so even equal values must fail
/// closed. FSVI v2 carries the required identity, but this refresh path still
/// consumes raw embeddings and publishes through the legacy two-tier writer;
/// allowing it to merge would discard that identity on the replacement. A
/// future identity-bound writer can replace this refusal with exact typed
/// identity comparison and v2 republication.
fn incremental_merge_refusal(
    fast_index: &VectorIndex,
    quality_index: Option<&VectorIndex>,
) -> SearchError {
    let legacy_tier = if fast_index.identity_v2().is_none() {
        Some("fast")
    } else if quality_index.is_some_and(|index| index.identity_v2().is_none()) {
        Some("quality")
    } else {
        None
    };

    if let Some(tier) = legacy_tier {
        return SearchError::InvalidConfig {
            field: format!("refresh.{tier}_index_identity"),
            value: "identityless-fsvi-v1".to_owned(),
            reason: "refusing incremental vector merge because the existing generation has no complete immutable embedding identity; a full identity-bound rebuild is required"
                .to_owned(),
        };
    }

    SearchError::InvalidConfig {
        field: "refresh.index_publication".to_owned(),
        value: "identity-bound-republication-unavailable".to_owned(),
        reason: "refusing incremental vector merge because this refresh writer cannot preserve and republish FSVI v2 identity; a full identity-bound rebuild is required"
            .to_owned(),
    }
}

fn index_has_live_vectors(index: &VectorIndex) -> bool {
    index.wal_records().next().is_some()
        || (0..index.record_count()).any(|record_index| !index.is_deleted(record_index))
}

// ---------------------------------------------------------------------------
// Refresh worker
// ---------------------------------------------------------------------------

/// Background worker that drains the embedding queue and rebuilds the index.
///
/// # Architecture
///
/// ```text
/// EmbeddingQueue ──drain──> RefreshWorker ──embed──> TwoTierIndexBuilder
///                                                         │
///                                                    ┌────┘
///                                                    ▼
///                                              IndexCache.replace()
/// ```
///
/// The worker is the single writer for vector indices. It:
/// 1. Drains pending jobs from the [`EmbeddingQueue`]
/// 2. Batch-embeds via the fast-tier [`Embedder`] (and optionally quality-tier)
/// 3. Rebuilds the full `TwoTierIndex` from scratch
/// 4. Atomically replaces the cached index via [`IndexCache::replace`]
///
/// # Legacy incremental safety boundary
///
/// The current two-tier writer emits identityless FSVI v1 artifacts. It may
/// publish the initial generation, but a later non-empty cycle refuses to merge
/// that generation before draining, leaving queued jobs and retry counts
/// untouched. Operators must run a full identity-bound rebuild until this path
/// consumes bound embeddings and republishes complete FSVI v2 identity.
///
/// # Cancellation
///
/// The worker checks `cx.is_cancel_requested()` at each cycle boundary.
/// When cancelled, it finishes the current batch (no half-written index)
/// before returning.
pub struct RefreshWorker {
    config: RefreshWorkerConfig,
    queue: Arc<EmbeddingQueue>,
    fast_embedder: Arc<dyn Embedder>,
    quality_embedder: Option<Arc<dyn Embedder>>,
    cache: Arc<IndexCache>,
    metrics: Arc<RefreshMetrics>,
}

impl std::fmt::Debug for RefreshWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RefreshWorker")
            .field("config", &self.config)
            .field("fast_embedder", &self.fast_embedder.id())
            .finish_non_exhaustive()
    }
}

impl RefreshWorker {
    /// Create a new refresh worker.
    #[must_use]
    pub fn new(
        config: RefreshWorkerConfig,
        queue: Arc<EmbeddingQueue>,
        fast_embedder: Arc<dyn Embedder>,
        cache: Arc<IndexCache>,
    ) -> Self {
        Self {
            config,
            queue,
            fast_embedder,
            quality_embedder: None,
            cache,
            metrics: Arc::new(RefreshMetrics::default()),
        }
    }

    /// Set the quality-tier embedder for two-tier index building.
    #[must_use]
    pub fn with_quality_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.quality_embedder = Some(embedder);
        self
    }

    /// Shared reference to the metrics counters.
    #[must_use]
    pub const fn metrics(&self) -> &Arc<RefreshMetrics> {
        &self.metrics
    }

    /// Run the refresh loop.
    ///
    /// Polls the queue at `poll_interval`, embeds batches, and rebuilds the
    /// index. Returns `Ok(())` when the `Cx` is cancelled.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` on unrecoverable failures (e.g., index directory
    /// inaccessible). Transient embedding failures are logged and retried
    /// via the queue's retry mechanism.
    pub async fn run(&self, cx: &Cx) -> SearchResult<()> {
        info!(
            target: "frankensearch.refresh",
            poll_interval_ms = u64::try_from(self.config.poll_interval.as_millis()).unwrap_or(u64::MAX),
            max_docs = self.config.max_docs_per_cycle,
            index_dir = %self.config.index_dir.display(),
            "refresh worker started"
        );

        loop {
            // Cancel-aware sleep.
            asupersync::time::sleep(asupersync::time::wall_now(), self.config.poll_interval).await;

            if cx.is_cancel_requested() {
                info!(
                    target: "frankensearch.refresh",
                    "refresh worker shutting down (cancel requested)"
                );
                return Ok(());
            }

            // Run one refresh cycle. Transient errors are logged, not propagated.
            match self.run_cycle(cx).await {
                Ok(0) => {
                    // No work to do — continue polling.
                }
                Ok(n) => {
                    debug!(
                        target: "frankensearch.refresh",
                        docs = n,
                        "refresh cycle complete"
                    );
                }
                Err(e) => {
                    error!(
                        target: "frankensearch.refresh",
                        error = %e,
                        "refresh cycle failed"
                    );
                    // Continue polling — next cycle may succeed.
                }
            }

            self.metrics.cycles.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Run a single refresh cycle.
    ///
    /// Returns the number of documents successfully embedded, or an error
    /// if the index rebuild itself failed.
    ///
    /// # Errors
    ///
    /// Returns identity-admission and index creation/writing errors. Embedding
    /// failures for individual documents are handled via retry (requeue) and
    /// do not cause the cycle to fail.
    pub async fn run_cycle(&self, cx: &Cx) -> SearchResult<usize> {
        // Avoid opening the index on idle polls, but refuse any unsafe merge
        // before draining work. A permanent identity-admission failure must not
        // consume retry budget or eventually drop the queued documents.
        if self.queue.is_empty() {
            return Ok(0);
        }
        self.ensure_no_existing_generation_for_merge()?;

        // Drain at most `max_docs_per_cycle` jobs from the queue.
        let mut all_jobs = Vec::new();
        let batch_limit = self.config.max_docs_per_cycle;

        while all_jobs.len() < batch_limit {
            let remaining = batch_limit - all_jobs.len();
            let batch = self.queue.drain_batch_up_to(remaining);
            if batch.is_empty() {
                break;
            }
            all_jobs.extend(batch);
        }

        if all_jobs.is_empty() {
            return Ok(0);
        }

        let total_jobs = all_jobs.len();
        debug!(
            target: "frankensearch.refresh",
            jobs = total_jobs,
            "starting refresh cycle"
        );

        // Embed all documents.
        let embedded = self.embed_batch(cx, &all_jobs).await;

        if embedded.is_empty() {
            // All embeddings failed — nothing to index.
            warn!(
                target: "frankensearch.refresh",
                jobs = total_jobs,
                "all embeddings failed in cycle"
            );
            return Ok(0);
        }

        let embedded_count = embedded.len();

        // Rebuild the index.
        let rebuild_start = Instant::now();
        match self.rebuild_index(&embedded) {
            Ok(new_index) => {
                let rebuild_us =
                    u64::try_from(rebuild_start.elapsed().as_micros()).unwrap_or(u64::MAX);
                self.metrics
                    .rebuild_time_us
                    .fetch_add(rebuild_us, Ordering::Relaxed);
                self.metrics.index_rebuilds.fetch_add(1, Ordering::Relaxed);

                // Record all embedded hashes so the queue can skip unchanged docs.
                for record in &embedded {
                    self.queue
                        .record_embedded(&record.doc_id, &record.content_hash);
                }

                // Atomically swap the cached index.
                self.cache.replace(new_index);

                info!(
                    target: "frankensearch.refresh",
                    docs = embedded_count,
                    rebuild_ms = rebuild_us / 1000,
                    "index rebuilt and swapped"
                );

                Ok(embedded_count)
            }
            Err(e) => {
                self.metrics
                    .rebuild_failures
                    .fetch_add(1, Ordering::Relaxed);

                // Requeue all jobs so they aren't lost.
                let mut dropped_requeues = 0usize;
                for job in all_jobs {
                    if !self.requeue_job(job, "index_rebuild_failed") {
                        dropped_requeues = dropped_requeues.saturating_add(1);
                    }
                }
                if dropped_requeues > 0 {
                    self.metrics.docs_failed.fetch_add(
                        u64::try_from(dropped_requeues).unwrap_or(u64::MAX),
                        Ordering::Relaxed,
                    );
                }

                error!(
                    target: "frankensearch.refresh",
                    error = %e,
                    dropped_requeues,
                    "index rebuild failed, attempted to requeue jobs"
                );

                Err(e)
            }
        }
    }

    /// Embed a batch of jobs using the fast (and optionally quality) embedder.
    ///
    /// Failed embeddings are requeued for retry. Returns only the successfully
    /// embedded records.
    async fn embed_batch(&self, cx: &Cx, jobs: &[EmbeddingJob]) -> Vec<EmbeddedRecord> {
        let embed_start = Instant::now();

        // Collect texts for batch embedding.
        let texts: Vec<&str> = jobs.iter().map(|j| j.canonical_text.as_str()).collect();

        // Fast-tier embedding (required).
        let fast_embeddings = match self.fast_embedder.embed_batch(cx, &texts).await {
            Ok(embeddings) => embeddings,
            Err(e) => {
                warn!(
                    target: "frankensearch.refresh",
                    error = %e,
                    batch_size = jobs.len(),
                    "fast-tier batch embedding failed, requeueing all"
                );
                let mut dropped_requeues = 0usize;
                for job in jobs {
                    if !self.requeue_job(job.clone(), "fast_batch_embedding_failed") {
                        dropped_requeues = dropped_requeues.saturating_add(1);
                    }
                }
                if dropped_requeues > 0 {
                    warn!(
                        target: "frankensearch.refresh",
                        dropped_requeues,
                        "failed to requeue some jobs after fast-tier batch failure"
                    );
                }
                self.metrics.docs_failed.fetch_add(
                    u64::try_from(jobs.len()).unwrap_or(u64::MAX),
                    Ordering::Relaxed,
                );
                return Vec::new();
            }
        };

        // Quality-tier embedding (optional).
        let quality_embeddings = if let Some(ref quality) = self.quality_embedder {
            match quality.embed_batch(cx, &texts).await {
                Ok(embeddings) => Some(embeddings),
                Err(e) => {
                    warn!(
                        target: "frankensearch.refresh",
                        error = %e,
                        "quality-tier batch embedding failed, proceeding with fast only"
                    );
                    None
                }
            }
        } else {
            None
        };

        let embed_us = u64::try_from(embed_start.elapsed().as_micros()).unwrap_or(u64::MAX);
        self.metrics
            .embed_time_us
            .fetch_add(embed_us, Ordering::Relaxed);

        // Assemble records.
        let mut records = Vec::with_capacity(jobs.len());
        for (i, job) in jobs.iter().enumerate() {
            let Some(fast_embedding) = fast_embeddings.get(i).cloned() else {
                // Embedder returned fewer vectors than inputs — skip this doc
                // and requeue so it can be retried next cycle.
                warn!(
                    target: "frankensearch.refresh",
                    doc_id = %job.doc_id,
                    expected = jobs.len(),
                    got = fast_embeddings.len(),
                    "fast embedder returned fewer vectors than inputs, requeueing"
                );
                if !self.requeue_job(job.clone(), "fast_batch_missing_vector") {
                    warn!(
                        target: "frankensearch.refresh",
                        doc_id = %job.doc_id,
                        "failed to requeue job after fast embedder returned fewer vectors than expected"
                    );
                }
                self.metrics.docs_failed.fetch_add(1, Ordering::Relaxed);
                continue;
            };
            let quality_embedding = quality_embeddings.as_ref().and_then(|q| q.get(i).cloned());

            records.push(EmbeddedRecord {
                doc_id: job.doc_id.clone(),
                fast_embedding,
                quality_embedding,
                content_hash: job.content_hash.clone(),
            });
        }

        self.metrics.docs_embedded.fetch_add(
            u64::try_from(records.len()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.queue
            .metrics()
            .total_embed_time_us
            .fetch_add(embed_us, Ordering::Relaxed);

        records
    }

    fn requeue_job(&self, job: EmbeddingJob, reason: &'static str) -> bool {
        let doc_id = job.doc_id.clone();
        match self.queue.requeue(job) {
            JobOutcome::Retryable => true,
            JobOutcome::Failed => {
                warn!(
                    target: "frankensearch.refresh",
                    doc_id = %doc_id,
                    reason,
                    "failed to requeue job"
                );
                false
            }
            outcome => {
                warn!(
                    target: "frankensearch.refresh",
                    doc_id = %doc_id,
                    reason,
                    ?outcome,
                    "unexpected requeue outcome"
                );
                false
            }
        }
    }

    /// Refuse to merge any active generation until refresh publication is
    /// identity-bound end to end.
    fn ensure_no_existing_generation_for_merge(&self) -> SearchResult<()> {
        let fast_path = self.config.index_dir.join(VECTOR_INDEX_FAST_FILENAME);
        let fallback_path = self.config.index_dir.join(VECTOR_INDEX_FALLBACK_FILENAME);
        let existing_fast_path = if fast_path.exists() {
            Some(fast_path)
        } else if fallback_path.exists() {
            Some(fallback_path)
        } else {
            None
        };

        let Some(existing_fast_path) = existing_fast_path else {
            return Ok(());
        };

        let fast_index = VectorIndex::open(&existing_fast_path)?;
        let quality_path = self.config.index_dir.join(VECTOR_INDEX_QUALITY_FILENAME);
        let quality_index = if quality_path.exists() {
            Some(VectorIndex::open(&quality_path)?)
        } else {
            None
        };

        // Replacing an empty bootstrap artifact cannot mix vector spaces. The
        // cache requires such a seed before the first real refresh, so admit it
        // only when neither tier has a live main-slab or WAL-resident vector.
        let fast_is_empty = !index_has_live_vectors(&fast_index);
        let quality_is_empty = quality_index
            .as_ref()
            .is_none_or(|index| !index_has_live_vectors(index));
        if fast_is_empty && quality_is_empty {
            return Ok(());
        }

        Err(incremental_merge_refusal(
            &fast_index,
            quality_index.as_ref(),
        ))
    }

    /// Rebuild the `TwoTierIndex` from embedded records.
    fn rebuild_index(&self, records: &[EmbeddedRecord]) -> SearchResult<TwoTierIndex> {
        // Repeat the admission check after embedding to close the race where
        // another publisher installs a generation between the pre-drain check
        // and this rebuild. Only this race fallback consumes one retry.
        self.ensure_no_existing_generation_for_merge()?;

        let mut builder =
            TwoTierIndex::create(&self.config.index_dir, self.config.index_config.clone())?;

        builder.set_fast_embedder_id(self.fast_embedder.id());
        if let Some(ref quality) = self.quality_embedder {
            builder.set_quality_embedder_id(quality.id());
        }

        // Keep only the latest update per doc_id from this cycle.
        let mut latest_by_doc_id = HashMap::new();
        let mut consumed = vec![false; records.len()];
        for (idx, record) in records.iter().enumerate() {
            if let Some(previous) = latest_by_doc_id.insert(record.doc_id.as_str(), idx) {
                consumed[previous] = true;
            }
        }

        for (idx, record) in records.iter().enumerate() {
            if consumed[idx] {
                continue;
            }
            builder.add_record(
                &record.doc_id,
                &record.fast_embedding,
                record.quality_embedding.as_deref(),
            )?;
        }

        builder.finish()
    }

    /// Reference to the index directory.
    #[must_use]
    pub fn index_dir(&self) -> &Path {
        &self.config.index_dir
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    use frankensearch_core::canonicalize::DefaultCanonicalizer;
    use frankensearch_core::config::TwoTierConfig;
    use frankensearch_core::error::SearchError;
    use frankensearch_core::traits::{ModelCategory, SearchFuture};
    use frankensearch_index::{VECTOR_INDEX_FAST_FILENAME, VectorIndex};

    use super::*;
    use crate::cache::SentinelFileDetector;
    use crate::queue::{EmbeddingQueueConfig, EmbeddingRequest, JobOutcome};

    // -- Stub embedder for tests -----------------------------------------------

    struct StubEmbedder {
        id: &'static str,
        dimension: usize,
        space_offset: f32,
    }

    impl StubEmbedder {
        const fn new(id: &'static str, dimension: usize) -> Self {
            Self::in_space(id, dimension, 0.0)
        }

        const fn in_space(id: &'static str, dimension: usize, space_offset: f32) -> Self {
            Self {
                id,
                dimension,
                space_offset,
            }
        }
    }

    impl Embedder for StubEmbedder {
        fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            let dim = self.dimension;
            let seed = text.len() as f32 + self.space_offset;
            Box::pin(async move { Ok((0..dim).map(|i| (seed + i as f32).sin()).collect()) })
        }

        fn id(&self) -> &str {
            self.id
        }

        fn model_name(&self) -> &str {
            self.id
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn is_ready(&self) -> bool {
            true
        }

        fn is_semantic(&self) -> bool {
            false
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::HashEmbedder
        }
    }

    struct FailingEmbedder;

    impl Embedder for FailingEmbedder {
        fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async {
                Err(SearchError::EmbeddingFailed {
                    model: "failing-embedder".into(),
                    source: Box::new(std::io::Error::other("intentional failure")),
                })
            })
        }

        fn id(&self) -> &'static str {
            "failing-embedder"
        }

        fn model_name(&self) -> &'static str {
            "failing-embedder"
        }

        fn dimension(&self) -> usize {
            256
        }

        fn is_ready(&self) -> bool {
            true
        }

        fn is_semantic(&self) -> bool {
            false
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::HashEmbedder
        }
    }

    // -- Test helpers ----------------------------------------------------------

    fn make_queue(capacity: usize) -> Arc<EmbeddingQueue> {
        Arc::new(EmbeddingQueue::new(
            EmbeddingQueueConfig {
                capacity,
                batch_size: 100,
                max_retries: 3,
            },
            Box::new(DefaultCanonicalizer::default()),
        ))
    }

    fn submit(queue: &EmbeddingQueue, doc_id: &str, text: &str) {
        queue
            .submit(EmbeddingRequest {
                doc_id: doc_id.into(),
                text: text.to_owned(),
                metadata: None,
                submitted_at: Instant::now(),
            })
            .unwrap();
    }

    /// Create a temporary directory with a unique name.
    fn temp_index_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "frankensearch-refresh-test-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Seed an initial index on disk (required for `IndexCache::open`).
    fn seed_index(dir: &Path, dimension: usize) -> TwoTierIndex {
        let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
        let writer = VectorIndex::create(&fast_path, "stub-fast", dimension).unwrap();
        writer.finish().unwrap();
        TwoTierIndex::open(dir, TwoTierConfig::default()).unwrap()
    }

    fn make_cache(dir: &Path, dimension: usize) -> Arc<IndexCache> {
        seed_index(dir, dimension);
        let detector = Box::new(SentinelFileDetector::new());
        Arc::new(IndexCache::open(dir, TwoTierConfig::default(), detector).unwrap())
    }

    fn make_worker(
        queue: Arc<EmbeddingQueue>,
        dir: &Path,
        dimension: usize,
    ) -> (RefreshWorker, Arc<IndexCache>) {
        let cache = make_cache(dir, dimension);
        let config = RefreshWorkerConfig::new(dir).with_poll_interval(Duration::from_millis(10));
        let fast = Arc::new(StubEmbedder::new("stub-fast", dimension));
        let worker = RefreshWorker::new(config, queue, fast, cache.clone());
        (worker, cache)
    }

    // -- Tests -----------------------------------------------------------------

    #[test]
    fn config_defaults() {
        let config = RefreshWorkerConfig::new("/tmp/test-idx");
        assert_eq!(config.poll_interval, Duration::from_secs(1));
        assert_eq!(config.max_docs_per_cycle, 1000);
    }

    #[test]
    fn config_builder_methods() {
        let config = RefreshWorkerConfig::new("/tmp/test-idx")
            .with_poll_interval(Duration::from_millis(500))
            .with_max_docs_per_cycle(50);
        assert_eq!(config.poll_interval, Duration::from_millis(500));
        assert_eq!(config.max_docs_per_cycle, 50);
    }

    #[test]
    fn metrics_snapshot() {
        let metrics = RefreshMetrics::default();
        metrics.cycles.fetch_add(5, Ordering::Relaxed);
        metrics.docs_embedded.fetch_add(100, Ordering::Relaxed);
        let snap = metrics.snapshot();
        assert_eq!(snap.cycles, 5);
        assert_eq!(snap.docs_embedded, 100);
        assert_eq!(snap.docs_failed, 0);
    }

    #[test]
    fn run_cycle_empty_queue_returns_zero() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("empty");
            let queue = make_queue(10);
            let (worker, _cache) = make_worker(queue, &dir, 256);

            let count = worker.run_cycle(&cx).await.unwrap();
            assert_eq!(count, 0);
        });
    }

    #[test]
    fn run_cycle_embeds_and_rebuilds_index() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("rebuild");
            let queue = make_queue(100);
            submit(&queue, "doc-1", "Hello world");
            submit(&queue, "doc-2", "Goodbye world");

            let (worker, cache) = make_worker(queue.clone(), &dir, 256);

            let count = worker.run_cycle(&cx).await.unwrap();
            assert_eq!(count, 2);

            // Verify index was rebuilt.
            assert_eq!(worker.metrics().index_rebuilds.load(Ordering::Relaxed), 1);
            assert_eq!(worker.metrics().docs_embedded.load(Ordering::Relaxed), 2);

            // Queue should be empty after processing.
            assert!(queue.is_empty());

            // Cache should have the new index (2 docs, not the seed).
            let current = cache.current();
            assert_eq!(current.doc_count(), 2);
        });
    }

    #[test]
    fn run_cycle_records_content_hashes() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("hashes");
            let queue = make_queue(100);
            submit(&queue, "doc-1", "First document text");

            let (worker, _cache) = make_worker(queue.clone(), &dir, 256);
            worker.run_cycle(&cx).await.unwrap();

            // Submitting the same text again should be deduped.
            let outcome = queue
                .submit(EmbeddingRequest {
                    doc_id: "doc-1".into(),
                    text: "First document text".to_owned(),
                    metadata: None,
                    submitted_at: Instant::now(),
                })
                .unwrap();
            assert_eq!(outcome, JobOutcome::SkippedUnchanged);
        });
    }

    #[test]
    fn run_cycle_with_quality_embedder() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("quality");
            let queue = make_queue(100);
            submit(&queue, "doc-1", "Test document");

            let cache = make_cache(&dir, 256);
            let config =
                RefreshWorkerConfig::new(&dir).with_poll_interval(Duration::from_millis(10));
            let fast = Arc::new(StubEmbedder::new("stub-fast", 256));
            let quality = Arc::new(StubEmbedder::new("stub-quality", 384));
            let worker = RefreshWorker::new(config, queue.clone(), fast, cache.clone())
                .with_quality_embedder(quality);

            let count = worker.run_cycle(&cx).await.unwrap();
            assert_eq!(count, 1);

            // Index should have been rebuilt with quality tier.
            let index = cache.current();
            assert!(index.has_quality_index());
        });
    }

    #[test]
    fn pre_drain_refusal_does_not_charge_embedding_or_rebuild_metrics() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("multi");
            let queue = make_queue(100);
            let (worker, _cache) = make_worker(queue.clone(), &dir, 256);

            // Cycle 1: 2 docs.
            submit(&queue, "doc-1", "First");
            submit(&queue, "doc-2", "Second");
            worker.run_cycle(&cx).await.unwrap();

            // Cycle 2 refuses before draining or embedding the pending job.
            submit(&queue, "doc-3", "Third");
            worker
                .run_cycle(&cx)
                .await
                .expect_err("legacy incremental merge must fail closed");

            let snap = worker.metrics().snapshot();
            assert_eq!(snap.docs_embedded, 2);
            assert_eq!(snap.index_rebuilds, 1);
            assert_eq!(snap.rebuild_failures, 0);
            assert_eq!(queue.pending_count(), 1);
        });
    }

    #[test]
    fn run_cycle_respects_max_docs_per_cycle() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("limit");
            let queue = make_queue(100);
            for i in 0..5 {
                submit(&queue, &format!("doc-{i}"), "Payload");
            }

            let cache = make_cache(&dir, 256);
            let config = RefreshWorkerConfig::new(&dir)
                .with_poll_interval(Duration::from_millis(10))
                .with_max_docs_per_cycle(3);
            let fast = Arc::new(StubEmbedder::new("stub-fast", 256));
            let worker = RefreshWorker::new(config, queue.clone(), fast, cache.clone());

            let count = worker.run_cycle(&cx).await.unwrap();
            assert_eq!(count, 3);
            assert_eq!(queue.pending_count(), 2);
        });
    }

    #[test]
    fn identityless_incremental_refusal_preserves_active_generation() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("preserve-existing");
            let queue = make_queue(100);
            let (worker, cache) = make_worker(queue.clone(), &dir, 256);

            submit(&queue, "doc-1", "First");
            submit(&queue, "doc-2", "Second");
            worker.run_cycle(&cx).await.expect("first cycle");

            let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
            let original_bytes = std::fs::read(&fast_path).expect("read active generation");

            submit(&queue, "doc-3", "Third");
            let error = worker
                .run_cycle(&cx)
                .await
                .expect_err("identityless incremental merge must fail closed");
            assert!(matches!(
                error,
                SearchError::InvalidConfig { ref field, ref value, .. }
                    if field == "refresh.fast_index_identity"
                        && value == "identityless-fsvi-v1"
            ));
            assert_eq!(queue.pending_count(), 1, "failed job must be requeued");
            assert_eq!(
                std::fs::read(&fast_path).expect("reread active generation"),
                original_bytes,
                "refusal must not mutate or replace the active generation"
            );

            let current = cache.current();
            assert_eq!(
                current.doc_count(),
                2,
                "cache must continue serving the prior active generation"
            );
            let doc_ids: Vec<String> = current.iter_doc_ids().filter_map(Result::ok).collect();
            assert!(doc_ids.iter().any(|id| id == "doc-1"));
            assert!(doc_ids.iter().any(|id| id == "doc-2"));
            assert!(!doc_ids.iter().any(|id| id == "doc-3"));
        });
    }

    #[test]
    fn same_id_same_dimension_changed_space_cannot_bypass_identityless_refusal() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("fast-id-change");
            let queue = make_queue(100);
            let cache = make_cache(&dir, 256);
            let config = RefreshWorkerConfig::new(&dir);
            let original_embedder = Arc::new(StubEmbedder::in_space("stub-fast", 256, 0.0));
            let changed_embedder = Arc::new(StubEmbedder::in_space("stub-fast", 256, 100.0));
            assert_eq!(original_embedder.id(), changed_embedder.id());
            assert_eq!(original_embedder.dimension(), changed_embedder.dimension());
            let original_probe = original_embedder
                .embed(&cx, "space probe")
                .await
                .expect("original probe");
            let changed_probe = changed_embedder
                .embed(&cx, "space probe")
                .await
                .expect("changed probe");
            assert!(
                original_probe
                    .iter()
                    .zip(&changed_probe)
                    .any(|(original, changed)| original.to_bits() != changed.to_bits()),
                "fixture must represent a real same-id, same-dimension space change"
            );
            let original = RefreshWorker::new(
                config.clone(),
                queue.clone(),
                original_embedder,
                cache.clone(),
            );

            submit(&queue, "doc-old", "old generation");
            original.run_cycle(&cx).await.expect("initial generation");

            let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
            let original_bytes = std::fs::read(&fast_path).expect("read original fast index");
            let changed = RefreshWorker::new(config, queue.clone(), changed_embedder, cache);

            submit(&queue, "doc-new", "new generation");
            let error = changed
                .run_cycle(&cx)
                .await
                .expect_err("identityless generation must fail closed before vector merge");

            assert!(matches!(
                error,
                SearchError::InvalidConfig { ref field, ref value, .. }
                    if field == "refresh.fast_index_identity"
                        && value == "identityless-fsvi-v1"
            ));
            assert_eq!(queue.pending_count(), 1, "failed job must be requeued");
            assert_eq!(
                std::fs::read(&fast_path).expect("reread fast index"),
                original_bytes,
                "failed refresh must not re-stamp the existing generation"
            );
            let reopened = VectorIndex::open(&fast_path).expect("reopen original fast index");
            assert_eq!(reopened.embedder_id(), "stub-fast");
            let live_ids = reopened.live_doc_ids().expect("live ids");
            assert_eq!(live_ids.len(), 1);
            assert!(live_ids.contains("doc-old"));
        });
    }

    #[test]
    fn unchanged_ids_cannot_exhaust_retry_budget_while_legacy_generation_is_blocked() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("unchanged-tier-ids");
            let queue = make_queue(100);
            let cache = make_cache(&dir, 256);
            let worker = RefreshWorker::new(
                RefreshWorkerConfig::new(&dir),
                queue.clone(),
                Arc::new(StubEmbedder::new("stub-fast", 256)),
                cache.clone(),
            )
            .with_quality_embedder(Arc::new(StubEmbedder::new("stub-quality", 384)));

            submit(&queue, "doc-old", "old generation");
            worker.run_cycle(&cx).await.expect("initial generation");

            let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
            let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
            let original_fast_bytes = std::fs::read(&fast_path).expect("read original fast index");
            let original_quality_bytes =
                std::fs::read(&quality_path).expect("read original quality index");

            submit(&queue, "doc-new", "new generation");
            let refusal_cycles = queue.config().max_retries + 2;
            for _ in 0..refusal_cycles {
                let error = worker
                    .run_cycle(&cx)
                    .await
                    .expect_err("matching display ids must not admit a legacy merge");
                assert!(matches!(
                    error,
                    SearchError::InvalidConfig { ref field, ref value, .. }
                        if field == "refresh.fast_index_identity"
                            && value == "identityless-fsvi-v1"
                ));
            }
            assert_eq!(
                queue.pending_count(),
                1,
                "permanent refusal must leave the job queued beyond max_retries"
            );
            assert_eq!(
                std::fs::read(&fast_path).expect("reread fast index"),
                original_fast_bytes,
                "refusal must not replace the fast generation"
            );
            assert_eq!(
                std::fs::read(&quality_path).expect("reread quality index"),
                original_quality_bytes,
                "refusal must not replace the quality generation"
            );
            assert_eq!(cache.current().doc_count(), 1);
            let fast = VectorIndex::open(&fast_path).expect("open fast index");
            let quality = VectorIndex::open(&quality_path).expect("open quality index");
            assert_eq!(fast.embedder_id(), "stub-fast");
            assert_eq!(quality.embedder_id(), "stub-quality");
            assert_eq!(fast.live_doc_ids().expect("fast live ids").len(), 1);
            assert_eq!(quality.live_doc_ids().expect("quality live ids").len(), 1);

            let pending = queue.drain_batch();
            assert_eq!(pending.len(), 1);
            assert_eq!(pending[0].doc_id, "doc-new");
            assert_eq!(
                pending[0].retry_count, 0,
                "pre-drain refusal must not consume retry budget"
            );
        });
    }

    #[test]
    fn failed_embedding_requeues_jobs() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("fail");
            let queue = make_queue(100);
            submit(&queue, "doc-1", "Will fail");

            let cache = make_cache(&dir, 256);
            let config = RefreshWorkerConfig::new(&dir);
            let failing = Arc::new(FailingEmbedder);
            let worker = RefreshWorker::new(config, queue.clone(), failing, cache);

            let count = worker.run_cycle(&cx).await.unwrap();
            assert_eq!(count, 0);

            // Job should have been requeued.
            assert_eq!(queue.pending_count(), 1);
            assert_eq!(worker.metrics().docs_failed.load(Ordering::Relaxed), 1);
        });
    }

    #[test]
    fn index_dir_accessor() {
        let dir = temp_index_dir("accessor");
        let queue = make_queue(10);
        let (worker, _cache) = make_worker(queue, &dir, 256);
        assert_eq!(worker.index_dir(), dir.as_path());
    }

    #[test]
    fn debug_format() {
        let dir = temp_index_dir("debug");
        let queue = make_queue(10);
        let (worker, _cache) = make_worker(queue, &dir, 256);
        let debug = format!("{worker:?}");
        assert!(debug.contains("RefreshWorker"));
        assert!(debug.contains("stub-fast"));
    }

    // ─── bd-qkop tests begin ───

    #[test]
    fn config_with_index_config() {
        let custom = TwoTierConfig::default();
        let config = RefreshWorkerConfig::new("/tmp/idx").with_index_config(custom.clone());
        assert!((config.index_config.rrf_k - custom.rrf_k).abs() < f64::EPSILON);
    }

    #[test]
    fn config_clone() {
        let config = RefreshWorkerConfig::new("/tmp/idx")
            .with_poll_interval(Duration::from_millis(42))
            .with_max_docs_per_cycle(7);
        #[allow(clippy::redundant_clone)]
        let cloned = config.clone();
        assert_eq!(cloned.poll_interval, Duration::from_millis(42));
        assert_eq!(cloned.max_docs_per_cycle, 7);
        assert_eq!(cloned.index_dir, PathBuf::from("/tmp/idx"));
    }

    #[test]
    fn config_debug() {
        let config = RefreshWorkerConfig::new("/tmp/idx");
        let debug = format!("{config:?}");
        assert!(debug.contains("RefreshWorkerConfig"));
        assert!(debug.contains("poll_interval"));
    }

    #[test]
    fn metrics_default_all_zeros() {
        let m = RefreshMetrics::default();
        assert_eq!(m.cycles.load(Ordering::Relaxed), 0);
        assert_eq!(m.docs_embedded.load(Ordering::Relaxed), 0);
        assert_eq!(m.docs_failed.load(Ordering::Relaxed), 0);
        assert_eq!(m.index_rebuilds.load(Ordering::Relaxed), 0);
        assert_eq!(m.rebuild_failures.load(Ordering::Relaxed), 0);
        assert_eq!(m.embed_time_us.load(Ordering::Relaxed), 0);
        assert_eq!(m.rebuild_time_us.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn metrics_snapshot_equality() {
        let a = RefreshMetricsSnapshot {
            cycles: 1,
            docs_embedded: 2,
            docs_failed: 0,
            index_rebuilds: 1,
            rebuild_failures: 0,
            embed_time_us: 100,
            rebuild_time_us: 50,
        };
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn metrics_snapshot_inequality() {
        let a = RefreshMetricsSnapshot {
            cycles: 1,
            docs_embedded: 2,
            docs_failed: 0,
            index_rebuilds: 1,
            rebuild_failures: 0,
            embed_time_us: 100,
            rebuild_time_us: 50,
        };
        let b = RefreshMetricsSnapshot { cycles: 2, ..a };
        assert_ne!(a, b);
    }

    #[test]
    fn metrics_snapshot_clone_copy() {
        let a = RefreshMetricsSnapshot {
            cycles: 5,
            docs_embedded: 10,
            docs_failed: 1,
            index_rebuilds: 3,
            rebuild_failures: 0,
            embed_time_us: 200,
            rebuild_time_us: 100,
        };
        #[allow(clippy::clone_on_copy)]
        let cloned = a.clone();
        let copied: RefreshMetricsSnapshot = a; // Copy
        assert_eq!(cloned, a);
        assert_eq!(copied, a);
    }

    #[test]
    fn metrics_snapshot_debug() {
        let snap = RefreshMetricsSnapshot {
            cycles: 0,
            docs_embedded: 0,
            docs_failed: 0,
            index_rebuilds: 0,
            rebuild_failures: 0,
            embed_time_us: 0,
            rebuild_time_us: 0,
        };
        let debug = format!("{snap:?}");
        assert!(debug.contains("RefreshMetricsSnapshot"));
        assert!(debug.contains("cycles"));
    }

    #[test]
    fn metrics_individual_increments() {
        let m = RefreshMetrics::default();
        m.cycles.fetch_add(3, Ordering::Relaxed);
        m.docs_failed.fetch_add(2, Ordering::Relaxed);
        m.rebuild_failures.fetch_add(1, Ordering::Relaxed);
        m.embed_time_us.fetch_add(500, Ordering::Relaxed);
        m.rebuild_time_us.fetch_add(300, Ordering::Relaxed);
        let snap = m.snapshot();
        assert_eq!(snap.cycles, 3);
        assert_eq!(snap.docs_failed, 2);
        assert_eq!(snap.rebuild_failures, 1);
        assert_eq!(snap.embed_time_us, 500);
        assert_eq!(snap.rebuild_time_us, 300);
    }

    #[test]
    fn worker_metrics_accessor_returns_shared_arc() {
        let dir = temp_index_dir("metrics-arc");
        let queue = make_queue(10);
        let (worker, _cache) = make_worker(queue, &dir, 256);
        let metrics = worker.metrics();
        metrics.cycles.fetch_add(1, Ordering::Relaxed);
        // Same Arc should reflect the update.
        assert_eq!(worker.metrics().cycles.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn worker_with_quality_embedder_sets_embedder() {
        let dir = temp_index_dir("quality-set");
        let queue = make_queue(10);
        let cache = make_cache(&dir, 256);
        let config = RefreshWorkerConfig::new(&dir);
        let fast = Arc::new(StubEmbedder::new("stub-fast", 256));
        let quality = Arc::new(StubEmbedder::new("stub-quality", 384));
        let worker = RefreshWorker::new(config, queue, fast, cache).with_quality_embedder(quality);
        // Should be set (verified indirectly by debug containing only fast id)
        let debug = format!("{worker:?}");
        assert!(debug.contains("stub-fast"));
    }

    #[test]
    fn duplicate_doc_id_keeps_latest_in_cycle() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("dedup-cycle");
            let queue = make_queue(100);
            // Submit same doc_id with different texts.
            submit(&queue, "doc-dup", "First version");
            submit(&queue, "doc-dup", "Second version");

            let (worker, cache) = make_worker(queue.clone(), &dir, 256);
            let count = worker.run_cycle(&cx).await.unwrap();
            // Both submitted but doc count should be 1 (deduped).
            assert!(count >= 1);
            let current = cache.current();
            // Only one doc-dup in the index.
            let doc_ids: Vec<String> = current.iter_doc_ids().filter_map(Result::ok).collect();
            let dup_count = doc_ids.iter().filter(|id| *id == "doc-dup").count();
            assert_eq!(dup_count, 1);
        });
    }

    #[test]
    fn quality_embedder_failure_proceeds_with_fast_only() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("quality-fail");
            let queue = make_queue(100);
            submit(&queue, "doc-1", "Test document");

            let cache = make_cache(&dir, 256);
            let config =
                RefreshWorkerConfig::new(&dir).with_poll_interval(Duration::from_millis(10));
            let fast = Arc::new(StubEmbedder::new("stub-fast", 256));
            let failing_quality: Arc<dyn Embedder> = Arc::new(FailingEmbedder);
            let worker = RefreshWorker::new(config, queue.clone(), fast, cache.clone())
                .with_quality_embedder(failing_quality);

            let count = worker.run_cycle(&cx).await.unwrap();
            assert_eq!(count, 1);

            // Index should still have the doc (fast-only).
            let current = cache.current();
            assert_eq!(current.doc_count(), 1);
            // But no quality index since quality embedder failed.
            assert!(!current.has_quality_index());
        });
    }

    #[test]
    fn run_cycle_empty_after_prior_cycle_returns_zero() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("empty-after");
            let queue = make_queue(100);
            submit(&queue, "doc-1", "First");
            let (worker, _cache) = make_worker(queue.clone(), &dir, 256);

            worker.run_cycle(&cx).await.unwrap();
            // Second cycle with empty queue.
            let count = worker.run_cycle(&cx).await.unwrap();
            assert_eq!(count, 0);
        });
    }

    // ─── bd-qkop tests end ───

    // ─── bd-wt20 tests begin ──────────────────────────────────────────

    #[test]
    fn refresh_metrics_debug_format() {
        let m = RefreshMetrics::default();
        m.cycles.fetch_add(7, Ordering::Relaxed);
        m.docs_embedded.fetch_add(42, Ordering::Relaxed);
        let debug = format!("{m:?}");
        assert!(debug.contains("RefreshMetrics"));
        assert!(debug.contains("cycles"));
        assert!(debug.contains("docs_embedded"));
    }

    #[test]
    fn config_index_dir_from_new() {
        let config = RefreshWorkerConfig::new("/data/my-index");
        assert_eq!(config.index_dir, PathBuf::from("/data/my-index"));
    }

    #[test]
    fn config_index_config_override() {
        let custom = TwoTierConfig {
            rrf_k: 42.0,
            ..TwoTierConfig::default()
        };
        let config = RefreshWorkerConfig::new("/tmp/idx").with_index_config(custom);
        assert!((config.index_config.rrf_k - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_concurrent_increments() {
        let m = Arc::new(RefreshMetrics::default());
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let metrics = m.clone();
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        metrics.docs_embedded.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(m.docs_embedded.load(Ordering::Relaxed), 400);
    }

    #[test]
    fn identityless_incremental_refusal_does_not_overwrite_existing_doc() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("update-doc");
            let queue = make_queue(100);
            let (worker, cache) = make_worker(queue.clone(), &dir, 256);

            // Cycle 1: index doc-1 with first text.
            submit(&queue, "doc-1", "First version of the document");
            worker.run_cycle(&cx).await.unwrap();
            assert_eq!(cache.current().doc_count(), 1);

            // Cycle 2: submit doc-1 again with different text (hash changed).
            // Force a new submission by changing text.
            queue
                .submit(EmbeddingRequest {
                    doc_id: "doc-1".into(),
                    text: "Second version completely different".to_owned(),
                    metadata: None,
                    submitted_at: Instant::now(),
                })
                .unwrap();
            worker
                .run_cycle(&cx)
                .await
                .expect_err("legacy update must require an identity-bound rebuild");

            // The active generation remains available, and the update remains
            // queued for a later full identity-bound rebuild.
            let current = cache.current();
            assert_eq!(current.doc_count(), 1);
            let doc_ids: Vec<String> = current.iter_doc_ids().filter_map(Result::ok).collect();
            assert_eq!(doc_ids.len(), 1);
            assert_eq!(doc_ids[0], "doc-1");
            assert_eq!(queue.pending_count(), 1);
        });
    }

    /// Fleet-review regression pin: refusing an identityless generation must
    /// neither resurrect soft-deleted documents nor drop WAL-resident appends.
    #[test]
    fn identityless_rebuild_refusal_preserves_tombstones_and_wal_residents() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("tombstone-wal-rebuild");
            let queue = make_queue(100);
            let cache = make_cache(&dir, 256);
            let config =
                RefreshWorkerConfig::new(&dir).with_poll_interval(Duration::from_millis(10));
            let fast = Arc::new(StubEmbedder::new("stub-fast", 256));
            let worker = RefreshWorker::new(config, queue.clone(), fast.clone(), cache.clone());

            submit(&queue, "doc-keep", "kept document");
            submit(&queue, "doc-delete", "doomed document");
            worker.run_cycle(&cx).await.unwrap();
            assert_eq!(cache.current().doc_count(), 2);

            // Out-of-band mutations through the public index API, exactly
            // as an application deleting/appending between cycles does.
            let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
            {
                let mut index =
                    frankensearch_index::VectorIndex::open(&fast_path).expect("open fast tier");
                assert!(index.soft_delete("doc-delete").expect("soft delete"));
                let dim = index.dimension();
                index
                    .append("doc-wal", &vec![0.25_f32; dim])
                    .expect("wal append");
            }

            submit(&queue, "doc-new", "new document");
            worker
                .run_cycle(&cx)
                .await
                .expect_err("legacy WAL-bearing generation must not be republished");

            let preserved =
                frankensearch_index::VectorIndex::open(&fast_path).expect("reopen fast tier");
            let live = preserved.live_doc_ids().expect("live ids");
            assert!(
                !live.contains("doc-delete"),
                "refusal must not resurrect the soft-deleted document: {live:?}"
            );
            assert!(
                live.contains("doc-wal"),
                "refusal must preserve the WAL-resident append: {live:?}"
            );
            assert!(live.contains("doc-keep"), "{live:?}");
            assert!(!live.contains("doc-new"), "{live:?}");
            assert_eq!(queue.pending_count(), 1);
        });
    }

    #[test]
    fn identityless_refusal_preserves_prior_quality_tier() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("quality-preserve");
            let queue = make_queue(100);

            // First cycle with quality embedder.
            let cache = make_cache(&dir, 256);
            let config =
                RefreshWorkerConfig::new(&dir).with_poll_interval(Duration::from_millis(10));
            let fast = Arc::new(StubEmbedder::new("stub-fast", 256));
            let quality = Arc::new(StubEmbedder::new("stub-quality", 384));
            let worker_with_quality =
                RefreshWorker::new(config, queue.clone(), fast.clone(), cache.clone())
                    .with_quality_embedder(quality);

            submit(&queue, "doc-1", "Test document");
            worker_with_quality.run_cycle(&cx).await.unwrap();
            assert!(cache.current().has_quality_index());

            // Second cycle without quality embedder (fast-only worker).
            let config2 =
                RefreshWorkerConfig::new(&dir).with_poll_interval(Duration::from_millis(10));
            let worker_fast_only = RefreshWorker::new(config2, queue.clone(), fast, cache.clone());

            submit(&queue, "doc-2", "Another document");
            worker_fast_only
                .run_cycle(&cx)
                .await
                .expect_err("legacy two-tier generation must fail closed");

            // The prior quality generation remains active and the new work is
            // requeued for a full identity-bound rebuild.
            let current = cache.current();
            assert_eq!(current.doc_count(), 1);
            assert!(current.has_quality_index());
            assert_eq!(queue.pending_count(), 1);
        });
    }

    #[test]
    fn run_cycle_metrics_timing_nonzero() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("timing");
            let queue = make_queue(100);
            submit(&queue, "doc-1", "Timing test");

            let (worker, _cache) = make_worker(queue, &dir, 256);
            worker.run_cycle(&cx).await.unwrap();

            let snap = worker.metrics().snapshot();
            // Embed and rebuild should take some nonzero time.
            assert!(snap.embed_time_us > 0 || snap.rebuild_time_us > 0);
        });
    }

    #[test]
    fn worker_new_without_quality_debug() {
        let dir = temp_index_dir("no-quality");
        let queue = make_queue(10);
        let (worker, _cache) = make_worker(queue, &dir, 256);
        // Debug uses `finish_non_exhaustive` and includes fast embedder id.
        let debug = format!("{worker:?}");
        assert!(debug.contains("RefreshWorker"));
        assert!(debug.contains("stub-fast"));
    }

    // ─── bd-wt20 tests end ────────────────────────────────────────────
}
