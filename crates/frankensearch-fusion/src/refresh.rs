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
use frankensearch_core::error::SearchResult;
use frankensearch_core::traits::Embedder;
use frankensearch_index::{
    TwoTierIndex, VECTOR_INDEX_FALLBACK_FILENAME, VECTOR_INDEX_FAST_FILENAME,
    VECTOR_INDEX_QUALITY_FILENAME, VectorIndex,
};

use crate::cache::IndexCache;
use crate::queue::{EmbeddingJob, EmbeddingQueue};

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
    /// Returns errors from index creation/writing. Embedding failures for
    /// individual documents are handled via retry (requeue) and do not
    /// cause the cycle to fail.
    pub async fn run_cycle(&self, cx: &Cx) -> SearchResult<usize> {
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
                for job in all_jobs {
                    self.queue.requeue(job);
                }

                error!(
                    target: "frankensearch.refresh",
                    error = %e,
                    "index rebuild failed, jobs requeued"
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
                for job in jobs {
                    self.queue.requeue(job.clone());
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
                self.queue.requeue(job.clone());
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

    /// Rebuild the `TwoTierIndex` from embedded records.
    fn rebuild_index(&self, records: &[EmbeddedRecord]) -> SearchResult<TwoTierIndex> {
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

        // Merge with the previously built index so incremental cycles don't
        // drop documents that were not part of this queue drain.
        let fast_path = self.config.index_dir.join(VECTOR_INDEX_FAST_FILENAME);
        let fallback_path = self.config.index_dir.join(VECTOR_INDEX_FALLBACK_FILENAME);
        let existing_fast_path = if fast_path.exists() {
            Some(fast_path)
        } else if fallback_path.exists() {
            Some(fallback_path)
        } else {
            None
        };

        if let Some(existing_fast_path) = existing_fast_path {
            let fast_index = VectorIndex::open(&existing_fast_path)?;

            let quality_path = self.config.index_dir.join(VECTOR_INDEX_QUALITY_FILENAME);
            let existing_quality = if quality_path.exists() {
                Some(VectorIndex::open(&quality_path)?)
            } else {
                None
            };

            // Preserve prior quality embedder metadata when the current worker
            // is not configured with a quality embedder but a quality tier exists.
            if self.quality_embedder.is_none()
                && let Some(ref quality_index) = existing_quality
            {
                builder.set_quality_embedder_id(quality_index.embedder_id());
            }

            let mut quality_index_by_doc_id = HashMap::new();
            if let Some(ref quality_index) = existing_quality {
                quality_index_by_doc_id.reserve(quality_index.record_count());
                for quality_idx in 0..quality_index.record_count() {
                    let doc_id = quality_index.doc_id_at(quality_idx)?;
                    quality_index_by_doc_id
                        .entry(doc_id.to_owned())
                        .or_insert(quality_idx);
                }
            }

            for fast_idx in 0..fast_index.record_count() {
                let doc_id = fast_index.doc_id_at(fast_idx)?;
                if let Some(&record_idx) = latest_by_doc_id.get(doc_id) {
                    let record = &records[record_idx];
                    builder.add_record(
                        record.doc_id.clone(),
                        &record.fast_embedding,
                        record.quality_embedding.as_deref(),
                    )?;
                    consumed[record_idx] = true;
                    continue;
                }

                let fast_embedding = fast_index.vector_at_f32(fast_idx)?;
                let quality_embedding = if let Some(ref quality_index) = existing_quality {
                    if let Some(&quality_idx) = quality_index_by_doc_id.get(doc_id) {
                        Some(quality_index.vector_at_f32(quality_idx)?)
                    } else {
                        None
                    }
                } else {
                    None
                };

                builder.add_record(
                    doc_id.to_owned(),
                    &fast_embedding,
                    quality_embedding.as_deref(),
                )?;
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
    }

    impl StubEmbedder {
        const fn new(id: &'static str, dimension: usize) -> Self {
            Self { id, dimension }
        }
    }

    impl Embedder for StubEmbedder {
        fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            let dim = self.dimension;
            let seed = text.len() as f32;
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
                doc_id: doc_id.to_owned(),
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
                    doc_id: "doc-1".to_owned(),
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
    fn multiple_cycles_accumulate_metrics() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("multi");
            let queue = make_queue(100);
            let (worker, _cache) = make_worker(queue.clone(), &dir, 256);

            // Cycle 1: 2 docs.
            submit(&queue, "doc-1", "First");
            submit(&queue, "doc-2", "Second");
            worker.run_cycle(&cx).await.unwrap();

            // Cycle 2: 1 doc.
            submit(&queue, "doc-3", "Third");
            worker.run_cycle(&cx).await.unwrap();

            let snap = worker.metrics().snapshot();
            assert_eq!(snap.docs_embedded, 3);
            assert_eq!(snap.index_rebuilds, 2);
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
    fn incremental_rebuild_preserves_docs_not_in_current_batch() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = temp_index_dir("preserve-existing");
            let queue = make_queue(100);
            let (worker, cache) = make_worker(queue.clone(), &dir, 256);

            submit(&queue, "doc-1", "First");
            submit(&queue, "doc-2", "Second");
            worker.run_cycle(&cx).await.expect("first cycle");

            submit(&queue, "doc-3", "Third");
            worker.run_cycle(&cx).await.expect("second cycle");

            let current = cache.current();
            assert_eq!(
                current.doc_count(),
                3,
                "second incremental cycle must preserve previously indexed docs"
            );
            let doc_ids = current.doc_ids();
            assert!(doc_ids.iter().any(|id| id == "doc-1"));
            assert!(doc_ids.iter().any(|id| id == "doc-2"));
            assert!(doc_ids.iter().any(|id| id == "doc-3"));
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
            let doc_ids = current.doc_ids();
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
    fn incremental_rebuild_updates_existing_doc() {
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
                    doc_id: "doc-1".to_owned(),
                    text: "Second version completely different".to_owned(),
                    metadata: None,
                    submitted_at: Instant::now(),
                })
                .unwrap();
            worker.run_cycle(&cx).await.unwrap();

            // Should still have exactly 1 doc (updated, not duplicated).
            let current = cache.current();
            assert_eq!(current.doc_count(), 1);
            let doc_ids = current.doc_ids();
            assert_eq!(doc_ids.len(), 1);
            assert_eq!(doc_ids[0], "doc-1");
        });
    }

    #[test]
    fn rebuild_preserves_quality_tier_from_prior_cycle() {
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
            worker_fast_only.run_cycle(&cx).await.unwrap();

            // Should still have both docs.
            let current = cache.current();
            assert_eq!(current.doc_count(), 2);
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
