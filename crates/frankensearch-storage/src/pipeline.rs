use std::io::{self, Read};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use asupersync::Cx;
use frankensearch_core::{Canonicalizer, Embedder, SearchError, SearchResult};
use fsqlite::Connection;
use fsqlite_types::value::SqliteValue;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Storage;
use crate::connection::map_storage_error;
use crate::content_hash::{ContentHasher, record_content_hash};
use crate::document::{DocumentRecord, EmbeddingStatus, upsert_document};
use crate::job_queue::{
    EnqueueOutcome, EnqueueRequest, PersistentJobQueue, enqueue_inner, fetch_queue_depth,
};
use crate::schema::row_i64;

const PIPELINE_SUBSYSTEM: &str = "storage_pipeline";
const CORRELATION_METADATA_KEY: &str = "correlation_id";
const HASH_EMBEDDER_PREFIX: &str = "fnv1a-";
const JL_EMBEDDER_PREFIX: &str = "jl-";
const LEGACY_HASH_EMBEDDER_ID: &str = "hash/fnv1a";
const MAX_CONTENT_PREVIEW_CHARS: usize = 400;
const MAX_SOURCE_FILE_BYTES: usize = 2 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestRequest {
    pub doc_id: String,
    pub text: String,
    pub source_path: Option<String>,
    pub metadata: Option<Value>,
    pub correlation_id: Option<String>,
    pub enqueue_quality: bool,
}

impl IngestRequest {
    #[must_use]
    pub fn new(doc_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            doc_id: doc_id.into(),
            text: text.into(),
            source_path: None,
            metadata: None,
            correlation_id: None,
            enqueue_quality: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IngestAction {
    New,
    Updated,
    Unchanged,
    Skipped { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestResult {
    pub doc_id: String,
    pub action: IngestAction,
    pub fast_job_enqueued: bool,
    pub quality_job_enqueued: bool,
    pub correlation_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct BatchIngestResult {
    pub requested: usize,
    pub inserted: usize,
    pub updated: usize,
    pub unchanged: usize,
    pub skipped: usize,
    pub fast_jobs_enqueued: usize,
    pub quality_jobs_enqueued: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub fast_priority: i32,
    pub quality_priority: i32,
    pub process_batch_size: usize,
    pub worker_idle_sleep_ms: u64,
    pub worker_max_idle_cycles: Option<usize>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            fast_priority: 1,
            quality_priority: 0,
            process_batch_size: 32,
            worker_idle_sleep_ms: 25,
            worker_max_idle_cycles: Some(1),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct BatchProcessResult {
    pub jobs_claimed: usize,
    pub jobs_completed: usize,
    pub jobs_failed: usize,
    pub jobs_skipped: usize,
    pub embed_time: Duration,
    pub total_time: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct WorkerReport {
    pub reclaimed_on_startup: usize,
    pub batches_processed: usize,
    pub jobs_completed: usize,
    pub jobs_failed: usize,
    pub jobs_skipped: usize,
    pub idle_cycles: usize,
}

#[derive(Debug, Default)]
pub struct PipelineMetrics {
    pub total_ingest_calls: AtomicU64,
    pub total_ingest_inserted: AtomicU64,
    pub total_ingest_updated: AtomicU64,
    pub total_ingest_unchanged: AtomicU64,
    pub total_ingest_skipped: AtomicU64,
    pub total_jobs_claimed: AtomicU64,
    pub total_jobs_completed: AtomicU64,
    pub total_jobs_failed: AtomicU64,
    pub total_jobs_skipped: AtomicU64,
    pub total_embed_time_us: AtomicU64,
    pub total_reclaimed: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PipelineMetricsSnapshot {
    pub total_ingest_calls: u64,
    pub total_ingest_inserted: u64,
    pub total_ingest_updated: u64,
    pub total_ingest_unchanged: u64,
    pub total_ingest_skipped: u64,
    pub total_jobs_claimed: u64,
    pub total_jobs_completed: u64,
    pub total_jobs_failed: u64,
    pub total_jobs_skipped: u64,
    pub total_embed_time_us: u64,
    pub total_reclaimed: u64,
}

impl PipelineMetrics {
    #[must_use]
    pub fn snapshot(&self) -> PipelineMetricsSnapshot {
        PipelineMetricsSnapshot {
            total_ingest_calls: self.total_ingest_calls.load(Ordering::Relaxed),
            total_ingest_inserted: self.total_ingest_inserted.load(Ordering::Relaxed),
            total_ingest_updated: self.total_ingest_updated.load(Ordering::Relaxed),
            total_ingest_unchanged: self.total_ingest_unchanged.load(Ordering::Relaxed),
            total_ingest_skipped: self.total_ingest_skipped.load(Ordering::Relaxed),
            total_jobs_claimed: self.total_jobs_claimed.load(Ordering::Relaxed),
            total_jobs_completed: self.total_jobs_completed.load(Ordering::Relaxed),
            total_jobs_failed: self.total_jobs_failed.load(Ordering::Relaxed),
            total_jobs_skipped: self.total_jobs_skipped.load(Ordering::Relaxed),
            total_embed_time_us: self.total_embed_time_us.load(Ordering::Relaxed),
            total_reclaimed: self.total_reclaimed.load(Ordering::Relaxed),
        }
    }
}

pub trait EmbeddingVectorSink: Send + Sync {
    fn persist(&self, doc_id: &str, embedder_id: &str, embedding: &[f32]) -> SearchResult<()>;
}

#[derive(Debug, Default)]
pub struct InMemoryVectorSink {
    entries: Mutex<Vec<PersistedEmbedding>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PersistedEmbedding {
    pub doc_id: String,
    pub embedder_id: String,
    pub embedding: Vec<f32>,
}

impl InMemoryVectorSink {
    #[must_use]
    pub fn entries(&self) -> Vec<PersistedEmbedding> {
        self.entries
            .lock()
            .unwrap_or_else(|poisoned| {
                tracing::warn!(
                    target: "frankensearch.pipeline",
                    "vector sink lock poisoned; using recovered state"
                );
                poisoned.into_inner()
            })
            .clone()
    }
}

impl EmbeddingVectorSink for InMemoryVectorSink {
    fn persist(&self, doc_id: &str, embedder_id: &str, embedding: &[f32]) -> SearchResult<()> {
        {
            let mut guard = self.entries.lock().unwrap_or_else(|poisoned| {
                tracing::warn!(
                    target: "frankensearch.pipeline",
                    "vector sink lock poisoned; using recovered state"
                );
                poisoned.into_inner()
            });
            guard.push(PersistedEmbedding {
                doc_id: doc_id.to_owned(),
                embedder_id: embedder_id.to_owned(),
                embedding: embedding.to_vec(),
            });
        }
        Ok(())
    }
}

pub struct StorageBackedJobRunner {
    storage: Arc<Storage>,
    queue: Arc<PersistentJobQueue>,
    canonicalizer: Arc<dyn Canonicalizer>,
    fast_embedder: Arc<dyn Embedder>,
    quality_embedder: Option<Arc<dyn Embedder>>,
    vector_sink: Arc<dyn EmbeddingVectorSink>,
    config: PipelineConfig,
    metrics: Arc<PipelineMetrics>,
}

impl std::fmt::Debug for StorageBackedJobRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageBackedJobRunner")
            .field("config", &self.config)
            .field("fast_embedder_id", &self.fast_embedder.id())
            .field(
                "quality_embedder_id",
                &self.quality_embedder.as_ref().map(|embedder| embedder.id()),
            )
            .finish_non_exhaustive()
    }
}

impl StorageBackedJobRunner {
    #[must_use]
    pub fn new(
        storage: Arc<Storage>,
        queue: Arc<PersistentJobQueue>,
        canonicalizer: Arc<dyn Canonicalizer>,
        fast_embedder: Arc<dyn Embedder>,
        vector_sink: Arc<dyn EmbeddingVectorSink>,
    ) -> Self {
        Self {
            storage,
            queue,
            canonicalizer,
            fast_embedder,
            quality_embedder: None,
            vector_sink,
            config: PipelineConfig::default(),
            metrics: Arc::new(PipelineMetrics::default()),
        }
    }

    #[must_use]
    pub fn with_quality_embedder(mut self, quality_embedder: Arc<dyn Embedder>) -> Self {
        self.quality_embedder = Some(quality_embedder);
        self
    }

    #[must_use]
    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    #[must_use]
    pub fn with_metrics(mut self, metrics: Arc<PipelineMetrics>) -> Self {
        self.metrics = metrics;
        self
    }

    #[must_use]
    pub const fn config(&self) -> &PipelineConfig {
        &self.config
    }

    #[must_use]
    pub fn metrics(&self) -> &PipelineMetrics {
        self.metrics.as_ref()
    }

    #[allow(clippy::too_many_lines)]
    pub fn ingest(&self, request: IngestRequest) -> SearchResult<IngestResult> {
        ensure_non_empty(&request.doc_id, "doc_id")?;
        let correlation_id = resolve_correlation_id(&request.doc_id, request.correlation_id);

        self.metrics
            .total_ingest_calls
            .fetch_add(1, Ordering::Relaxed);

        let canonical_text = self.canonicalizer.canonicalize(&request.text);
        if canonical_text.trim().is_empty() {
            self.metrics
                .total_ingest_skipped
                .fetch_add(1, Ordering::Relaxed);
            tracing::info!(
                target: "frankensearch.storage.pipeline",
                stage = "ingest",
                doc_id = %request.doc_id,
                correlation_id = %correlation_id,
                action = "skip_empty_canonical_text",
                "document ingest skipped"
            );
            return Ok(IngestResult {
                doc_id: request.doc_id,
                action: IngestAction::Skipped {
                    reason: "empty_canonical_text".to_owned(),
                },
                fast_job_enqueued: false,
                quality_job_enqueued: false,
                correlation_id,
            });
        }

        /* Backpressure check moved inside transaction for TOCTOU safety */

        let now_ms = unix_timestamp_ms()?;
        let content_hash = ContentHasher::hash(&canonical_text);
        let content_hash_hex = ContentHasher::hash_hex(&canonical_text);
        let preview = truncate_chars(&canonical_text, MAX_CONTENT_PREVIEW_CHARS);
        let content_length = canonical_text.chars().count();
        let metadata = Some(with_correlation_metadata(request.metadata, &correlation_id));
        let document = DocumentRecord {
            doc_id: request.doc_id.clone(),
            source_path: request.source_path,
            content_preview: preview,
            content_hash,
            content_length,
            created_at: now_ms,
            updated_at: now_ms,
            metadata,
        };
        let fast_embedder_id = self.fast_embedder.id().to_owned();
        let maybe_quality_embedder_id = self
            .quality_embedder
            .as_ref()
            .map(|embedder| embedder.id().to_owned());

        let quality_requested = request.enqueue_quality
            && maybe_quality_embedder_id.as_deref() != Some(fast_embedder_id.as_str());

        let tx_result = self.storage.transaction(|conn| {
            let fast_dedup =
                dedup_state_for_doc(conn, &request.doc_id, &content_hash, &fast_embedder_id)?;
            let quality_needs_requeue = if fast_dedup.state == DedupState::Unchanged
                && quality_requested
            {
                if let Some(quality_embedder_id) = maybe_quality_embedder_id.as_deref() {
                    dedup_state_for_doc(conn, &request.doc_id, &content_hash, quality_embedder_id)?
                        .state
                        != DedupState::Unchanged
                } else {
                    false
                }
            } else {
                false
            };

            if fast_dedup.state == DedupState::Unchanged && !quality_needs_requeue {
                upsert_document(conn, &document)?;
                return Ok(IngestTxResult {
                    action: IngestAction::Unchanged,
                    fast_job_enqueued: false,
                    quality_job_enqueued: false,
                });
            }

            let ready_params = [SqliteValue::Integer(now_ms)];
            let ready_rows = conn
                .query_with_params(
                    "SELECT COUNT(*) FROM embedding_jobs WHERE status = 'pending' AND submitted_at <= ?1;",
                    &ready_params,
                )
                .map_err(map_storage_error)?;
            let ready_pending = if let Some(row) = ready_rows.first() {
                usize::try_from(row_i64(row, 0, "embedding_jobs.ready_pending")?)
                    .unwrap_or(usize::MAX)
            } else {
                0
            };

            if ready_pending > self.queue.config().backpressure_threshold {
                return Err(SearchError::QueueFull {
                    pending: ready_pending,
                    capacity: self.queue.config().backpressure_threshold,
                });
            }

            if fast_dedup.state == DedupState::Unchanged {
                upsert_document(conn, &document)?;
                let quality_job_enqueued =
                    if let Some(quality_embedder_id) = maybe_quality_embedder_id.as_ref() {
                        let quality_request = EnqueueRequest::new(
                            request.doc_id.clone(),
                            quality_embedder_id.clone(),
                            content_hash,
                            self.config.quality_priority,
                        );
                        let quality_outcome = enqueue_inner(
                            conn,
                            &quality_request,
                            now_ms,
                            self.queue.config().max_retries,
                        )?;
                        matches!(
                            quality_outcome,
                            EnqueueOutcome::Inserted | EnqueueOutcome::Replaced
                        )
                    } else {
                        false
                    };

                return Ok(IngestTxResult {
                    action: IngestAction::Unchanged,
                    fast_job_enqueued: false,
                    quality_job_enqueued,
                });
            }

            upsert_document(conn, &document)?;
            if fast_dedup.had_existing_row {
                reset_embedding_status(conn, &request.doc_id)?;
            }

            let _ = record_content_hash(conn, &content_hash_hex, &request.doc_id, now_ms)?;

            let fast_request = EnqueueRequest::new(
                request.doc_id.clone(),
                fast_embedder_id.clone(),
                content_hash,
                self.config.fast_priority,
            );
            let fast_outcome =
                enqueue_inner(conn, &fast_request, now_ms, self.queue.config().max_retries)?;
            let fast_job_enqueued = matches!(
                fast_outcome,
                EnqueueOutcome::Inserted | EnqueueOutcome::Replaced
            );

            let quality_job_enqueued = if quality_requested {
                if let Some(quality_embedder_id) = maybe_quality_embedder_id.as_ref() {
                    let quality_request = EnqueueRequest::new(
                        request.doc_id.clone(),
                        quality_embedder_id.clone(),
                        content_hash,
                        self.config.quality_priority,
                    );
                    let quality_outcome = enqueue_inner(
                        conn,
                        &quality_request,
                        now_ms,
                        self.queue.config().max_retries,
                    )?;
                    matches!(
                        quality_outcome,
                        EnqueueOutcome::Inserted | EnqueueOutcome::Replaced
                    )
                } else {
                    false
                }
            } else {
                false
            };

            let action = match fast_dedup.state {
                DedupState::New => IngestAction::New,
                DedupState::Changed => IngestAction::Updated,
                DedupState::Unchanged => IngestAction::Unchanged,
            };

            Ok(IngestTxResult {
                action,
                fast_job_enqueued,
                quality_job_enqueued,
            })
        })?;

        self.record_ingest_metrics(&tx_result);

        tracing::info!(
            target: "frankensearch.storage.pipeline",
            stage = "ingest",
            event = "document_ingested",
            doc_id = %request.doc_id,
            correlation_id = %correlation_id,
            action = %ingest_action_name(&tx_result.action),
            content_hash = %content_hash_hex,
            fast_job_enqueued = tx_result.fast_job_enqueued,
            quality_job_enqueued = tx_result.quality_job_enqueued,
            "document ingest completed"
        );

        Ok(IngestResult {
            doc_id: request.doc_id,
            action: tx_result.action,
            fast_job_enqueued: tx_result.fast_job_enqueued,
            quality_job_enqueued: tx_result.quality_job_enqueued,
            correlation_id,
        })
    }

    pub fn ingest_batch(&self, requests: &[IngestRequest]) -> SearchResult<BatchIngestResult> {
        let mut summary = BatchIngestResult {
            requested: requests.len(),
            ..BatchIngestResult::default()
        };
        for request in requests {
            let result = self.ingest(request.clone())?;
            match result.action {
                IngestAction::New => summary.inserted += 1,
                IngestAction::Updated => summary.updated += 1,
                IngestAction::Unchanged => summary.unchanged += 1,
                IngestAction::Skipped { .. } => summary.skipped += 1,
            }
            if result.fast_job_enqueued {
                summary.fast_jobs_enqueued += 1;
            }
            if result.quality_job_enqueued {
                summary.quality_jobs_enqueued += 1;
            }
        }
        Ok(summary)
    }

    #[allow(clippy::too_many_lines, clippy::future_not_send)]
    pub async fn process_batch(
        &self,
        cx: &Cx,
        worker_id: &str,
    ) -> SearchResult<BatchProcessResult> {
        ensure_non_empty(worker_id, "worker_id")?;
        let total_start = Instant::now();
        let claimed = self
            .queue
            .claim_batch(worker_id, self.config.process_batch_size)?;
        let mut result = BatchProcessResult {
            jobs_claimed: claimed.len(),
            ..BatchProcessResult::default()
        };

        if claimed.is_empty() {
            result.total_time = total_start.elapsed();
            return Ok(result);
        }

        self.metrics
            .total_jobs_claimed
            .fetch_add(usize_to_u64(claimed.len()), Ordering::Relaxed);

        let embed_start = Instant::now();
        for job in &claimed {
            let job_started = Instant::now();
            let doc = self.storage.get_document(&job.doc_id)?;
            let Some(doc) = doc else {
                let message = format!("document {} missing during process_batch", job.doc_id);
                if let Err(fail_err) = self.queue.fail(job.job_id, &message) {
                    tracing::warn!(
                        target: "frankensearch.storage.pipeline",
                        job_id = job.job_id,
                        error = %fail_err,
                        "failed to record job failure in queue"
                    );
                }
                result.jobs_failed += 1;
                tracing::warn!(
                    target: "frankensearch.storage.pipeline",
                    stage = "process_batch",
                    worker_id,
                    doc_id = %job.doc_id,
                    embedder_id = %job.embedder_id,
                    reason = "document_missing",
                    "embedding job failed"
                );
                continue;
            };

            let correlation_id = extract_correlation_id(doc.metadata.as_ref())
                .unwrap_or_else(|| fallback_correlation_id(&job.doc_id));

            let text_cow = if let Some(path) = &doc.source_path {
                match read_source_text_with_limit(path, MAX_SOURCE_FILE_BYTES) {
                    Ok(raw) => std::borrow::Cow::Owned(self.canonicalizer.canonicalize(&raw)),
                    Err(error) => {
                        tracing::warn!(
                            target: "frankensearch.storage.pipeline",
                            stage = "process_batch",
                            worker_id,
                            doc_id = %job.doc_id,
                            path = %path,
                            max_source_file_bytes = MAX_SOURCE_FILE_BYTES,
                            error = %error,
                            "failed to read source file; falling back to content preview"
                        );
                        std::borrow::Cow::Borrowed(doc.content_preview.as_str())
                    }
                }
            } else {
                if doc.content_length > MAX_CONTENT_PREVIEW_CHARS {
                    tracing::warn!(
                        target: "frankensearch.storage.pipeline",
                        stage = "process_batch",
                        worker_id,
                        doc_id = %job.doc_id,
                        content_length = doc.content_length,
                        "document has no source_path and exceeds preview length; embedding will be truncated"
                    );
                }
                std::borrow::Cow::Borrowed(doc.content_preview.as_str())
            };
            let text = text_cow.as_ref();

            if text.trim().is_empty() {
                let skip_reason = "empty content preview";
                self.queue.skip(job.job_id, skip_reason)?;
                if let Err(error) = self.storage.mark_skipped(&job.doc_id, &job.embedder_id, skip_reason) {
                    tracing::warn!(
                        target: "frankensearch.storage.pipeline",
                        stage = "mark_skipped",
                        worker_id,
                        correlation_id = %correlation_id,
                        doc_id = %job.doc_id,
                        embedder_id = %job.embedder_id,
                        error = %error,
                        "failed to record skipped status"
                    );
                }
                result.jobs_skipped += 1;
                tracing::info!(
                    target: "frankensearch.storage.pipeline",
                    stage = "process_batch",
                    worker_id,
                    correlation_id = %correlation_id,
                    doc_id = %job.doc_id,
                    embedder_id = %job.embedder_id,
                    reason = "empty_content_preview",
                    "embedding job skipped"
                );
                continue;
            }

            if is_hash_embedder(&job.embedder_id) {
                let skip_reason = "hash embeddings computed on-the-fly";
                self.queue.skip(job.job_id, skip_reason)?;
                if let Err(error) = self.storage.mark_skipped(&job.doc_id, &job.embedder_id, skip_reason) {
                    tracing::warn!(
                        target: "frankensearch.storage.pipeline",
                        stage = "mark_skipped",
                        worker_id,
                        correlation_id = %correlation_id,
                        doc_id = %job.doc_id,
                        embedder_id = %job.embedder_id,
                        error = %error,
                        "failed to record skipped status"
                    );
                }
                result.jobs_skipped += 1;
                tracing::info!(
                    target: "frankensearch.storage.pipeline",
                    stage = "process_batch",
                    worker_id,
                    correlation_id = %correlation_id,
                    doc_id = %job.doc_id,
                    embedder_id = %job.embedder_id,
                    reason = "hash_embedder_on_the_fly",
                    "embedding job skipped"
                );
                continue;
            }

            let embedder = match self.embedder_for_id(&job.embedder_id) {
                Ok(embedder) => embedder,
                Err(error) => {
                    let error_message = error.to_string();
                    self.handle_job_failure(job, &error);
                    result.jobs_failed += 1;
                    tracing::warn!(
                        target: "frankensearch.storage.pipeline",
                        stage = "process_batch",
                        worker_id,
                        doc_id = %job.doc_id,
                        embedder_id = %job.embedder_id,
                        error = %error_message,
                        "failed to resolve embedder for queued job"
                    );
                    continue;
                }
            };
            let embedding = match embedder.embed(cx, text).await {
                Ok(embedding) => embedding,
                Err(error) => {
                    self.handle_job_failure(job, &error);
                    result.jobs_failed += 1;
                    tracing::warn!(
                        target: "frankensearch.storage.pipeline",
                        stage = "embed",
                        worker_id,
                        correlation_id = %correlation_id,
                        doc_id = %job.doc_id,
                        embedder_id = %job.embedder_id,
                        error = %error,
                        "embedding inference failed"
                    );
                    continue;
                }
            };

            let write_result = self
                .vector_sink
                .persist(&job.doc_id, &job.embedder_id, &embedding);
            if let Err(error) = write_result {
                self.handle_job_failure(job, &error);
                result.jobs_failed += 1;
                tracing::warn!(
                    target: "frankensearch.storage.pipeline",
                    stage = "persist",
                    worker_id,
                    correlation_id = %correlation_id,
                    doc_id = %job.doc_id,
                    embedder_id = %job.embedder_id,
                    error = %error,
                    "embedding persistence failed"
                );
                continue;
            }

            if let Err(error) = self.storage.mark_embedded(&job.doc_id, &job.embedder_id) {
                self.handle_job_failure(job, &error);
                result.jobs_failed += 1;
                tracing::warn!(
                    target: "frankensearch.storage.pipeline",
                    stage = "mark_embedded",
                    worker_id,
                    correlation_id = %correlation_id,
                    doc_id = %job.doc_id,
                    embedder_id = %job.embedder_id,
                    error = %error,
                    "failed to record embedded status"
                );
                continue;
            }
            if let Err(error) = self.queue.complete(job.job_id) {
                if crate::job_queue::is_queue_conflict(&error) {
                    let skip_reason =
                        "embedding persisted after completion conflict; skipping reclaimed job";
                    match self.queue.skip(job.job_id, skip_reason) {
                        Ok(()) => {}
                        Err(skip_error) if crate::job_queue::is_queue_conflict(&skip_error) => {}
                        Err(skip_error) => return Err(skip_error),
                    }
                    result.jobs_skipped += 1;
                    tracing::warn!(
                        target: "frankensearch.storage.pipeline",
                        stage = "complete_conflict",
                        worker_id,
                        correlation_id = %correlation_id,
                        job_id = job.job_id,
                        doc_id = %job.doc_id,
                        embedder_id = %job.embedder_id,
                        error = %error,
                        "queue completion raced with lease reclaim; job left non-fatal"
                    );
                    continue;
                }
                return Err(error);
            }
            result.jobs_completed += 1;
            tracing::info!(
                target: "frankensearch.storage.pipeline",
                stage = "complete",
                event = "job_completed",
                worker_id,
                correlation_id = %correlation_id,
                job_id = job.job_id,
                doc_id = %job.doc_id,
                embedder_id = %job.embedder_id,
                duration_ms = duration_as_u64(job_started.elapsed().as_millis()),
                "embedding job completed"
            );
        }

        result.embed_time = embed_start.elapsed();
        result.total_time = total_start.elapsed();
        self.record_process_metrics(&result);
        Ok(result)
    }

    #[allow(clippy::future_not_send)]
    pub async fn run_worker(
        &self,
        cx: &Cx,
        worker_id: &str,
        shutdown: &AtomicBool,
    ) -> SearchResult<WorkerReport> {
        ensure_non_empty(worker_id, "worker_id")?;

        let reclaimed = self.queue.reclaim_stale_jobs()?;
        self.metrics
            .total_reclaimed
            .fetch_add(usize_to_u64(reclaimed), Ordering::Relaxed);
        if reclaimed > 0 {
            tracing::warn!(
                target: "frankensearch.storage.pipeline",
                stage = "startup",
                event = "crash_recovery",
                worker_id,
                recovered_job_count = reclaimed,
                "reclaimed stale embedding jobs on worker startup"
            );
        }

        let mut report = WorkerReport {
            reclaimed_on_startup: reclaimed,
            ..WorkerReport::default()
        };
        let mut idle_cycles = 0_usize;

        while !shutdown.load(Ordering::Relaxed) {
            let batch = self.process_batch(cx, worker_id).await?;
            if batch.jobs_claimed == 0 {
                idle_cycles += 1;
                report.idle_cycles = idle_cycles;
                if self
                    .config
                    .worker_max_idle_cycles
                    .is_some_and(|limit| idle_cycles >= limit)
                {
                    break;
                }
                asupersync::time::sleep(
                    asupersync::time::wall_now(),
                    Duration::from_millis(self.config.worker_idle_sleep_ms),
                )
                .await;
                continue;
            }

            idle_cycles = 0;
            report.batches_processed += 1;
            report.jobs_completed += batch.jobs_completed;
            report.jobs_failed += batch.jobs_failed;
            report.jobs_skipped += batch.jobs_skipped;
        }

        tracing::info!(
            target: "frankensearch.storage.pipeline",
            stage = "worker_exit",
            worker_id,
            reclaimed_on_startup = report.reclaimed_on_startup,
            batches_processed = report.batches_processed,
            jobs_completed = report.jobs_completed,
            jobs_failed = report.jobs_failed,
            jobs_skipped = report.jobs_skipped,
            idle_cycles = report.idle_cycles,
            "storage-backed embedding worker exited"
        );

        Ok(report)
    }

    fn embedder_for_id(&self, embedder_id: &str) -> SearchResult<Arc<dyn Embedder>> {
        if self.fast_embedder.id() == embedder_id {
            return Ok(Arc::clone(&self.fast_embedder));
        }
        if let Some(quality) = self.quality_embedder.as_ref()
            && quality.id() == embedder_id
        {
            return Ok(Arc::clone(quality));
        }
        Err(pipeline_error(format!(
            "no embedder configured for queued embedder_id {embedder_id:?}"
        )))
    }

    fn handle_job_failure(&self, job: &crate::ClaimedJob, error: &SearchError) {
        let error_message = error.to_string();
        let fail_result = self.queue.fail(job.job_id, &error_message);
        
        let is_terminal = match &fail_result {
            Ok(crate::job_queue::FailResult::TerminalFailed { .. }) => true,
            Ok(crate::job_queue::FailResult::Retried { .. }) => false,
            Err(_) => true,
        };

        if let Err(fail_err) = fail_result {
            tracing::warn!(
                target: "frankensearch.storage.pipeline",
                job_id = job.job_id,
                error = %fail_err,
                "failed to record job failure in queue"
            );
        }
        
        if is_terminal {
            if let Err(mark_err) =
                self.storage
                    .mark_failed(&job.doc_id, &job.embedder_id, &error_message)
            {
                tracing::warn!(
                    target: "frankensearch.storage.pipeline",
                    doc_id = %job.doc_id,
                    error = %mark_err,
                    "failed to mark document as failed in storage"
                );
            }
        }
    }

    fn record_ingest_metrics(&self, tx_result: &IngestTxResult) {
        match tx_result.action {
            IngestAction::New => {
                self.metrics
                    .total_ingest_inserted
                    .fetch_add(1, Ordering::Relaxed);
            }
            IngestAction::Updated => {
                self.metrics
                    .total_ingest_updated
                    .fetch_add(1, Ordering::Relaxed);
            }
            IngestAction::Unchanged => {
                self.metrics
                    .total_ingest_unchanged
                    .fetch_add(1, Ordering::Relaxed);
            }
            IngestAction::Skipped { .. } => {
                self.metrics
                    .total_ingest_skipped
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn record_process_metrics(&self, result: &BatchProcessResult) {
        self.metrics
            .total_jobs_completed
            .fetch_add(usize_to_u64(result.jobs_completed), Ordering::Relaxed);
        self.metrics
            .total_jobs_failed
            .fetch_add(usize_to_u64(result.jobs_failed), Ordering::Relaxed);
        self.metrics
            .total_jobs_skipped
            .fetch_add(usize_to_u64(result.jobs_skipped), Ordering::Relaxed);

        let embed_time_us = duration_as_u64(result.embed_time.as_micros());
        self.metrics
            .total_embed_time_us
            .fetch_add(embed_time_us, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DedupState {
    New,
    Changed,
    Unchanged,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DedupCheckResult {
    state: DedupState,
    had_existing_row: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IngestTxResult {
    action: IngestAction,
    fast_job_enqueued: bool,
    quality_job_enqueued: bool,
}

fn dedup_state_for_doc(
    conn: &Connection,
    doc_id: &str,
    new_hash: &[u8; 32],
    embedder_id: &str,
) -> SearchResult<DedupCheckResult> {
    let params = [
        SqliteValue::Text(doc_id.to_owned()),
        SqliteValue::Text(embedder_id.to_owned()),
    ];
    let rows = conn
        .query_with_params(
            "SELECT d.content_hash, e.status, j.job_id \
             FROM documents d \
             LEFT JOIN embedding_status e \
               ON d.doc_id = e.doc_id AND e.embedder_id = ?2 \
             LEFT JOIN embedding_jobs j \
               ON d.doc_id = j.doc_id \
              AND j.embedder_id = ?2 \
              AND j.status IN ('pending', 'processing') \
             WHERE d.doc_id = ?1 \
             LIMIT 1;",
            &params,
        )
        .map_err(map_storage_error)?;
    let Some(row) = rows.first() else {
        return Ok(DedupCheckResult {
            state: DedupState::New,
            had_existing_row: false,
        });
    };

    let existing_hash = row_blob_32(row, 0, "documents.content_hash")?;
    if &existing_hash != new_hash {
        return Ok(DedupCheckResult {
            state: DedupState::Changed,
            had_existing_row: true,
        });
    }

    let raw_status = row_optional_text(row, 1)?;
    let status = raw_status.as_deref().and_then(EmbeddingStatus::from_str);
    let has_active_job = match row.get(2) {
        Some(SqliteValue::Integer(_)) => true,
        Some(SqliteValue::Null) | None => false,
        Some(other) => {
            return Err(pipeline_error(format!(
                "unexpected type for dedup active-job flag: {other:?}"
            )));
        }
    };
    let state = if has_active_job {
        DedupState::Unchanged
    } else {
        match status {
            Some(EmbeddingStatus::Embedded | EmbeddingStatus::Skipped) => DedupState::Unchanged,
            Some(EmbeddingStatus::Pending | EmbeddingStatus::Failed) | None => DedupState::New,
        }
    };

    Ok(DedupCheckResult {
        state,
        had_existing_row: true,
    })
}

fn reset_embedding_status(conn: &Connection, doc_id: &str) -> SearchResult<()> {
    let params = [SqliteValue::Text(doc_id.to_owned())];
    conn.execute_with_params("DELETE FROM embedding_status WHERE doc_id = ?1;", &params)
        .map_err(map_storage_error)?;
    Ok(())
}

fn with_correlation_metadata(metadata: Option<Value>, correlation_id: &str) -> Value {
    let mut object = match metadata {
        Some(Value::Object(map)) => map,
        Some(other) => {
            let mut map = Map::new();
            map.insert("payload".to_owned(), other);
            map
        }
        None => Map::new(),
    };
    object.insert(
        CORRELATION_METADATA_KEY.to_owned(),
        Value::String(correlation_id.to_owned()),
    );
    Value::Object(object)
}

fn extract_correlation_id(metadata: Option<&Value>) -> Option<String> {
    metadata?
        .as_object()?
        .get(CORRELATION_METADATA_KEY)?
        .as_str()
        .map(ToOwned::to_owned)
}

fn resolve_correlation_id(doc_id: &str, correlation_id: Option<String>) -> String {
    match correlation_id {
        Some(correlation_id) if !correlation_id.trim().is_empty() => correlation_id,
        _ => fallback_correlation_id(doc_id),
    }
}

fn fallback_correlation_id(doc_id: &str) -> String {
    let millis = unix_timestamp_ms().unwrap_or(0);
    format!("ingest-{doc_id}-{millis}")
}

fn ingest_action_name(action: &IngestAction) -> &'static str {
    match action {
        IngestAction::New => "new",
        IngestAction::Updated => "updated",
        IngestAction::Unchanged => "unchanged",
        IngestAction::Skipped { .. } => "skipped",
    }
}

fn ensure_non_empty(value: &str, field: &str) -> SearchResult<()> {
    if value.trim().is_empty() {
        return Err(pipeline_error(format!("{field} must not be empty")));
    }
    Ok(())
}

fn pipeline_error(message: impl Into<String>) -> SearchError {
    SearchError::SubsystemError {
        subsystem: PIPELINE_SUBSYSTEM,
        source: Box::new(io::Error::other(message.into())),
    }
}

fn is_hash_embedder(embedder_id: &str) -> bool {
    embedder_id.starts_with(HASH_EMBEDDER_PREFIX)
        || embedder_id.starts_with(JL_EMBEDDER_PREFIX)
        || embedder_id == LEGACY_HASH_EMBEDDER_ID
}

fn read_source_text_with_limit(path: &str, max_bytes: usize) -> io::Result<String> {
    let max_bytes_u64 = u64::try_from(max_bytes).unwrap_or(u64::MAX);
    let metadata = std::fs::metadata(path)?;
    if metadata.len() > max_bytes_u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("source file exceeds {max_bytes} bytes"),
        ));
    }

    let initial_capacity = usize::try_from(metadata.len())
        .unwrap_or(max_bytes)
        .min(max_bytes);
    let mut bytes = Vec::with_capacity(initial_capacity);
    let file = std::fs::File::open(path)?;
    file.take(max_bytes_u64.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() > max_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("source file exceeds {max_bytes} bytes"),
        ));
    }

    String::from_utf8(bytes).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_owned();
    }
    value.chars().take(max_chars).collect()
}

fn unix_timestamp_ms() -> SearchResult<i64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(map_storage_error)?;
    i64::try_from(duration.as_millis())
        .map_err(|_| pipeline_error("system timestamp overflowed i64 milliseconds"))
}

fn row_optional_text(row: &fsqlite::Row, index: usize) -> SearchResult<Option<String>> {
    match row.get(index) {
        Some(SqliteValue::Text(value)) => Ok(Some(value.clone())),
        Some(SqliteValue::Null) | None => Ok(None),
        Some(other) => Err(pipeline_error(format!(
            "unexpected optional text column type at index {index}: {other:?}"
        ))),
    }
}

fn row_blob_32(row: &fsqlite::Row, index: usize, field: &str) -> SearchResult<[u8; 32]> {
    let bytes = match row.get(index) {
        Some(SqliteValue::Blob(value)) => value,
        Some(other) => {
            return Err(pipeline_error(format!(
                "unexpected type for {field}: {other:?}"
            )));
        }
        None => return Err(pipeline_error(format!("missing column for {field}"))),
    };

    if bytes.len() != 32 {
        return Err(pipeline_error(format!(
            "{field} expected 32 bytes, found {}",
            bytes.len()
        )));
    }

    let mut out = [0_u8; 32];
    out.copy_from_slice(bytes);
    Ok(out)
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn duration_as_u64(value: u128) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::io::Write;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, Mutex};

    use frankensearch_core::canonicalize::DefaultCanonicalizer;
    use frankensearch_core::traits::{ModelCategory, SearchFuture};

    use crate::job_queue::{FailResult, JobQueueConfig};

    use super::*;

    #[derive(Debug)]
    struct StubEmbedder {
        id: &'static str,
        dim: usize,
        fail_on_substring: Option<&'static str>,
        fill: f32,
    }

    impl StubEmbedder {
        const fn new(
            id: &'static str,
            dim: usize,
            fail_on_substring: Option<&'static str>,
            fill: f32,
        ) -> Self {
            Self {
                id,
                dim,
                fail_on_substring,
                fill,
            }
        }
    }

    impl Embedder for StubEmbedder {
        fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            let should_fail = self
                .fail_on_substring
                .is_some_and(|needle| text.contains(needle));
            let dim = self.dim;
            let fill = self.fill;
            let model_id = self.id.to_owned();
            Box::pin(async move {
                if should_fail {
                    return Err(SearchError::EmbeddingFailed {
                        model: model_id,
                        source: Box::new(io::Error::other("stub embedder failure")),
                    });
                }
                Ok(vec![fill; dim])
            })
        }

        fn dimension(&self) -> usize {
            self.dim
        }

        fn id(&self) -> &str {
            self.id
        }

        fn model_name(&self) -> &str {
            self.id
        }

        fn is_semantic(&self) -> bool {
            !is_hash_embedder(self.id)
        }

        fn category(&self) -> ModelCategory {
            if is_hash_embedder(self.id) {
                ModelCategory::HashEmbedder
            } else {
                ModelCategory::StaticEmbedder
            }
        }
    }

    #[allow(clippy::arc_with_non_send_sync)]
    fn make_runner(
        queue_config: JobQueueConfig,
        pipeline_config: PipelineConfig,
        fast_embedder: Arc<dyn Embedder>,
        quality_embedder: Option<Arc<dyn Embedder>>,
        sink: Arc<InMemoryVectorSink>,
    ) -> StorageBackedJobRunner {
        let storage = Arc::new(Storage::open_in_memory().expect("storage should open"));
        let queue = Arc::new(PersistentJobQueue::new(Arc::clone(&storage), queue_config));
        let canonicalizer: Arc<dyn Canonicalizer> = Arc::new(DefaultCanonicalizer::default());
        let mut runner =
            StorageBackedJobRunner::new(storage, queue, canonicalizer, fast_embedder, sink)
                .with_config(pipeline_config);
        if let Some(quality_embedder) = quality_embedder {
            runner = runner.with_quality_embedder(quality_embedder);
        }
        runner
    }

    #[derive(Debug)]
    struct TestLogWriter {
        buffer: Arc<Mutex<Vec<u8>>>,
    }

    impl Write for TestLogWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.buffer
                .lock()
                .unwrap_or_else(|poisoned| {
                    tracing::warn!(
                        target: "frankensearch.pipeline",
                        "test log buffer lock poisoned; using recovered state"
                    );
                    poisoned.into_inner()
                })
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn with_captured_logs<R>(run: impl FnOnce() -> R) -> (R, String) {
        let buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
        let writer_buffer = Arc::clone(&buffer);
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .without_time()
            .with_writer(move || TestLogWriter {
                buffer: Arc::clone(&writer_buffer),
            })
            .finish();
        let result = tracing::subscriber::with_default(subscriber, run);
        let logs = {
            let guard = buffer.lock().unwrap_or_else(|poisoned| {
                tracing::warn!(
                    target: "frankensearch.pipeline",
                    "test log buffer lock poisoned; using recovered state"
                );
                poisoned.into_inner()
            });
            String::from_utf8_lossy(&guard).into_owned()
        };
        (result, logs)
    }

    #[test]
    fn ingest_new_document_enqueues_fast_and_quality_jobs() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fast-tier", 4, None, 1.0));
            let quality = Arc::new(StubEmbedder::new("quality-tier", 4, None, 2.0));
            let runner = make_runner(
                JobQueueConfig::default(),
                PipelineConfig::default(),
                fast,
                Some(quality),
                Arc::clone(&sink),
            );

            let ingest = runner
                .ingest(IngestRequest::new(
                    "doc-1",
                    "Rust ownership improves reliability",
                ))
                .expect("ingest should succeed");
            assert_eq!(ingest.action, IngestAction::New);
            assert!(ingest.fast_job_enqueued);
            assert!(ingest.quality_job_enqueued);

            let processed = runner
                .process_batch(&cx, "worker-a")
                .await
                .expect("process_batch should succeed");
            assert_eq!(processed.jobs_claimed, 2);
            assert_eq!(processed.jobs_completed, 2);

            let fast_counts = runner
                .storage
                .count_by_status("fast-tier")
                .expect("status counts should succeed");
            let quality_counts = runner
                .storage
                .count_by_status("quality-tier")
                .expect("status counts should succeed");
            assert_eq!(fast_counts.embedded, 1);
            assert_eq!(quality_counts.embedded, 1);

            let persisted = sink.entries();
            assert_eq!(persisted.len(), 2);
        });
    }

    #[test]
    fn ingest_unchanged_document_skips_reenqueue() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fast-tier", 3, None, 1.0));
            let runner = make_runner(
                JobQueueConfig::default(),
                PipelineConfig::default(),
                fast,
                None,
                sink,
            );

            let _ = runner
                .ingest(IngestRequest::new("doc-stable", "same content"))
                .expect("initial ingest should succeed");
            let _ = runner
                .process_batch(&cx, "worker-a")
                .await
                .expect("initial process should succeed");

            let second = runner
                .ingest(IngestRequest::new("doc-stable", "same content"))
                .expect("second ingest should succeed");
            assert_eq!(second.action, IngestAction::Unchanged);
            assert!(!second.fast_job_enqueued);
            assert!(!second.quality_job_enqueued);

            let depth = runner
                .queue
                .queue_depth()
                .expect("queue depth should succeed");
            assert_eq!(depth.pending, 0);
            assert_eq!(depth.processing, 0);
        });
    }

    #[test]
    fn ingest_unchanged_document_requeues_quality_after_quality_failure() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fast-tier", 3, None, 1.0));
            let quality = Arc::new(StubEmbedder::new(
                "quality-tier",
                3,
                Some("fail-quality"),
                2.0,
            ));
            let runner = make_runner(
                JobQueueConfig {
                    max_retries: 0,
                    ..JobQueueConfig::default()
                },
                PipelineConfig::default(),
                fast,
                Some(quality),
                sink,
            );

            let first = runner
                .ingest(IngestRequest::new("doc-quality-retry", "fail-quality"))
                .expect("initial ingest should succeed");
            assert_eq!(first.action, IngestAction::New);
            assert!(first.fast_job_enqueued);
            assert!(first.quality_job_enqueued);

            let processed = runner
                .process_batch(&cx, "worker-quality-failure")
                .await
                .expect("process_batch should succeed");
            assert_eq!(processed.jobs_claimed, 2);
            assert_eq!(processed.jobs_completed, 1);
            assert_eq!(processed.jobs_failed, 1);

            let second = runner
                .ingest(IngestRequest::new("doc-quality-retry", "fail-quality"))
                .expect("second ingest should succeed");
            assert_eq!(second.action, IngestAction::Unchanged);
            assert!(!second.fast_job_enqueued);
            assert!(
                second.quality_job_enqueued,
                "quality job should requeue when quality tier previously failed"
            );

            let depth = runner
                .queue
                .queue_depth()
                .expect("queue depth should succeed");
            assert_eq!(depth.pending, 1);
        });
    }

    #[test]
    fn ingest_unchanged_document_with_pending_job_is_not_reenqueued() {
        let sink = Arc::new(InMemoryVectorSink::default());
        let fast = Arc::new(StubEmbedder::new("fast-tier", 3, None, 1.0));
        let runner = make_runner(
            JobQueueConfig::default(),
            PipelineConfig::default(),
            fast,
            None,
            sink,
        );

        let first = runner
            .ingest(IngestRequest::new("doc-pending", "same content"))
            .expect("initial ingest should succeed");
        assert_eq!(first.action, IngestAction::New);
        assert!(first.fast_job_enqueued);

        let second = runner
            .ingest(IngestRequest::new("doc-pending", "same content"))
            .expect("second ingest should succeed");
        assert_eq!(second.action, IngestAction::Unchanged);
        assert!(!second.fast_job_enqueued);
        assert!(!second.quality_job_enqueued);

        let depth = runner
            .queue
            .queue_depth()
            .expect("queue depth should succeed");
        assert_eq!(depth.pending, 1, "pending job should be deduplicated");
    }

    #[test]
    fn process_batch_failure_marks_terminal_failed_with_zero_retries() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fast-tier", 4, Some("fail-me"), 1.0));
            let runner = make_runner(
                JobQueueConfig {
                    max_retries: 0,
                    ..JobQueueConfig::default()
                },
                PipelineConfig::default(),
                fast,
                None,
                sink,
            );

            let _ = runner
                .ingest(IngestRequest::new("doc-fail", "please fail-me now"))
                .expect("ingest should succeed");
            let processed = runner
                .process_batch(&cx, "worker-fail")
                .await
                .expect("process_batch should succeed");

            assert_eq!(processed.jobs_claimed, 1);
            assert_eq!(processed.jobs_completed, 0);
            assert_eq!(processed.jobs_failed, 1);

            let depth = runner
                .queue
                .queue_depth()
                .expect("queue depth should succeed");
            assert_eq!(depth.failed, 1);

            let counts = runner
                .storage
                .count_by_status("fast-tier")
                .expect("status counts should succeed");
            assert_eq!(counts.failed, 1);
        });
    }

    #[test]
    fn process_batch_unknown_embedder_fails_job_without_aborting_batch() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fast-tier", 4, None, 1.0));
            let runner = make_runner(
                JobQueueConfig {
                    max_retries: 0,
                    ..JobQueueConfig::default()
                },
                PipelineConfig::default(),
                fast,
                None,
                sink,
            );

            let _ = runner
                .ingest(IngestRequest::new(
                    "doc-unknown-embedder",
                    "valid content for processing",
                ))
                .expect("ingest should succeed");

            let queued_unknown = runner
                .queue
                .enqueue("doc-unknown-embedder", "missing-tier", &[7_u8; 32], 0)
                .expect("unknown embedder job enqueue should succeed");
            assert!(queued_unknown, "unknown embedder job should be enqueued");

            let processed = runner
                .process_batch(&cx, "worker-unknown-embedder")
                .await
                .expect("process_batch should not abort on unknown embedder id");

            assert_eq!(processed.jobs_claimed, 2);
            assert_eq!(processed.jobs_completed, 1);
            assert_eq!(processed.jobs_failed, 1);

            let depth = runner
                .queue
                .queue_depth()
                .expect("queue depth should succeed");
            assert_eq!(depth.processing, 0);
            assert_eq!(depth.failed, 1);

            let missing_tier_counts = runner
                .storage
                .count_by_status("missing-tier")
                .expect("missing-tier status counts should succeed");
            assert_eq!(missing_tier_counts.failed, 1);
            assert_eq!(missing_tier_counts.pending, 0);
        });
    }

    #[test]
    fn run_worker_reclaims_stale_jobs_before_processing() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fast-tier", 2, None, 1.0));
            let runner = make_runner(
                JobQueueConfig {
                    visibility_timeout_ms: 5,
                    stale_job_threshold_ms: 5,
                    ..JobQueueConfig::default()
                },
                PipelineConfig {
                    process_batch_size: 1,
                    worker_idle_sleep_ms: 1,
                    worker_max_idle_cycles: Some(1),
                    ..PipelineConfig::default()
                },
                fast,
                None,
                Arc::clone(&sink),
            );

            let _ = runner
                .ingest(IngestRequest::new("doc-stale", "stale queue row"))
                .expect("ingest should succeed");
            let claimed = runner
                .queue
                .claim_batch("worker-pre", 1)
                .expect("claim should succeed");
            assert_eq!(claimed.len(), 1);

            let stale_started = unix_timestamp_ms()
                .expect("timestamp should resolve")
                .saturating_sub(10_000);
            let params = [
                SqliteValue::Integer(stale_started),
                SqliteValue::Integer(claimed[0].job_id),
            ];
            runner
                .storage
                .connection()
                .execute_with_params(
                    "UPDATE embedding_jobs SET started_at = ?1 WHERE job_id = ?2;",
                    &params,
                )
                .expect("stale timestamp update should succeed");

            let shutdown = AtomicBool::new(false);
            let report = runner
                .run_worker(&cx, "worker-main", &shutdown)
                .await
                .expect("run_worker should succeed");
            assert_eq!(report.reclaimed_on_startup, 1);
            assert_eq!(report.jobs_completed, 1);
            assert_eq!(sink.entries().len(), 1);
        });
    }

    #[test]
    fn process_batch_with_multiple_workers_keeps_disjoint_results() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fast-tier", 3, None, 0.5));
            let runner = make_runner(
                JobQueueConfig::default(),
                PipelineConfig {
                    process_batch_size: 1,
                    ..PipelineConfig::default()
                },
                fast,
                None,
                Arc::clone(&sink),
            );

            for idx in 0..3 {
                let request = IngestRequest::new(format!("doc-{idx}"), format!("text-{idx}"));
                let _ = runner.ingest(request).expect("ingest should succeed");
            }

            let _ = runner
                .process_batch(&cx, "worker-a")
                .await
                .expect("worker a should succeed");
            let _ = runner
                .process_batch(&cx, "worker-b")
                .await
                .expect("worker b should succeed");
            let _ = runner
                .process_batch(&cx, "worker-c")
                .await
                .expect("worker c should succeed");

            let persisted = sink.entries();
            let doc_ids: HashSet<_> = persisted.iter().map(|entry| entry.doc_id.clone()).collect();
            assert_eq!(persisted.len(), 3);
            assert_eq!(doc_ids.len(), 3);

            let depth = runner
                .queue
                .queue_depth()
                .expect("queue depth should succeed");
            assert_eq!(depth.pending, 0);
            assert_eq!(depth.processing, 0);
        });
    }

    #[test]
    fn ingest_recovers_from_zombie_pending_state() {
        asupersync::test_utils::run_test_with_cx(|_cx| async move {
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fast-tier", 2, None, 1.0));
            let runner = make_runner(
                JobQueueConfig::default(),
                PipelineConfig::default(),
                fast,
                None,
                sink,
            );

            // 1. Ingest document (creates job + pending status)
            let _ = runner
                .ingest(IngestRequest::new("doc-zombie", "content"))
                .expect("initial ingest should succeed");

            let initial_depth = runner.queue.queue_depth().expect("depth");
            assert_eq!(initial_depth.pending, 1);

            // 2. Manually delete the job to create zombie state
            // (Status remains 'pending' in embedding_status, but job is gone)
            runner
                .storage
                .connection()
                .execute("DELETE FROM embedding_jobs;")
                .expect("manual delete should succeed");

            let zombie_depth = runner.queue.queue_depth().expect("depth");
            assert_eq!(zombie_depth.pending, 0);

            // 3. Ingest same document again
            let repair = runner
                .ingest(IngestRequest::new("doc-zombie", "content"))
                .expect("repair ingest should succeed");

            // 4. Should be treated as New (to trigger re-enqueue)
            assert_eq!(repair.action, IngestAction::New);
            assert!(repair.fast_job_enqueued);

            // 5. Job should be back
            let repaired_depth = runner.queue.queue_depth().expect("depth");
            assert_eq!(repaired_depth.pending, 1);
        });
    }

    #[test]
    fn ingest_stores_correlation_id_in_metadata() {
        let sink = Arc::new(InMemoryVectorSink::default());
        let fast = Arc::new(StubEmbedder::new("fast-tier", 2, None, 1.0));
        let runner = make_runner(
            JobQueueConfig::default(),
            PipelineConfig::default(),
            fast,
            None,
            sink,
        );

        let mut request = IngestRequest::new("doc-corr", "correlated");
        request.correlation_id = Some("corr-123".to_owned());
        let result = runner.ingest(request).expect("ingest should succeed");
        assert_eq!(result.correlation_id, "corr-123");

        let doc = runner
            .storage
            .get_document("doc-corr")
            .expect("fetch should succeed")
            .expect("document should exist");
        let stored =
            extract_correlation_id(doc.metadata.as_ref()).expect("correlation id should exist");
        assert_eq!(stored, "corr-123");
    }

    #[test]
    fn ingest_unchanged_document_updates_metadata_without_reenqueue() {
        let sink = Arc::new(InMemoryVectorSink::default());
        let fast = Arc::new(StubEmbedder::new("fast-tier", 2, None, 1.0));
        let runner = make_runner(
            JobQueueConfig::default(),
            PipelineConfig::default(),
            fast,
            None,
            sink,
        );

        let mut first = IngestRequest::new("doc-metadata-refresh", "stable content");
        first.correlation_id = Some("corr-1".to_owned());
        first.metadata = Some(Value::String("first".to_owned()));
        let first_result = runner.ingest(first).expect("first ingest should succeed");
        assert_eq!(first_result.action, IngestAction::New);
        assert!(first_result.fast_job_enqueued);

        let mut second = IngestRequest::new("doc-metadata-refresh", "stable content");
        second.correlation_id = Some("corr-2".to_owned());
        second.metadata = Some(Value::String("second".to_owned()));
        let second_result = runner.ingest(second).expect("second ingest should succeed");
        assert_eq!(second_result.action, IngestAction::Unchanged);
        assert!(!second_result.fast_job_enqueued);
        assert!(!second_result.quality_job_enqueued);

        let doc = runner
            .storage
            .get_document("doc-metadata-refresh")
            .expect("fetch should succeed")
            .expect("document should exist");
        let correlation_id =
            extract_correlation_id(doc.metadata.as_ref()).expect("correlation id should exist");
        assert_eq!(correlation_id, "corr-2");
        let payload = doc
            .metadata
            .as_ref()
            .and_then(|value| value.get("payload"))
            .and_then(Value::as_str)
            .expect("payload metadata should exist");
        assert_eq!(payload, "second");

        let depth = runner
            .queue
            .queue_depth()
            .expect("queue depth should succeed");
        assert_eq!(depth.pending, 1);
    }

    #[test]
    fn ingest_returns_queue_full_when_backpressured() {
        let sink = Arc::new(InMemoryVectorSink::default());
        let fast = Arc::new(StubEmbedder::new("fast-tier", 2, None, 1.0));
        let runner = make_runner(
            JobQueueConfig {
                backpressure_threshold: 0,
                ..JobQueueConfig::default()
            },
            PipelineConfig::default(),
            fast,
            None,
            sink,
        );

        let _ = runner
            .ingest(IngestRequest::new("doc-first", "first"))
            .expect("first ingest should seed pending queue");
        let error = runner
            .ingest(IngestRequest::new("doc-second", "second"))
            .expect_err("second ingest should trip queue backpressure");

        match error {
            SearchError::QueueFull { pending, capacity } => {
                assert!(pending >= 1, "expected non-zero pending jobs");
                assert_eq!(capacity, 0);
            }
            other => panic!("expected QueueFull error, received {other:?}"),
        }
    }

    #[test]
    fn ingest_allows_new_docs_when_only_delayed_pending_jobs_exist() {
        let sink = Arc::new(InMemoryVectorSink::default());
        let fast = Arc::new(StubEmbedder::new("fast-tier", 2, None, 1.0));
        let runner = make_runner(
            JobQueueConfig {
                backpressure_threshold: 0,
                retry_base_delay_ms: 60_000,
                ..JobQueueConfig::default()
            },
            PipelineConfig::default(),
            fast,
            None,
            sink,
        );

        let _ = runner
            .ingest(IngestRequest::new("doc-delayed", "first"))
            .expect("first ingest should seed queue");
        let claimed = runner
            .queue
            .claim_batch("worker-delayed", 1)
            .expect("claim should succeed");
        assert_eq!(claimed.len(), 1);

        let fail_result = runner
            .queue
            .fail(claimed[0].job_id, "transient")
            .expect("fail should schedule retry");
        assert!(
            matches!(fail_result, FailResult::Retried { .. }),
            "first failure should schedule a retry, got {fail_result:?}"
        );

        let depth = runner
            .queue
            .queue_depth()
            .expect("queue depth should succeed");
        assert_eq!(depth.pending, 1, "one delayed pending retry should exist");
        assert_eq!(
            depth.ready_pending, 0,
            "delayed pending retry must not count as ready pending"
        );

        let result = runner
            .ingest(IngestRequest::new("doc-fresh", "second"))
            .expect("ingest should not be blocked by delayed-only pending jobs");
        assert_eq!(result.action, IngestAction::New);
        assert!(result.fast_job_enqueued);
    }

    #[test]
    fn ingest_emits_document_ingested_log_with_content_hash() {
        let sink = Arc::new(InMemoryVectorSink::default());
        let fast = Arc::new(StubEmbedder::new("fast-tier", 2, None, 1.0));
        let runner = make_runner(
            JobQueueConfig::default(),
            PipelineConfig::default(),
            fast,
            None,
            sink,
        );

        let text = "observability-ready payload";
        let expected_hash = ContentHasher::hash_hex(text);
        let ((), logs) = with_captured_logs(move || {
            let result = runner
                .ingest(IngestRequest::new("doc-log", text))
                .expect("ingest should succeed");
            assert_eq!(result.action, IngestAction::New);
        });

        assert!(logs.contains("document_ingested"), "logs: {logs}");
        assert!(logs.contains("doc_id=doc-log"), "logs: {logs}");
        assert!(logs.contains("action=new"), "logs: {logs}");
        assert!(logs.contains("content_hash="), "logs: {logs}");
        assert!(logs.contains(expected_hash.as_str()), "logs: {logs}");
    }

    #[test]
    fn process_batch_emits_job_completed_log_with_duration_ms() {
        let ((), logs) = with_captured_logs(|| {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let sink = Arc::new(InMemoryVectorSink::default());
                let fast = Arc::new(StubEmbedder::new("fast-tier", 2, None, 1.0));
                let runner = make_runner(
                    JobQueueConfig::default(),
                    PipelineConfig::default(),
                    fast,
                    None,
                    sink,
                );

                let _ = runner
                    .ingest(IngestRequest::new("doc-job-log", "job completion logging"))
                    .expect("ingest should succeed");
                let processed = runner
                    .process_batch(&cx, "worker-log")
                    .await
                    .expect("process_batch should succeed");
                assert_eq!(processed.jobs_completed, 1);
            });
        });

        assert!(logs.contains("job_completed"), "logs: {logs}");
        assert!(logs.contains("embedder_id=fast-tier"), "logs: {logs}");
        assert!(logs.contains("job_id="), "logs: {logs}");
        assert!(logs.contains("duration_ms="), "logs: {logs}");
    }

    #[test]
    fn run_worker_emits_crash_recovery_warn_with_recovered_job_count() {
        let ((), logs) = with_captured_logs(|| {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let sink = Arc::new(InMemoryVectorSink::default());
                let fast = Arc::new(StubEmbedder::new("fast-tier", 2, None, 1.0));
                let runner = make_runner(
                    JobQueueConfig {
                        visibility_timeout_ms: 5,
                        stale_job_threshold_ms: 5,
                        ..JobQueueConfig::default()
                    },
                    PipelineConfig {
                        process_batch_size: 1,
                        worker_idle_sleep_ms: 1,
                        worker_max_idle_cycles: Some(1),
                        ..PipelineConfig::default()
                    },
                    fast,
                    None,
                    Arc::clone(&sink),
                );

                let _ = runner
                    .ingest(IngestRequest::new("doc-recover", "recover me"))
                    .expect("ingest should succeed");
                let claimed = runner
                    .queue
                    .claim_batch("worker-pre", 1)
                    .expect("claim should succeed");
                let stale_started = unix_timestamp_ms()
                    .expect("timestamp should resolve")
                    .saturating_sub(10_000);
                let params = [
                    SqliteValue::Integer(stale_started),
                    SqliteValue::Integer(claimed[0].job_id),
                ];
                runner
                    .storage
                    .connection()
                    .execute_with_params(
                        "UPDATE embedding_jobs SET started_at = ?1 WHERE job_id = ?2;",
                        &params,
                    )
                    .expect("stale timestamp update should succeed");

                let shutdown = AtomicBool::new(false);
                let report = runner
                    .run_worker(&cx, "worker-recover", &shutdown)
                    .await
                    .expect("run_worker should succeed");
                assert_eq!(report.reclaimed_on_startup, 1);
            });
        });

        assert!(logs.contains("crash_recovery"), "logs: {logs}");
        assert!(logs.contains("worker-recover"), "logs: {logs}");
        assert!(logs.contains("recovered_job_count=1"), "logs: {logs}");
    }

    #[test]
    fn shutdown_flag_stops_worker_loop() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fast-tier", 2, None, 1.0));
            let runner = make_runner(
                JobQueueConfig::default(),
                PipelineConfig {
                    worker_idle_sleep_ms: 1,
                    worker_max_idle_cycles: None,
                    ..PipelineConfig::default()
                },
                fast,
                None,
                sink,
            );

            let shutdown = AtomicBool::new(true);
            let report = runner
                .run_worker(&cx, "worker-stop", &shutdown)
                .await
                .expect("run_worker should stop cleanly");
            assert_eq!(report.batches_processed, 0);
            assert_eq!(report.jobs_completed, 0);

            // Ensure no work was accidentally executed.
            let depth = runner
                .queue
                .queue_depth()
                .expect("queue depth should succeed");
            assert_eq!(depth.pending, 0);
            assert_eq!(depth.processing, 0);
        });
    }

    #[test]
    fn worker_respects_idle_cycle_limit() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fast-tier", 2, None, 1.0));
            let runner = make_runner(
                JobQueueConfig::default(),
                PipelineConfig {
                    worker_idle_sleep_ms: 1,
                    worker_max_idle_cycles: Some(2),
                    ..PipelineConfig::default()
                },
                fast,
                None,
                sink,
            );

            let shutdown = AtomicBool::new(false);
            let report = runner
                .run_worker(&cx, "worker-idle", &shutdown)
                .await
                .expect("run_worker should stop after idle cycles");
            assert_eq!(report.idle_cycles, 2);
        });
    }

    #[test]
    fn batch_ingest_reports_action_breakdown() {
        let sink = Arc::new(InMemoryVectorSink::default());
        let fast = Arc::new(StubEmbedder::new("fast-tier", 2, None, 1.0));
        let runner = make_runner(
            JobQueueConfig::default(),
            PipelineConfig::default(),
            fast,
            None,
            sink,
        );

        let first = IngestRequest::new("doc-a", "hello");
        let second = IngestRequest::new("doc-b", "");
        let third = IngestRequest::new("doc-a", "hello");
        let summary = runner
            .ingest_batch(&[first, second, third])
            .expect("batch ingest should succeed");

        assert_eq!(summary.requested, 3);
        assert_eq!(summary.inserted, 1);
        assert_eq!(summary.skipped, 1);
        assert_eq!(summary.unchanged, 1);
    }

    #[test]
    fn pipeline_metrics_accumulate_ingest_and_processing_counts() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fast-tier", 2, None, 1.0));
            let metrics = Arc::new(PipelineMetrics::default());
            let runner = make_runner(
                JobQueueConfig::default(),
                PipelineConfig::default(),
                fast,
                None,
                sink,
            )
            .with_metrics(Arc::clone(&metrics));

            let _ = runner
                .ingest(IngestRequest::new("doc-m", "metrics"))
                .expect("ingest should succeed");
            let _ = runner
                .process_batch(&cx, "worker-m")
                .await
                .expect("process should succeed");

            let snapshot = metrics.snapshot();
            assert_eq!(snapshot.total_ingest_calls, 1);
            assert_eq!(snapshot.total_ingest_inserted, 1);
            assert_eq!(snapshot.total_jobs_claimed, 1);
            assert_eq!(snapshot.total_jobs_completed, 1);
            assert_eq!(snapshot.total_jobs_failed, 0);
        });
    }

    #[test]
    fn hash_embedder_jobs_are_skipped_during_processing() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fnv1a-384", 2, None, 1.0));
            let runner = make_runner(
                JobQueueConfig::default(),
                PipelineConfig::default(),
                fast,
                None,
                Arc::clone(&sink),
            );

            let result = runner
                .ingest(IngestRequest::new("doc-hash", "hash tier"))
                .expect("ingest should succeed");
            assert!(!result.fast_job_enqueued, "hash tier should not enqueue");

            // Manually insert a hash-tier queue row to exercise skip path in process_batch.
            let params = [
                SqliteValue::Text("doc-hash".to_owned()),
                SqliteValue::Text("fnv1a-384".to_owned()),
                SqliteValue::Integer(1),
                SqliteValue::Integer(unix_timestamp_ms().expect("timestamp should resolve")),
                SqliteValue::Blob(ContentHasher::hash("hash tier").to_vec()),
            ];
            runner
                .storage
                .connection()
                .execute_with_params(
                    "INSERT INTO embedding_jobs (\
                        doc_id, embedder_id, priority, submitted_at, status, retry_count, max_retries, content_hash\
                     ) VALUES (?1, ?2, ?3, ?4, 'pending', 0, 1, ?5);",
                    &params,
                )
                .expect("manual queue insert should succeed");

            let processed = runner
                .process_batch(&cx, "worker-hash")
                .await
                .expect("process should succeed");
            assert_eq!(processed.jobs_skipped, 1);
            assert!(
                sink.entries().is_empty(),
                "hash tier should not persist vectors"
            );
        });
    }

    // ─── Utility function tests (bd-1vp2) ───────────────────────────────

    #[test]
    fn truncate_chars_empty_string() {
        assert_eq!(truncate_chars("", 10), "");
    }

    #[test]
    fn truncate_chars_shorter_than_limit() {
        assert_eq!(truncate_chars("hello", 10), "hello");
    }

    #[test]
    fn truncate_chars_exact_boundary() {
        assert_eq!(truncate_chars("abcde", 5), "abcde");
    }

    #[test]
    fn truncate_chars_one_over_boundary() {
        assert_eq!(truncate_chars("abcdef", 5), "abcde");
    }

    #[test]
    fn truncate_chars_multibyte_unicode() {
        // 3 chars: each 3 bytes in UTF-8
        let input = "\u{1F600}\u{1F601}\u{1F602}";
        let result = truncate_chars(input, 2);
        assert_eq!(result.chars().count(), 2);
    }

    #[test]
    fn is_hash_embedder_prefix_match() {
        assert!(is_hash_embedder("fnv1a-384"));
        assert!(is_hash_embedder("fnv1a-256"));
        assert!(is_hash_embedder("fnv1a-"));
    }

    #[test]
    fn is_hash_embedder_legacy_id() {
        assert!(is_hash_embedder("hash/fnv1a"));
    }

    #[test]
    fn is_hash_embedder_non_hash() {
        assert!(!is_hash_embedder("fast-tier"));
        assert!(!is_hash_embedder("quality-tier"));
        assert!(!is_hash_embedder(""));
        assert!(!is_hash_embedder("fnv1a"));
    }

    #[test]
    fn ensure_non_empty_rejects_empty() {
        let err = ensure_non_empty("", "doc_id").unwrap_err();
        assert!(err.to_string().contains("doc_id"));
    }

    #[test]
    fn ensure_non_empty_rejects_whitespace_only() {
        let err = ensure_non_empty("   ", "worker_id").unwrap_err();
        assert!(err.to_string().contains("worker_id"));
    }

    #[test]
    fn ensure_non_empty_accepts_valid() {
        assert!(ensure_non_empty("hello", "field").is_ok());
    }

    #[test]
    fn ingest_action_name_all_variants() {
        assert_eq!(ingest_action_name(&IngestAction::New), "new");
        assert_eq!(ingest_action_name(&IngestAction::Updated), "updated");
        assert_eq!(ingest_action_name(&IngestAction::Unchanged), "unchanged");
        assert_eq!(
            ingest_action_name(&IngestAction::Skipped {
                reason: "test".to_owned()
            }),
            "skipped"
        );
    }

    #[test]
    fn resolve_correlation_id_explicit() {
        let result = resolve_correlation_id("doc-1", Some("corr-42".to_owned()));
        assert_eq!(result, "corr-42");
    }

    #[test]
    fn resolve_correlation_id_empty_falls_back() {
        let result = resolve_correlation_id("doc-1", Some(String::new()));
        assert!(result.starts_with("ingest-doc-1-"));
    }

    #[test]
    fn resolve_correlation_id_whitespace_falls_back() {
        let result = resolve_correlation_id("doc-1", Some("   ".to_owned()));
        assert!(result.starts_with("ingest-doc-1-"));
    }

    #[test]
    fn resolve_correlation_id_none_falls_back() {
        let result = resolve_correlation_id("doc-x", None);
        assert!(result.starts_with("ingest-doc-x-"));
    }

    #[test]
    fn with_correlation_metadata_from_none() {
        let result = with_correlation_metadata(None, "corr-1");
        let obj = result.as_object().expect("should be object");
        assert_eq!(
            obj.get("correlation_id").unwrap().as_str().unwrap(),
            "corr-1"
        );
    }

    #[test]
    fn with_correlation_metadata_from_object() {
        let mut map = Map::new();
        map.insert("key".to_owned(), Value::String("value".to_owned()));
        let result = with_correlation_metadata(Some(Value::Object(map)), "corr-2");
        let obj = result.as_object().unwrap();
        assert_eq!(obj.get("key").unwrap().as_str().unwrap(), "value");
        assert_eq!(
            obj.get("correlation_id").unwrap().as_str().unwrap(),
            "corr-2"
        );
    }

    #[test]
    fn with_correlation_metadata_from_non_object_value() {
        let result = with_correlation_metadata(Some(Value::Number(42.into())), "corr-3");
        let obj = result.as_object().unwrap();
        assert_eq!(obj.get("payload").unwrap().as_u64().unwrap(), 42);
        assert_eq!(
            obj.get("correlation_id").unwrap().as_str().unwrap(),
            "corr-3"
        );
    }

    #[test]
    fn extract_correlation_id_present() {
        let mut map = Map::new();
        map.insert(
            "correlation_id".to_owned(),
            Value::String("found".to_owned()),
        );
        let value = Value::Object(map);
        assert_eq!(extract_correlation_id(Some(&value)).unwrap(), "found");
    }

    #[test]
    fn extract_correlation_id_missing_key() {
        let map = Map::new();
        let value = Value::Object(map);
        assert!(extract_correlation_id(Some(&value)).is_none());
    }

    #[test]
    fn extract_correlation_id_non_object() {
        let value = Value::String("not-an-object".to_owned());
        assert!(extract_correlation_id(Some(&value)).is_none());
    }

    #[test]
    fn extract_correlation_id_none_metadata() {
        assert!(extract_correlation_id(None).is_none());
    }

    #[test]
    fn usize_to_u64_normal() {
        assert_eq!(usize_to_u64(42), 42_u64);
        assert_eq!(usize_to_u64(0), 0_u64);
    }

    #[test]
    fn duration_as_u64_normal() {
        assert_eq!(duration_as_u64(100), 100_u64);
        assert_eq!(duration_as_u64(0), 0_u64);
    }

    #[test]
    fn duration_as_u64_overflow() {
        assert_eq!(duration_as_u64(u128::MAX), u64::MAX);
    }

    // ─── Type defaults and serde (bd-1vp2) ──────────────────────────────

    #[test]
    fn ingest_request_new_defaults() {
        let req = IngestRequest::new("d1", "hello");
        assert_eq!(req.doc_id, "d1");
        assert_eq!(req.text, "hello");
        assert!(req.source_path.is_none());
        assert!(req.metadata.is_none());
        assert!(req.correlation_id.is_none());
        assert!(req.enqueue_quality);
    }

    #[test]
    fn pipeline_config_default_values() {
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.fast_priority, 1);
        assert_eq!(cfg.quality_priority, 0);
        assert_eq!(cfg.process_batch_size, 32);
        assert_eq!(cfg.worker_idle_sleep_ms, 25);
        assert_eq!(cfg.worker_max_idle_cycles, Some(1));
    }

    #[test]
    fn batch_ingest_result_default_all_zeros() {
        let r = BatchIngestResult::default();
        assert_eq!(r.requested, 0);
        assert_eq!(r.inserted, 0);
        assert_eq!(r.updated, 0);
        assert_eq!(r.unchanged, 0);
        assert_eq!(r.skipped, 0);
        assert_eq!(r.fast_jobs_enqueued, 0);
        assert_eq!(r.quality_jobs_enqueued, 0);
    }

    #[test]
    fn batch_process_result_default_all_zeros() {
        let r = BatchProcessResult::default();
        assert_eq!(r.jobs_claimed, 0);
        assert_eq!(r.jobs_completed, 0);
        assert_eq!(r.jobs_failed, 0);
        assert_eq!(r.jobs_skipped, 0);
        assert_eq!(r.embed_time, Duration::ZERO);
        assert_eq!(r.total_time, Duration::ZERO);
    }

    #[test]
    fn worker_report_default_all_zeros() {
        let r = WorkerReport::default();
        assert_eq!(r.reclaimed_on_startup, 0);
        assert_eq!(r.batches_processed, 0);
        assert_eq!(r.jobs_completed, 0);
        assert_eq!(r.jobs_failed, 0);
        assert_eq!(r.jobs_skipped, 0);
        assert_eq!(r.idle_cycles, 0);
    }

    #[test]
    fn pipeline_metrics_snapshot_default() {
        let snap = PipelineMetricsSnapshot::default();
        assert_eq!(snap.total_ingest_calls, 0);
        assert_eq!(snap.total_ingest_inserted, 0);
        assert_eq!(snap.total_ingest_updated, 0);
        assert_eq!(snap.total_ingest_unchanged, 0);
        assert_eq!(snap.total_ingest_skipped, 0);
        assert_eq!(snap.total_jobs_claimed, 0);
        assert_eq!(snap.total_jobs_completed, 0);
        assert_eq!(snap.total_jobs_failed, 0);
        assert_eq!(snap.total_jobs_skipped, 0);
        assert_eq!(snap.total_embed_time_us, 0);
        assert_eq!(snap.total_reclaimed, 0);
    }

    #[test]
    fn pipeline_metrics_default_snapshot_is_zero() {
        let metrics = PipelineMetrics::default();
        let snap = metrics.snapshot();
        assert_eq!(snap.total_ingest_calls, 0);
        assert_eq!(snap.total_jobs_completed, 0);
        assert_eq!(snap.total_embed_time_us, 0);
    }

    #[test]
    fn in_memory_vector_sink_default_is_empty() {
        let sink = InMemoryVectorSink::default();
        assert!(sink.entries().is_empty());
    }

    #[test]
    fn in_memory_vector_sink_persist_accumulates() {
        let sink = InMemoryVectorSink::default();
        sink.persist("doc-1", "fast", &[1.0, 2.0]).unwrap();
        sink.persist("doc-2", "fast", &[3.0, 4.0]).unwrap();
        let entries = sink.entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].doc_id, "doc-1");
        assert_eq!(entries[1].doc_id, "doc-2");
    }

    #[test]
    fn ingest_request_serde_roundtrip() {
        let req = IngestRequest {
            doc_id: "d1".to_owned(),
            text: "hello".to_owned(),
            source_path: Some("/tmp/f.rs".to_owned()),
            metadata: Some(Value::Bool(true)),
            correlation_id: Some("corr-99".to_owned()),
            enqueue_quality: false,
        };
        let json = serde_json::to_string(&req).unwrap();
        let decoded: IngestRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, req);
    }

    #[test]
    fn ingest_action_serde_roundtrip_all_variants() {
        for action in [
            IngestAction::New,
            IngestAction::Updated,
            IngestAction::Unchanged,
            IngestAction::Skipped {
                reason: "empty".to_owned(),
            },
        ] {
            let json = serde_json::to_string(&action).unwrap();
            let decoded: IngestAction = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, action);
        }
    }

    #[test]
    fn pipeline_config_serde_roundtrip() {
        let cfg = PipelineConfig {
            fast_priority: 5,
            quality_priority: 3,
            process_batch_size: 64,
            worker_idle_sleep_ms: 50,
            worker_max_idle_cycles: Some(10),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let decoded: PipelineConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, cfg);
    }

    #[test]
    fn batch_process_result_serde_roundtrip() {
        let r = BatchProcessResult {
            jobs_claimed: 10,
            jobs_completed: 8,
            jobs_failed: 1,
            jobs_skipped: 1,
            embed_time: Duration::from_millis(42),
            total_time: Duration::from_millis(100),
        };
        let json = serde_json::to_string(&r).unwrap();
        let decoded: BatchProcessResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, r);
    }

    #[test]
    fn worker_report_serde_roundtrip() {
        let r = WorkerReport {
            reclaimed_on_startup: 3,
            batches_processed: 5,
            jobs_completed: 40,
            jobs_failed: 2,
            jobs_skipped: 1,
            idle_cycles: 7,
        };
        let json = serde_json::to_string(&r).unwrap();
        let decoded: WorkerReport = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, r);
    }

    // ─── Edge case tests requiring storage (bd-1vp2) ────────────────────

    #[test]
    fn ingest_empty_doc_id_rejected() {
        let sink = Arc::new(InMemoryVectorSink::default());
        let fast = Arc::new(StubEmbedder::new("fast-tier", 2, None, 1.0));
        let runner = make_runner(
            JobQueueConfig::default(),
            PipelineConfig::default(),
            fast,
            None,
            sink,
        );

        let err = runner
            .ingest(IngestRequest::new("", "content"))
            .expect_err("empty doc_id should fail");
        assert!(err.to_string().contains("doc_id"), "error: {err}");
    }

    #[test]
    fn ingest_whitespace_only_doc_id_rejected() {
        let sink = Arc::new(InMemoryVectorSink::default());
        let fast = Arc::new(StubEmbedder::new("fast-tier", 2, None, 1.0));
        let runner = make_runner(
            JobQueueConfig::default(),
            PipelineConfig::default(),
            fast,
            None,
            sink,
        );

        let err = runner
            .ingest(IngestRequest::new("   ", "content"))
            .expect_err("whitespace doc_id should fail");
        assert!(err.to_string().contains("doc_id"), "error: {err}");
    }

    #[test]
    fn ingest_whitespace_only_text_skipped() {
        let sink = Arc::new(InMemoryVectorSink::default());
        let fast = Arc::new(StubEmbedder::new("fast-tier", 2, None, 1.0));
        let runner = make_runner(
            JobQueueConfig::default(),
            PipelineConfig::default(),
            fast,
            None,
            sink,
        );

        let result = runner
            .ingest(IngestRequest::new("doc-ws", "   \t\n  "))
            .expect("whitespace text should not error");
        assert!(
            matches!(result.action, IngestAction::Skipped { .. }),
            "expected Skipped, got {:?}",
            result.action
        );
        assert!(!result.fast_job_enqueued);
        assert!(!result.quality_job_enqueued);
    }

    #[test]
    fn process_batch_empty_worker_id_rejected() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fast-tier", 2, None, 1.0));
            let runner = make_runner(
                JobQueueConfig::default(),
                PipelineConfig::default(),
                fast,
                None,
                sink,
            );

            let err = runner
                .process_batch(&cx, "")
                .await
                .expect_err("empty worker_id should fail");
            assert!(err.to_string().contains("worker_id"), "error: {err}");
        });
    }

    #[test]
    fn run_worker_empty_worker_id_rejected() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fast-tier", 2, None, 1.0));
            let runner = make_runner(
                JobQueueConfig::default(),
                PipelineConfig::default(),
                fast,
                None,
                sink,
            );

            let shutdown = AtomicBool::new(false);
            let err = runner
                .run_worker(&cx, "", &shutdown)
                .await
                .expect_err("empty worker_id should fail");
            assert!(err.to_string().contains("worker_id"), "error: {err}");
        });
    }

    #[test]
    fn process_batch_no_jobs_returns_zero_claimed() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fast-tier", 2, None, 1.0));
            let runner = make_runner(
                JobQueueConfig::default(),
                PipelineConfig::default(),
                fast,
                None,
                sink,
            );

            let result = runner
                .process_batch(&cx, "idle-worker")
                .await
                .expect("empty batch should succeed");
            assert_eq!(result.jobs_claimed, 0);
            assert_eq!(result.jobs_completed, 0);
            assert_eq!(result.jobs_failed, 0);
            assert_eq!(result.jobs_skipped, 0);
        });
    }

    #[test]
    fn ingest_without_quality_embedder_skips_quality_job() {
        let sink = Arc::new(InMemoryVectorSink::default());
        let fast = Arc::new(StubEmbedder::new("fast-tier", 2, None, 1.0));
        let runner = make_runner(
            JobQueueConfig::default(),
            PipelineConfig::default(),
            fast,
            None, // no quality embedder
            sink,
        );

        let result = runner
            .ingest(IngestRequest::new("doc-nq", "no quality"))
            .expect("ingest should succeed");
        assert!(result.fast_job_enqueued);
        assert!(!result.quality_job_enqueued);
    }

    #[test]
    fn ingest_with_enqueue_quality_false_skips_quality() {
        let sink = Arc::new(InMemoryVectorSink::default());
        let fast = Arc::new(StubEmbedder::new("fast-tier", 2, None, 1.0));
        let quality = Arc::new(StubEmbedder::new("quality-tier", 4, None, 2.0));
        let runner = make_runner(
            JobQueueConfig::default(),
            PipelineConfig::default(),
            fast,
            Some(quality),
            sink,
        );

        let mut req = IngestRequest::new("doc-noq", "skip quality");
        req.enqueue_quality = false;
        let result = runner.ingest(req).expect("ingest should succeed");
        assert!(result.fast_job_enqueued);
        assert!(!result.quality_job_enqueued);
    }

    #[test]
    fn runner_config_accessor_returns_configured_values() {
        let sink = Arc::new(InMemoryVectorSink::default());
        let fast = Arc::new(StubEmbedder::new("fast-tier", 2, None, 1.0));
        let custom_config = PipelineConfig {
            fast_priority: 10,
            quality_priority: 5,
            process_batch_size: 64,
            worker_idle_sleep_ms: 100,
            worker_max_idle_cycles: Some(99),
        };
        let runner = make_runner(JobQueueConfig::default(), custom_config, fast, None, sink);

        let cfg = runner.config();
        assert_eq!(cfg.fast_priority, 10);
        assert_eq!(cfg.quality_priority, 5);
        assert_eq!(cfg.process_batch_size, 64);
        assert_eq!(cfg.worker_idle_sleep_ms, 100);
        assert_eq!(cfg.worker_max_idle_cycles, Some(99));
    }

    #[test]
    fn runner_debug_shows_embedder_ids() {
        let sink = Arc::new(InMemoryVectorSink::default());
        let fast = Arc::new(StubEmbedder::new("fast-tier", 2, None, 1.0));
        let quality = Arc::new(StubEmbedder::new("quality-tier", 4, None, 2.0));
        let runner = make_runner(
            JobQueueConfig::default(),
            PipelineConfig::default(),
            fast,
            Some(quality),
            sink,
        );

        let debug = format!("{runner:?}");
        assert!(debug.contains("fast-tier"), "debug: {debug}");
        assert!(debug.contains("quality-tier"), "debug: {debug}");
    }

    #[test]
    fn fallback_correlation_id_contains_doc_id() {
        let id = fallback_correlation_id("my-doc");
        assert!(id.starts_with("ingest-my-doc-"));
    }

    #[test]
    fn pipeline_error_creates_subsystem_error() {
        let err = pipeline_error("test message");
        match err {
            SearchError::SubsystemError { subsystem, .. } => {
                assert_eq!(subsystem, "storage_pipeline");
            }
            other => panic!("expected SubsystemError, got {other:?}"),
        }
    }

    #[test]
    fn persisted_embedding_clone_and_eq() {
        let e = PersistedEmbedding {
            doc_id: "d".to_owned(),
            embedder_id: "e".to_owned(),
            embedding: vec![1.0, 2.0],
        };
        let e2 = e.clone();
        assert_eq!(e, e2);
    }

    #[test]
    fn ingest_result_clone_and_serde() {
        let r = IngestResult {
            doc_id: "d".to_owned(),
            action: IngestAction::New,
            fast_job_enqueued: true,
            quality_job_enqueued: false,
            correlation_id: "c".to_owned(),
        };
        let json = serde_json::to_string(&r).unwrap();
        let decoded: IngestResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, r);
    }

    #[test]
    fn ingest_same_as_fast_embedder_id_skips_quality() {
        // When quality embedder has the SAME id as fast, quality job should not be enqueued
        let sink = Arc::new(InMemoryVectorSink::default());
        let fast = Arc::new(StubEmbedder::new("shared-tier", 2, None, 1.0));
        let quality = Arc::new(StubEmbedder::new("shared-tier", 4, None, 2.0));
        let runner = make_runner(
            JobQueueConfig::default(),
            PipelineConfig::default(),
            fast,
            Some(quality),
            sink,
        );

        let result = runner
            .ingest(IngestRequest::new("doc-dup-tier", "same tier"))
            .expect("ingest should succeed");
        assert!(result.fast_job_enqueued);
        assert!(
            !result.quality_job_enqueued,
            "quality should be skipped when same as fast"
        );
    }
}

// ─── Integration tests (bd-3w1.17) ─────────────────────────────────────────
//
// These tests verify cross-component interactions within the storage pipeline:
// document ingestion, embedding job processing, staleness detection, crash
// recovery, content change detection, and metrics accumulation.
#[cfg(test)]
mod integration_tests {
    #![allow(clippy::arc_with_non_send_sync)]

    use std::collections::HashSet;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use frankensearch_core::canonicalize::DefaultCanonicalizer;
    use frankensearch_core::traits::{ModelCategory, SearchFuture};
    use frankensearch_core::{Canonicalizer, Embedder, SearchError};
    use fsqlite_types::value::SqliteValue;

    use crate::connection::Storage;
    use crate::index_metadata::{BuildTrigger, RecordBuildParams};
    use crate::job_queue::{JobQueueConfig, PersistentJobQueue};
    use crate::schema::SCHEMA_VERSION;
    use crate::staleness::{
        RecommendedAction, StalenessConfig, StalenessLevel, StorageBackedStaleness,
    };

    use super::*;

    // ── Shared test helpers ─────────────────────────────────────────────

    #[derive(Debug)]
    struct StubEmbedder {
        id: &'static str,
        dim: usize,
        fail_on_substring: Option<&'static str>,
        fill: f32,
    }

    impl StubEmbedder {
        const fn new(
            id: &'static str,
            dim: usize,
            fail_on_substring: Option<&'static str>,
            fill: f32,
        ) -> Self {
            Self {
                id,
                dim,
                fail_on_substring,
                fill,
            }
        }
    }

    impl Embedder for StubEmbedder {
        fn embed<'a>(
            &'a self,
            _cx: &'a asupersync::Cx,
            text: &'a str,
        ) -> SearchFuture<'a, Vec<f32>> {
            let should_fail = self
                .fail_on_substring
                .is_some_and(|needle| text.contains(needle));
            let dim = self.dim;
            let fill = self.fill;
            let model_id = self.id.to_owned();
            Box::pin(async move {
                if should_fail {
                    return Err(SearchError::EmbeddingFailed {
                        model: model_id,
                        source: Box::new(std::io::Error::other("stub embedder failure")),
                    });
                }
                Ok(vec![fill; dim])
            })
        }

        fn dimension(&self) -> usize {
            self.dim
        }

        fn id(&self) -> &str {
            self.id
        }

        fn model_name(&self) -> &str {
            self.id
        }

        fn is_semantic(&self) -> bool {
            !is_hash_embedder(self.id)
        }

        fn category(&self) -> ModelCategory {
            if is_hash_embedder(self.id) {
                ModelCategory::HashEmbedder
            } else {
                ModelCategory::StaticEmbedder
            }
        }
    }

    fn make_runner(
        queue_config: JobQueueConfig,
        pipeline_config: PipelineConfig,
        fast_embedder: Arc<dyn Embedder>,
        quality_embedder: Option<Arc<dyn Embedder>>,
        sink: Arc<InMemoryVectorSink>,
    ) -> StorageBackedJobRunner {
        let storage = Arc::new(Storage::open_in_memory().expect("storage should open"));
        let queue = Arc::new(PersistentJobQueue::new(Arc::clone(&storage), queue_config));
        let canonicalizer: Arc<dyn Canonicalizer> = Arc::new(DefaultCanonicalizer::default());
        let mut runner =
            StorageBackedJobRunner::new(storage, queue, canonicalizer, fast_embedder, sink)
                .with_config(pipeline_config);
        if let Some(quality_embedder) = quality_embedder {
            runner = runner.with_quality_embedder(quality_embedder);
        }
        runner
    }

    fn make_runner_with_storage(
        storage: Arc<Storage>,
        queue_config: JobQueueConfig,
        pipeline_config: PipelineConfig,
        fast_embedder: Arc<dyn Embedder>,
        quality_embedder: Option<Arc<dyn Embedder>>,
        sink: Arc<InMemoryVectorSink>,
    ) -> StorageBackedJobRunner {
        let queue = Arc::new(PersistentJobQueue::new(Arc::clone(&storage), queue_config));
        let canonicalizer: Arc<dyn Canonicalizer> = Arc::new(DefaultCanonicalizer::default());
        let mut runner =
            StorageBackedJobRunner::new(storage, queue, canonicalizer, fast_embedder, sink)
                .with_config(pipeline_config);
        if let Some(quality_embedder) = quality_embedder {
            runner = runner.with_quality_embedder(quality_embedder);
        }
        runner
    }

    fn sample_build_params(name: &str, embedder_id: &str, doc_count: i64) -> RecordBuildParams {
        RecordBuildParams {
            index_name: name.to_owned(),
            index_type: "fsvi".to_owned(),
            embedder_id: embedder_id.to_owned(),
            embedder_revision: Some("v1.0".to_owned()),
            dimension: 256,
            record_count: doc_count,
            source_doc_count: doc_count,
            build_duration_ms: 42,
            trigger: BuildTrigger::Initial,
            file_path: None,
            file_size_bytes: None,
            file_hash: None,
            schema_version: Some(SCHEMA_VERSION),
            config_json: None,
            fec_path: None,
            fec_size_bytes: None,
            notes: None,
            mean_norm: None,
            variance: None,
        }
    }

    // ── Test 1: Full pipeline (ingest 100 docs → embed → verify metrics) ──

    #[test]
    fn full_pipeline_ingest_process_and_verify_metrics() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fast-tier", 4, None, 1.0));
            let quality = Arc::new(StubEmbedder::new("quality-tier", 4, None, 2.0));
            let metrics = Arc::new(PipelineMetrics::default());
            let runner = make_runner(
                JobQueueConfig::default(),
                PipelineConfig {
                    process_batch_size: 50,
                    worker_max_idle_cycles: Some(1),
                    worker_idle_sleep_ms: 1,
                    ..PipelineConfig::default()
                },
                fast,
                Some(quality),
                Arc::clone(&sink),
            )
            .with_metrics(Arc::clone(&metrics));

            // Ingest 100 documents.
            for i in 0..100 {
                let request = IngestRequest::new(
                    format!("doc-{i}"),
                    format!("Document number {i} about Rust ownership and borrowing"),
                );
                let result = runner.ingest(request).expect("ingest should succeed");
                assert_eq!(result.action, IngestAction::New);
                assert!(result.fast_job_enqueued);
                assert!(result.quality_job_enqueued);
            }

            // Verify queue depth: 200 jobs (100 fast + 100 quality).
            let depth = runner.queue.queue_depth().expect("queue depth");
            assert_eq!(depth.pending, 200);

            // Process all jobs via worker loop.
            let shutdown = AtomicBool::new(false);
            let report = runner
                .run_worker(&cx, "integration-worker", &shutdown)
                .await
                .expect("worker should succeed");

            assert_eq!(report.jobs_completed, 200);
            assert_eq!(report.jobs_failed, 0);

            // Verify embeddings persisted: 100 fast + 100 quality = 200.
            let entries = sink.entries();
            assert_eq!(entries.len(), 200);

            let fast_entries: Vec<_> = entries
                .iter()
                .filter(|e| e.embedder_id == "fast-tier")
                .collect();
            assert_eq!(fast_entries.len(), 100);
            assert_eq!(
                entries
                    .iter()
                    .filter(|e| e.embedder_id == "quality-tier")
                    .count(),
                100
            );

            // Verify unique doc_ids across fast tier.
            let fast_doc_ids: HashSet<_> = fast_entries.iter().map(|e| e.doc_id.clone()).collect();
            assert_eq!(fast_doc_ids.len(), 100);

            // Verify metrics counters.
            let snap = metrics.snapshot();
            assert_eq!(snap.total_ingest_calls, 100);
            assert_eq!(snap.total_ingest_inserted, 100);
            assert_eq!(snap.total_jobs_claimed, 200);
            assert_eq!(snap.total_jobs_completed, 200);
            assert_eq!(snap.total_jobs_failed, 0);

            // Verify embedding status in storage.
            let fast_counts = runner.storage.count_by_status("fast-tier").expect("counts");
            assert_eq!(fast_counts.embedded, 100);
            assert_eq!(fast_counts.pending, 0);

            let quality_counts = runner
                .storage
                .count_by_status("quality-tier")
                .expect("counts");
            assert_eq!(quality_counts.embedded, 100);

            // Queue should be completely drained.
            let final_depth = runner.queue.queue_depth().expect("final depth");
            assert_eq!(final_depth.pending, 0);
            assert_eq!(final_depth.processing, 0);
            assert_eq!(final_depth.completed, 200);
        });
    }

    // ── Test 2: Incremental update with staleness detection ────────────

    #[test]
    fn incremental_update_detects_new_documents_as_stale() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let storage = Arc::new(Storage::open_in_memory().expect("storage"));
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fast-tier", 4, None, 1.0));
            let runner = make_runner_with_storage(
                Arc::clone(&storage),
                JobQueueConfig::default(),
                PipelineConfig {
                    process_batch_size: 100,
                    worker_max_idle_cycles: Some(1),
                    worker_idle_sleep_ms: 1,
                    ..PipelineConfig::default()
                },
                fast,
                None,
                Arc::clone(&sink),
            );

            // Phase 1: Ingest and process initial 50 documents.
            for i in 0..50 {
                runner
                    .ingest(IngestRequest::new(
                        format!("doc-{i}"),
                        format!("Initial content for document {i}"),
                    ))
                    .expect("ingest");
            }
            let shutdown = AtomicBool::new(false);
            let report = runner
                .run_worker(&cx, "worker-phase1", &shutdown)
                .await
                .expect("phase1 worker");
            assert_eq!(report.jobs_completed, 50);

            // Record an index build so staleness has a baseline.
            storage
                .record_index_build(&sample_build_params("idx-fast", "fast-tier", 50))
                .expect("record build");

            // Staleness should be None right after build (no new docs).
            let detector = StorageBackedStaleness::with_defaults(Arc::clone(&storage));
            let report = detector.check("idx-fast", Some("v1.0")).expect("check");
            assert!(!report.is_stale);
            assert_eq!(report.level, StalenessLevel::None);

            // Phase 2: Ingest 20 more documents.
            for i in 50..70 {
                runner
                    .ingest(IngestRequest::new(
                        format!("doc-{i}"),
                        format!("New content for document {i}"),
                    ))
                    .expect("ingest");
            }

            // Staleness should now detect new documents.
            let report = detector.check("idx-fast", Some("v1.0")).expect("check");
            assert!(report.is_stale);
            assert!(report.stats.docs_changed_since_build >= 20);
            // 20 new docs / 50 baseline = 40% > 30% threshold → FullRebuild.
            assert!(
                matches!(
                    &report.recommended_action,
                    RecommendedAction::FullRebuild { .. }
                        | RecommendedAction::IncrementalUpdate { .. }
                ),
                "expected rebuild or incremental, got {:?}",
                report.recommended_action
            );

            // Process incremental embedding jobs.
            let shutdown2 = AtomicBool::new(false);
            let report = runner
                .run_worker(&cx, "worker-phase2", &shutdown2)
                .await
                .expect("phase2 worker");
            assert_eq!(report.jobs_completed, 20);

            // All 70 documents should now have embeddings.
            assert_eq!(sink.entries().len(), 70);
            let doc_ids: HashSet<_> = sink.entries().iter().map(|e| e.doc_id.clone()).collect();
            assert_eq!(doc_ids.len(), 70);
        });
    }

    // ── Test 3: Content change detection ───────────────────────────────

    #[test]
    fn content_change_detection_re_embeds_updated_documents() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fast-tier", 4, None, 1.0));
            let runner = make_runner(
                JobQueueConfig::default(),
                PipelineConfig {
                    process_batch_size: 100,
                    worker_max_idle_cycles: Some(1),
                    worker_idle_sleep_ms: 1,
                    ..PipelineConfig::default()
                },
                fast,
                None,
                Arc::clone(&sink),
            );

            // Phase 1: Ingest and process 50 documents.
            for i in 0..50 {
                runner
                    .ingest(IngestRequest::new(
                        format!("doc-{i}"),
                        format!("Original content for document {i}"),
                    ))
                    .expect("ingest");
            }
            let shutdown = AtomicBool::new(false);
            let report = runner
                .run_worker(&cx, "worker-phase1", &shutdown)
                .await
                .expect("phase1");
            assert_eq!(report.jobs_completed, 50);

            // Phase 2: Update 10 documents with new content (same doc_ids).
            let mut updated_count = 0;
            for i in 0..10 {
                let result = runner
                    .ingest(IngestRequest::new(
                        format!("doc-{i}"),
                        format!("UPDATED content for document {i} with totally new text"),
                    ))
                    .expect("re-ingest");
                assert_eq!(
                    result.action,
                    IngestAction::Updated,
                    "doc-{i} should be detected as Updated"
                );
                assert!(result.fast_job_enqueued);
                updated_count += 1;
            }
            assert_eq!(updated_count, 10);

            // Re-ingesting same unchanged documents should detect no change.
            for i in 10..20 {
                let result = runner
                    .ingest(IngestRequest::new(
                        format!("doc-{i}"),
                        format!("Original content for document {i}"),
                    ))
                    .expect("unchanged ingest");
                assert_eq!(result.action, IngestAction::Unchanged);
                assert!(!result.fast_job_enqueued);
            }

            // Process the 10 re-embedding jobs.
            let shutdown2 = AtomicBool::new(false);
            let report = runner
                .run_worker(&cx, "worker-phase2", &shutdown2)
                .await
                .expect("phase2");
            assert_eq!(report.jobs_completed, 10);

            // Verify: 60 total embeddings (50 original + 10 re-embeds).
            assert_eq!(sink.entries().len(), 60);

            // Verify the updated docs appear with the latest embedding.
            // Each updated doc should have 2 entries (original + update).
            assert_eq!(
                sink.entries()
                    .iter()
                    .filter(|e| {
                        let idx: usize = e.doc_id.strip_prefix("doc-").unwrap().parse().unwrap();
                        idx < 10
                    })
                    .count(),
                20
            );
        });
    }

    // ── Test 4: Crash recovery (stale job reclamation) ─────────────────

    #[test]
    fn crash_recovery_reclaims_stale_jobs_and_avoids_duplicates() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let storage = Arc::new(Storage::open_in_memory().expect("storage"));
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fast-tier", 4, None, 1.0));

            let queue_config = JobQueueConfig {
                visibility_timeout_ms: 5,
                stale_job_threshold_ms: 5,
                ..JobQueueConfig::default()
            };

            // Phase 1: Ingest 50 documents.
            let runner1 = make_runner_with_storage(
                Arc::clone(&storage),
                queue_config,
                PipelineConfig {
                    process_batch_size: 25,
                    worker_max_idle_cycles: Some(1),
                    worker_idle_sleep_ms: 1,
                    ..PipelineConfig::default()
                },
                Arc::clone(&fast) as Arc<dyn Embedder>,
                None,
                Arc::clone(&sink),
            );

            for i in 0..50 {
                runner1
                    .ingest(IngestRequest::new(
                        format!("doc-{i}"),
                        format!("Document content {i}"),
                    ))
                    .expect("ingest");
            }

            // Claim a batch of 25 jobs (simulating a worker that starts).
            let claimed = runner1
                .queue
                .claim_batch("crashing-worker", 25)
                .expect("claim");
            assert_eq!(claimed.len(), 25);

            // "Crash": Force the claimed jobs to look stale by backdating started_at.
            let stale_ts = unix_timestamp_ms()
                .expect("timestamp")
                .saturating_sub(100_000);
            for job in &claimed {
                let params = [
                    SqliteValue::Integer(stale_ts),
                    SqliteValue::Integer(job.job_id),
                ];
                storage
                    .connection()
                    .execute_with_params(
                        "UPDATE embedding_jobs SET started_at = ?1 WHERE job_id = ?2;",
                        &params,
                    )
                    .expect("backdate");
            }

            // Phase 2: New runner on same storage (simulating restart).
            let sink2 = Arc::new(InMemoryVectorSink::default());
            let runner2 = make_runner_with_storage(
                Arc::clone(&storage),
                queue_config,
                PipelineConfig {
                    process_batch_size: 50,
                    worker_max_idle_cycles: Some(1),
                    worker_idle_sleep_ms: 1,
                    ..PipelineConfig::default()
                },
                fast,
                None,
                Arc::clone(&sink2),
            );

            // run_worker calls reclaim_stale_jobs() on startup.
            let shutdown = AtomicBool::new(false);
            let report = runner2
                .run_worker(&cx, "recovery-worker", &shutdown)
                .await
                .expect("recovery worker");

            // All 25 stale jobs should have been reclaimed.
            assert_eq!(report.reclaimed_on_startup, 25);

            // All 50 jobs should be completed (25 remaining pending + 25 reclaimed).
            assert_eq!(report.jobs_completed, 50);

            // Verify no duplicate embeddings: each doc_id should appear exactly once.
            let entries = sink2.entries();
            assert_eq!(entries.len(), 50);
            let doc_ids: HashSet<_> = entries.iter().map(|e| e.doc_id.clone()).collect();
            assert_eq!(doc_ids.len(), 50);

            // Queue should be fully drained.
            let depth = runner2.queue.queue_depth().expect("depth");
            assert_eq!(depth.pending, 0);
            assert_eq!(depth.processing, 0);
        });
    }

    // ── Test 5: Batch ingest with mixed actions ────────────────────────

    #[test]
    fn batch_ingest_tracks_mixed_new_unchanged_and_skipped() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fast-tier", 4, None, 1.0));
            let runner = make_runner(
                JobQueueConfig::default(),
                PipelineConfig {
                    process_batch_size: 100,
                    worker_max_idle_cycles: Some(1),
                    worker_idle_sleep_ms: 1,
                    ..PipelineConfig::default()
                },
                fast,
                None,
                Arc::clone(&sink),
            );

            // Initial batch: 20 new documents.
            let initial: Vec<_> = (0..20)
                .map(|i| IngestRequest::new(format!("doc-{i}"), format!("Content {i}")))
                .collect();
            let summary = runner.ingest_batch(&initial).expect("batch1");
            assert_eq!(summary.requested, 20);
            assert_eq!(summary.inserted, 20);
            assert_eq!(summary.unchanged, 0);
            assert_eq!(summary.skipped, 0);

            // Process all.
            let shutdown = AtomicBool::new(false);
            runner
                .run_worker(&cx, "batch-worker", &shutdown)
                .await
                .expect("worker");

            // Second batch: 10 unchanged + 5 updated + 5 empty (skipped).
            let second_batch: Vec<_> = (0..10)
                .map(|i| {
                    // These have the same content → Unchanged.
                    IngestRequest::new(format!("doc-{i}"), format!("Content {i}"))
                })
                .chain((10..15).map(|i| {
                    // Different content → Updated.
                    IngestRequest::new(format!("doc-{i}"), format!("UPDATED content {i}"))
                }))
                .chain((100..105).map(|i| {
                    // Empty text → Skipped.
                    IngestRequest::new(format!("empty-{i}"), String::new())
                }))
                .collect();

            let summary2 = runner.ingest_batch(&second_batch).expect("batch2");
            assert_eq!(summary2.requested, 20);
            assert_eq!(summary2.unchanged, 10);
            assert_eq!(summary2.updated, 5);
            assert_eq!(summary2.skipped, 5);
        });
    }

    // ── Test 6: Metrics accumulation across full ingest-process cycle ──

    #[test]
    fn metrics_accumulate_correctly_across_multiple_cycles() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fast-tier", 4, Some("fail-me"), 1.0));
            let metrics = Arc::new(PipelineMetrics::default());
            let runner = make_runner(
                JobQueueConfig {
                    max_retries: 0,
                    ..JobQueueConfig::default()
                },
                PipelineConfig {
                    process_batch_size: 100,
                    worker_max_idle_cycles: Some(1),
                    worker_idle_sleep_ms: 1,
                    ..PipelineConfig::default()
                },
                fast,
                None,
                Arc::clone(&sink),
            )
            .with_metrics(Arc::clone(&metrics));

            // Cycle 1: 8 good documents + 2 that fail embedding.
            for i in 0..8 {
                runner
                    .ingest(IngestRequest::new(
                        format!("good-{i}"),
                        format!("Safe text {i}"),
                    ))
                    .expect("ingest good");
            }
            for i in 0..2 {
                runner
                    .ingest(IngestRequest::new(
                        format!("bad-{i}"),
                        format!("This will fail-me {i}"),
                    ))
                    .expect("ingest bad");
            }
            // Also ingest 3 empty docs (skipped at ingest time).
            for i in 0..3 {
                let result = runner
                    .ingest(IngestRequest::new(format!("empty-{i}"), String::new()))
                    .expect("ingest empty");
                assert!(matches!(result.action, IngestAction::Skipped { .. }));
            }

            let shutdown = AtomicBool::new(false);
            let report = runner
                .run_worker(&cx, "metrics-worker", &shutdown)
                .await
                .expect("worker");
            assert_eq!(report.jobs_completed, 8);
            assert_eq!(report.jobs_failed, 2);

            let snap = metrics.snapshot();
            assert_eq!(snap.total_ingest_calls, 13);
            assert_eq!(snap.total_ingest_inserted, 10);
            assert_eq!(snap.total_ingest_skipped, 3);
            assert_eq!(snap.total_jobs_claimed, 10);
            assert_eq!(snap.total_jobs_completed, 8);
            assert_eq!(snap.total_jobs_failed, 2);

            // Storage metrics.
            let storage_snap = runner.storage.metrics_snapshot();
            assert!(
                storage_snap.tx_commits > 0,
                "should have committed transactions"
            );
        });
    }

    // ── Test 7: Staleness levels and recommended actions ───────────────

    #[test]
    fn staleness_integration_across_lifecycle() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let storage = Arc::new(Storage::open_in_memory().expect("storage"));
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fast-tier", 4, None, 1.0));
            let runner = make_runner_with_storage(
                Arc::clone(&storage),
                JobQueueConfig::default(),
                PipelineConfig {
                    process_batch_size: 200,
                    worker_max_idle_cycles: Some(1),
                    worker_idle_sleep_ms: 1,
                    ..PipelineConfig::default()
                },
                fast,
                None,
                Arc::clone(&sink),
            );

            let detector = StorageBackedStaleness::new(
                Arc::clone(&storage),
                StalenessConfig {
                    min_change_threshold: 5,
                    ..StalenessConfig::default()
                },
            );

            // Before any build: NeverBuilt → Critical.
            let report = detector.check("idx", None).expect("check");
            assert_eq!(report.level, StalenessLevel::Critical);
            assert!(matches!(
                report.recommended_action,
                RecommendedAction::FullRebuild { .. }
            ));

            // Ingest and process 50 documents.
            for i in 0..50 {
                runner
                    .ingest(IngestRequest::new(
                        format!("doc-{i}"),
                        format!("Lifecycle test content {i}"),
                    ))
                    .expect("ingest");
            }
            let shutdown = AtomicBool::new(false);
            runner
                .run_worker(&cx, "lifecycle-worker", &shutdown)
                .await
                .expect("worker");

            // Record build.
            storage
                .record_index_build(&sample_build_params("idx", "fast-tier", 50))
                .expect("build");

            // After build: should be fresh.
            let report = detector.check("idx", Some("v1.0")).expect("check");
            assert!(!report.is_stale);
            assert_eq!(report.level, StalenessLevel::None);

            // Add 3 docs (below threshold of 5) → Minor.
            for i in 50..53 {
                runner
                    .ingest(IngestRequest::new(
                        format!("doc-{i}"),
                        format!("New content {i}"),
                    ))
                    .expect("ingest");
            }
            let report = detector.check("idx", Some("v1.0")).expect("check");
            assert!(report.is_stale);
            assert_eq!(report.level, StalenessLevel::Minor);

            // Add 7 more (total 10 new, above threshold) → Significant.
            for i in 53..60 {
                runner
                    .ingest(IngestRequest::new(
                        format!("doc-{i}"),
                        format!("New content {i}"),
                    ))
                    .expect("ingest");
            }
            let report = detector.check("idx", Some("v1.0")).expect("check");
            assert!(report.is_stale);
            assert_eq!(report.level, StalenessLevel::Significant);
            assert!(matches!(
                report.recommended_action,
                RecommendedAction::IncrementalUpdate { doc_count: 10 }
            ));

            // Change embedder revision → Critical.
            let report = detector.check("idx", Some("v2.0")).expect("check");
            assert_eq!(report.level, StalenessLevel::Critical);
            assert!(matches!(
                report.recommended_action,
                RecommendedAction::FullRebuild { .. }
            ));
        });
    }

    // ── Test 8: Two-tier pipeline with fast + quality embedders ────────

    #[test]
    fn two_tier_pipeline_processes_both_tiers_independently() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fast-tier", 256, None, 0.5));
            let quality = Arc::new(StubEmbedder::new("quality-tier", 384, None, 0.9));
            let metrics = Arc::new(PipelineMetrics::default());
            let runner = make_runner(
                JobQueueConfig::default(),
                PipelineConfig {
                    process_batch_size: 200,
                    worker_max_idle_cycles: Some(1),
                    worker_idle_sleep_ms: 1,
                    ..PipelineConfig::default()
                },
                fast,
                Some(quality),
                Arc::clone(&sink),
            )
            .with_metrics(Arc::clone(&metrics));

            // Ingest 30 documents.
            for i in 0..30 {
                let result = runner
                    .ingest(IngestRequest::new(
                        format!("doc-{i}"),
                        format!("Two-tier test content for document number {i}"),
                    ))
                    .expect("ingest");
                assert!(result.fast_job_enqueued);
                assert!(result.quality_job_enqueued);
            }

            // 60 total jobs (30 fast + 30 quality).
            let depth = runner.queue.queue_depth().expect("depth");
            assert_eq!(depth.pending, 60);

            // Process all.
            let shutdown = AtomicBool::new(false);
            let report = runner
                .run_worker(&cx, "two-tier-worker", &shutdown)
                .await
                .expect("worker");
            assert_eq!(report.jobs_completed, 60);

            // Verify embeddings by tier.
            let entries = sink.entries();
            assert_eq!(entries.len(), 60);

            let fast_entries: Vec<_> = entries
                .iter()
                .filter(|e| e.embedder_id == "fast-tier")
                .collect();
            let quality_entries: Vec<_> = entries
                .iter()
                .filter(|e| e.embedder_id == "quality-tier")
                .collect();
            assert_eq!(fast_entries.len(), 30);
            assert_eq!(quality_entries.len(), 30);

            // Verify dimensionality.
            for e in &fast_entries {
                assert_eq!(e.embedding.len(), 256);
                assert!((e.embedding[0] - 0.5).abs() < f32::EPSILON);
            }
            for e in &quality_entries {
                assert_eq!(e.embedding.len(), 384);
                assert!((e.embedding[0] - 0.9).abs() < f32::EPSILON);
            }

            // Verify both tiers show as embedded in storage.
            let fast_counts = runner.storage.count_by_status("fast-tier").expect("counts");
            let quality_counts = runner
                .storage
                .count_by_status("quality-tier")
                .expect("counts");
            assert_eq!(fast_counts.embedded, 30);
            assert_eq!(quality_counts.embedded, 30);
        });
    }

    // ── Test 9: Quick staleness check integration ──────────────────────

    #[test]
    fn quick_staleness_check_tracks_embedding_progress() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let storage = Arc::new(Storage::open_in_memory().expect("storage"));
            let sink = Arc::new(InMemoryVectorSink::default());
            let fast = Arc::new(StubEmbedder::new("fast-tier", 4, None, 1.0));
            let runner = make_runner_with_storage(
                Arc::clone(&storage),
                JobQueueConfig::default(),
                PipelineConfig {
                    process_batch_size: 10,
                    worker_max_idle_cycles: Some(1),
                    worker_idle_sleep_ms: 1,
                    ..PipelineConfig::default()
                },
                fast,
                None,
                Arc::clone(&sink),
            );

            let detector = StorageBackedStaleness::with_defaults(Arc::clone(&storage));

            // Initially no documents: not stale.
            let quick = detector.quick_check("fast-tier").expect("quick");
            assert!(!quick.is_stale);
            assert_eq!(quick.pending_count, 0);

            // Ingest 20 documents. Quick check should detect pending.
            for i in 0..20 {
                runner
                    .ingest(IngestRequest::new(
                        format!("doc-{i}"),
                        format!("Quick check content {i}"),
                    ))
                    .expect("ingest");
            }

            let quick = detector.quick_check("fast-tier").expect("quick");
            assert!(quick.is_stale);
            assert!(quick.pending_count > 0);

            // Process first batch of 10.
            runner
                .process_batch(&cx, "quick-worker")
                .await
                .expect("batch1");

            // Still stale (10 remaining).
            let quick = detector.quick_check("fast-tier").expect("quick");
            assert!(quick.is_stale);

            // Process second batch.
            runner
                .process_batch(&cx, "quick-worker")
                .await
                .expect("batch2");

            // All embedded: not stale.
            let quick = detector.quick_check("fast-tier").expect("quick");
            assert!(!quick.is_stale);
            assert_eq!(quick.pending_count, 0);
        });
    }

    // ── Test 10: Worker partial failure does not block good jobs ────────

    #[test]
    fn partial_failures_do_not_block_successful_jobs() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let sink = Arc::new(InMemoryVectorSink::default());
            // This embedder fails on text containing "poison".
            let fast = Arc::new(StubEmbedder::new("fast-tier", 4, Some("poison"), 1.0));
            let metrics = Arc::new(PipelineMetrics::default());
            let runner = make_runner(
                JobQueueConfig {
                    max_retries: 0,
                    ..JobQueueConfig::default()
                },
                PipelineConfig {
                    process_batch_size: 100,
                    worker_max_idle_cycles: Some(1),
                    worker_idle_sleep_ms: 1,
                    ..PipelineConfig::default()
                },
                fast,
                None,
                Arc::clone(&sink),
            )
            .with_metrics(Arc::clone(&metrics));

            // Ingest a mix of good and bad documents.
            for i in 0..15 {
                runner
                    .ingest(IngestRequest::new(
                        format!("good-{i}"),
                        format!("Perfectly safe content {i}"),
                    ))
                    .expect("ingest good");
            }
            for i in 0..5 {
                runner
                    .ingest(IngestRequest::new(
                        format!("bad-{i}"),
                        format!("This contains poison text {i}"),
                    ))
                    .expect("ingest bad");
            }

            let shutdown = AtomicBool::new(false);
            let report = runner
                .run_worker(&cx, "resilience-worker", &shutdown)
                .await
                .expect("worker");

            // 15 good jobs succeed, 5 fail.
            assert_eq!(report.jobs_completed, 15);
            assert_eq!(report.jobs_failed, 5);

            // Only good embeddings persisted.
            assert_eq!(sink.entries().len(), 15);

            // Failed jobs tracked in queue.
            let depth = runner.queue.queue_depth().expect("depth");
            assert_eq!(depth.failed, 5);
            assert_eq!(depth.completed, 15);

            // Metrics consistent.
            let snap = metrics.snapshot();
            assert_eq!(snap.total_jobs_completed, 15);
            assert_eq!(snap.total_jobs_failed, 5);

            // Storage tracks failed status.
            let counts = runner.storage.count_by_status("fast-tier").expect("counts");
            assert_eq!(counts.embedded, 15);
            assert_eq!(counts.failed, 5);
        });
    }

    #[test]
    fn process_batch_reads_source_file_content_if_available() {
        use std::io::Write;
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            // Create a SpyEmbedder that records the text it sees.
            #[derive(Debug)]
            struct SpyEmbedder {
                captured: Arc<Mutex<Option<String>>>,
            }
            impl Embedder for SpyEmbedder {
                fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
                    *self.captured.lock().unwrap() = Some(text.to_owned());
                    Box::pin(async { Ok(vec![0.0; 4]) })
                }
                fn dimension(&self) -> usize {
                    4
                }
                fn id(&self) -> &'static str {
                    "spy"
                }
                fn model_name(&self) -> &'static str {
                    "spy"
                }
                fn is_semantic(&self) -> bool {
                    true
                }
                fn category(&self) -> ModelCategory {
                    ModelCategory::StaticEmbedder
                }
            }
            let sink = Arc::new(InMemoryVectorSink::default());
            let captured = Arc::new(Mutex::new(None));
            let fast = Arc::new(SpyEmbedder {
                captured: Arc::clone(&captured),
            });

            let runner = make_runner(
                JobQueueConfig::default(),
                PipelineConfig::default(),
                fast,
                None,
                sink,
            );

            // Create a temp file with content longer than preview limit (400 chars).
            let long_content = "A".repeat(1000);
            let mut temp = tempfile::NamedTempFile::new().expect("tempfile");
            temp.write_all(long_content.as_bytes()).expect("write");
            let path = temp.path().to_string_lossy().into_owned();

            // Ingest with source_path.
            let mut req = IngestRequest::new("doc-file", &long_content);
            req.source_path = Some(path);

            let result = runner.ingest(req).expect("ingest");
            assert_eq!(result.action, IngestAction::New);

            // Check that the preview is truncated in storage.
            let doc = runner
                .storage
                .get_document("doc-file")
                .expect("get")
                .expect("exists");
            assert_eq!(doc.content_preview.len(), 400);

            // Process batch.
            let _ = runner
                .process_batch(&cx, "worker-spy")
                .await
                .expect("process");

            // Verify embedder received full content.
            let captured_text = captured
                .lock()
                .unwrap()
                .take()
                .expect("should have captured text");
            assert_eq!(
                captured_text.len(),
                1000,
                "embedder should see full content"
            );
            assert_eq!(captured_text, long_content);
        });
    }

    #[test]
    fn process_batch_falls_back_to_preview_when_source_file_is_too_large() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            #[derive(Debug)]
            struct SpyEmbedder {
                captured: Arc<Mutex<Option<String>>>,
            }
            impl Embedder for SpyEmbedder {
                fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
                    *self.captured.lock().unwrap() = Some(text.to_owned());
                    Box::pin(async { Ok(vec![0.0; 4]) })
                }
                fn dimension(&self) -> usize {
                    4
                }
                fn id(&self) -> &'static str {
                    "spy"
                }
                fn model_name(&self) -> &'static str {
                    "spy"
                }
                fn is_semantic(&self) -> bool {
                    true
                }
                fn category(&self) -> ModelCategory {
                    ModelCategory::StaticEmbedder
                }
            }

            let sink = Arc::new(InMemoryVectorSink::default());
            let captured = Arc::new(Mutex::new(None));
            let fast = Arc::new(SpyEmbedder {
                captured: Arc::clone(&captured),
            });
            let runner = make_runner(
                JobQueueConfig::default(),
                PipelineConfig::default(),
                fast,
                None,
                sink,
            );

            let mut temp = tempfile::NamedTempFile::new().expect("tempfile");
            temp.as_file_mut()
                .set_len(u64::try_from(MAX_SOURCE_FILE_BYTES + 1).expect("size fits in u64"))
                .expect("set oversized source file length");
            let path = temp.path().to_string_lossy().into_owned();

            let preview_source = "P".repeat(MAX_CONTENT_PREVIEW_CHARS + 120);
            let mut req = IngestRequest::new("doc-file-too-large", &preview_source);
            req.source_path = Some(path);
            let ingest_result = runner.ingest(req).expect("ingest");
            assert_eq!(ingest_result.action, IngestAction::New);

            let _ = runner
                .process_batch(&cx, "worker-spy")
                .await
                .expect("process");

            let captured_text = captured
                .lock()
                .unwrap()
                .take()
                .expect("should have captured fallback preview text");
            assert_eq!(captured_text.len(), MAX_CONTENT_PREVIEW_CHARS);
            assert_eq!(
                captured_text,
                truncate_chars(&preview_source, MAX_CONTENT_PREVIEW_CHARS)
            );
        });
    }
}
