use std::io;
use std::io::Write;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;

use asupersync::Cx;
use frankensearch_core::traits::{ModelCategory, SearchFuture};
use frankensearch_core::{Canonicalizer, Embedder, SearchError};
use frankensearch_storage::{
    InMemoryVectorSink, IngestAction, IngestRequest, JobQueueConfig, PersistentJobQueue,
    PipelineConfig, Storage, StorageBackedJobRunner,
};
use fsqlite_types::value::SqliteValue;

#[derive(Debug)]
struct StubEmbedder {
    id: &'static str,
    dim: usize,
    fill: f32,
}

impl StubEmbedder {
    const fn new(id: &'static str, dim: usize, fill: f32) -> Self {
        Self { id, dim, fill }
    }
}

impl Embedder for StubEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        let dim = self.dim;
        let fill = self.fill;
        Box::pin(async move { Ok(vec![fill; dim]) })
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
        !self.id.starts_with("fnv1a-")
    }

    fn category(&self) -> ModelCategory {
        if self.id.starts_with("fnv1a-") {
            ModelCategory::HashEmbedder
        } else {
            ModelCategory::StaticEmbedder
        }
    }
}

#[derive(Debug)]
struct SelectiveFailEmbedder {
    id: &'static str,
    dim: usize,
    fill: f32,
    fail_when_contains: &'static str,
}

impl SelectiveFailEmbedder {
    const fn new(
        id: &'static str,
        dim: usize,
        fill: f32,
        fail_when_contains: &'static str,
    ) -> Self {
        Self {
            id,
            dim,
            fill,
            fail_when_contains,
        }
    }
}

impl Embedder for SelectiveFailEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        let dim = self.dim;
        let fill = self.fill;
        let id = self.id;
        let fail_when_contains = self.fail_when_contains;
        Box::pin(async move {
            if text.contains(fail_when_contains) {
                return Err(SearchError::EmbeddingFailed {
                    model: id.to_owned(),
                    source: Box::new(io::Error::other("forced embedder failure")),
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
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::StaticEmbedder
    }
}

#[derive(Clone, Debug)]
struct TestLogWriter {
    buffer: Arc<Mutex<Vec<u8>>>,
}

impl Write for TestLogWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer
            .lock()
            .expect("log buffer lock poisoned")
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
        let guard = buffer.lock().expect("log buffer lock poisoned");
        String::from_utf8_lossy(&guard).into_owned()
    };
    (result, logs)
}

#[allow(clippy::type_complexity, clippy::arc_with_non_send_sync)]
fn make_runner(
    queue_config: JobQueueConfig,
    pipeline_config: PipelineConfig,
    fast_embedder: Arc<dyn Embedder>,
    quality_embedder: Option<Arc<dyn Embedder>>,
) -> (
    StorageBackedJobRunner,
    Arc<Storage>,
    Arc<PersistentJobQueue>,
    Arc<InMemoryVectorSink>,
) {
    let storage = Arc::new(Storage::open_in_memory().expect("storage should open"));
    let queue = Arc::new(PersistentJobQueue::new(Arc::clone(&storage), queue_config));
    let sink = Arc::new(InMemoryVectorSink::default());
    let canonicalizer: Arc<dyn Canonicalizer> =
        Arc::new(frankensearch_core::canonicalize::DefaultCanonicalizer::default());
    let mut runner = StorageBackedJobRunner::new(
        Arc::clone(&storage),
        Arc::clone(&queue),
        canonicalizer,
        fast_embedder,
        Arc::clone(&sink) as Arc<_>,
    )
    .with_config(pipeline_config);

    if let Some(quality) = quality_embedder {
        runner = runner.with_quality_embedder(quality);
    }

    (runner, storage, queue, sink)
}

#[test]
fn ingest_returns_queue_full_when_backpressure_threshold_is_hit() {
    let fast = Arc::new(StubEmbedder::new("fast-tier", 8, 1.0));
    let (runner, _storage, _queue, _sink) = make_runner(
        JobQueueConfig {
            backpressure_threshold: 1,
            ..JobQueueConfig::default()
        },
        PipelineConfig::default(),
        fast,
        None,
    );

    let first = runner
        .ingest(IngestRequest::new("doc-a", "first document"))
        .expect("first ingest should succeed");
    assert!(first.fast_job_enqueued);

    let second = runner
        .ingest(IngestRequest::new("doc-b", "second document"))
        .expect("second ingest should still succeed");
    assert!(second.fast_job_enqueued);

    let third = runner.ingest(IngestRequest::new("doc-c", "third document"));
    let error = third.expect_err("third ingest should fail with queue backpressure");
    match error {
        SearchError::QueueFull { pending, capacity } => {
            assert_eq!(capacity, 1);
            assert!(pending >= 1);
        }
        other => panic!("expected QueueFull, got {other:?}"),
    }
}

#[test]
fn ingest_reports_new_unchanged_updated_and_skipped_actions() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Arc::new(StubEmbedder::new("fast-tier", 8, 1.0));
        let (runner, _storage, queue, _sink) = make_runner(
            JobQueueConfig::default(),
            PipelineConfig::default(),
            fast,
            None,
        );

        let first = runner
            .ingest(IngestRequest::new("doc-variant", "stable content"))
            .expect("first ingest should succeed");
        assert_eq!(first.action, IngestAction::New);
        assert!(first.fast_job_enqueued);

        let processed = runner
            .process_batch(&cx, "variant-worker")
            .await
            .expect("initial process_batch should succeed");
        assert_eq!(processed.jobs_completed, 1);

        let unchanged = runner
            .ingest(IngestRequest::new("doc-variant", "stable content"))
            .expect("unchanged ingest should succeed");
        assert_eq!(unchanged.action, IngestAction::Unchanged);
        assert!(!unchanged.fast_job_enqueued);

        let updated = runner
            .ingest(IngestRequest::new("doc-variant", "updated content"))
            .expect("updated ingest should succeed");
        assert_eq!(updated.action, IngestAction::Updated);
        assert!(updated.fast_job_enqueued);

        let skipped = runner
            .ingest(IngestRequest::new("doc-empty", "   \n\t"))
            .expect("empty canonical text should be skipped");
        assert_eq!(
            skipped.action,
            IngestAction::Skipped {
                reason: "empty_canonical_text".to_owned(),
            }
        );
        assert!(!skipped.fast_job_enqueued);

        let depth = queue.queue_depth().expect("queue depth should succeed");
        assert_eq!(depth.pending, 1);
    });
}

#[test]
fn batch_ingest_reports_mixed_actions_and_enqueue_counts() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Arc::new(StubEmbedder::new("fast-tier", 8, 1.0));
        let (runner, _storage, _queue, _sink) = make_runner(
            JobQueueConfig::default(),
            PipelineConfig::default(),
            fast,
            None,
        );

        let initial = [
            IngestRequest::new("doc-a", "alpha"),
            IngestRequest::new("doc-b", "bravo"),
            IngestRequest::new("doc-empty", ""),
        ];
        let first_summary = runner
            .ingest_batch(&initial)
            .expect("initial batch ingest should succeed");
        assert_eq!(first_summary.requested, 3);
        assert_eq!(first_summary.inserted, 2);
        assert_eq!(first_summary.skipped, 1);
        assert_eq!(first_summary.fast_jobs_enqueued, 2);

        let processed = runner
            .process_batch(&cx, "batch-worker")
            .await
            .expect("initial process_batch should succeed");
        assert_eq!(processed.jobs_completed, 2);

        let second = [
            IngestRequest::new("doc-a", "alpha"),
            IngestRequest::new("doc-b", "bravo-updated"),
            IngestRequest::new("doc-empty-2", " "),
        ];
        let second_summary = runner
            .ingest_batch(&second)
            .expect("second batch ingest should succeed");
        assert_eq!(second_summary.requested, 3);
        assert_eq!(second_summary.unchanged, 1);
        assert_eq!(second_summary.updated, 1);
        assert_eq!(second_summary.skipped, 1);
        assert_eq!(second_summary.fast_jobs_enqueued, 1);
    });
}

#[test]
fn hash_embedder_fast_tier_does_not_enqueue_jobs() {
    let hash_fast = Arc::new(StubEmbedder::new("fnv1a-fast", 8, 1.0));
    let (runner, _storage, queue, _sink) = make_runner(
        JobQueueConfig::default(),
        PipelineConfig::default(),
        hash_fast,
        None,
    );

    let result = runner
        .ingest(IngestRequest::new("doc-hash-tier", "hash-only"))
        .expect("ingest should succeed");
    assert_eq!(result.action, IngestAction::New);
    assert!(!result.fast_job_enqueued);
    assert!(!result.quality_job_enqueued);

    let depth = queue.queue_depth().expect("queue depth should succeed");
    assert_eq!(depth.pending, 0);
}

#[test]
fn failed_job_does_not_abort_other_jobs_in_same_batch() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fast = Arc::new(SelectiveFailEmbedder::new("fast-tier", 8, 1.0, "fail-me"));
        let (runner, _storage, queue, sink) = make_runner(
            JobQueueConfig {
                max_retries: 0,
                ..JobQueueConfig::default()
            },
            PipelineConfig {
                process_batch_size: 8,
                ..PipelineConfig::default()
            },
            fast,
            None,
        );

        runner
            .ingest(IngestRequest::new("doc-ok", "this should embed"))
            .expect("doc-ok ingest should succeed");
        runner
            .ingest(IngestRequest::new("doc-fail", "please fail-me now"))
            .expect("doc-fail ingest should succeed");

        let processed = runner
            .process_batch(&cx, "worker-fail-isolation")
            .await
            .expect("process_batch should succeed");
        assert_eq!(processed.jobs_claimed, 2);
        assert_eq!(processed.jobs_completed, 1);
        assert_eq!(processed.jobs_failed, 1);

        let entries = sink.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].doc_id, "doc-ok");

        let depth = queue.queue_depth().expect("queue depth should succeed");
        assert_eq!(depth.completed, 1);
        assert_eq!(depth.failed, 1);
    });
}

#[test]
fn claim_batch_orders_two_tier_jobs_by_priority() {
    let fast = Arc::new(StubEmbedder::new("fast-tier", 8, 1.0));
    let quality = Arc::new(StubEmbedder::new("quality-tier", 8, 2.0));
    let (runner, _storage, queue, _sink) = make_runner(
        JobQueueConfig::default(),
        PipelineConfig {
            fast_priority: 9,
            quality_priority: 1,
            ..PipelineConfig::default()
        },
        fast,
        Some(quality),
    );

    let ingest = runner
        .ingest(IngestRequest::new("doc-priority", "priority ordering"))
        .expect("ingest should succeed");
    assert!(ingest.fast_job_enqueued);
    assert!(ingest.quality_job_enqueued);

    let claimed = queue
        .claim_batch("priority-worker", 2)
        .expect("claim should succeed");
    assert_eq!(claimed.len(), 2);
    assert_eq!(claimed[0].embedder_id, "fast-tier");
    assert_eq!(claimed[1].embedder_id, "quality-tier");
    assert!(claimed[0].priority > claimed[1].priority);
}

#[test]
fn logs_include_ingest_and_job_completion_events() {
    let ((), logs) = with_captured_logs(|| {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Arc::new(StubEmbedder::new("fast-tier", 4, 1.0));
            let (runner, _storage, _queue, _sink) = make_runner(
                JobQueueConfig::default(),
                PipelineConfig::default(),
                fast,
                None,
            );

            runner
                .ingest(IngestRequest::new("doc-log", "log me"))
                .expect("ingest should succeed");

            let batch = runner
                .process_batch(&cx, "log-worker")
                .await
                .expect("process_batch should succeed");
            assert_eq!(batch.jobs_completed, 1);
        });
    });

    assert!(logs.contains("document ingest completed"), "logs: {logs}");
    assert!(logs.contains("embedding job completed"), "logs: {logs}");
}

#[test]
fn worker_reclaims_stale_jobs_on_startup_and_logs_exit() {
    let ((), logs) = with_captured_logs(|| {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Arc::new(StubEmbedder::new("fast-tier", 4, 1.0));
            let (runner, storage, queue, sink) = make_runner(
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
            );

            runner
                .ingest(IngestRequest::new("doc-reclaim", "stale job"))
                .expect("ingest should succeed");

            let claimed = queue
                .claim_batch("pre-crash-worker", 1)
                .expect("claim should succeed");
            assert_eq!(claimed.len(), 1);

            let params = [
                SqliteValue::Integer(1),
                SqliteValue::Integer(claimed[0].job_id),
            ];
            storage
                .connection()
                .execute_with_params(
                    "UPDATE embedding_jobs SET started_at = ?1 WHERE job_id = ?2;",
                    &params,
                )
                .expect("stale timestamp update should succeed");

            let shutdown = AtomicBool::new(false);
            let report = runner
                .run_worker(&cx, "reclaim-worker", &shutdown)
                .await
                .expect("run_worker should succeed");

            assert_eq!(report.reclaimed_on_startup, 1);
            assert_eq!(sink.entries().len(), 1);
        });
    });

    assert!(
        logs.contains("storage-backed embedding worker exited"),
        "logs: {logs}"
    );
}
