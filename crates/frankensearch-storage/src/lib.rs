//! FrankenSQLite-backed storage primitives for frankensearch.
//!
//! This crate owns schema bootstrap, document metadata persistence,
//! content-hash dedup bookkeeping, and an embedding job queue.
#![allow(
    clippy::missing_const_for_fn,
    clippy::missing_errors_doc,
    clippy::uninlined_format_args
)]

pub mod connection;
pub mod content_hash;
pub mod document;
#[cfg(feature = "fts5")]
pub mod fts5_adapter;
pub mod history;
pub mod index_metadata;
pub mod job_queue;
pub mod metrics;
pub mod pipeline;
pub mod schema;
pub mod staleness;

pub use connection::{Storage, StorageConfig};
pub use content_hash::{
    ContentHashRecord, ContentHasher, DeduplicationDecision, lookup_content_hash,
    record_content_hash, sha256_hex,
};
pub use document::{
    BatchResult, CrudErrorKind, DocumentRecord, EmbeddingStatus, StatusCounts, count_documents,
    list_document_ids, upsert_document,
};
#[cfg(feature = "fts5")]
pub use fts5_adapter::{
    Fts5AdapterConfig, Fts5ContentMode, Fts5Hit, Fts5LexicalSearch, Fts5TokenizerChoice,
};
pub use history::{
    Bookmark, SearchHistoryEntry, add_bookmark, count_bookmarks, count_search_history,
    is_bookmarked, list_bookmarks, list_search_history, record_search, remove_bookmark,
    remove_bookmark_by_doc, search_history_prefix, truncate_search_history,
};
pub use index_metadata::{
    BuildTrigger, IndexBuildRecord, IndexMetadata, RecordBuildParams, StalenessCheck,
    StalenessReason,
};
pub use job_queue::{
    BatchEnqueueResult, ClaimedJob, EnqueueRequest, FailResult, JobQueueConfig, JobQueueMetrics,
    JobQueueMetricsSnapshot, JobStatus, PersistentJobQueue, QueueDepth, QueueErrorKind,
};
pub use metrics::{StorageMetrics, StorageMetricsSnapshot};
pub use pipeline::{
    BatchIngestResult, BatchProcessResult, EmbeddingVectorSink, InMemoryVectorSink, IngestAction,
    IngestRequest, IngestResult, PersistedEmbedding, PipelineConfig, PipelineMetrics,
    PipelineMetricsSnapshot, StorageBackedJobRunner, WorkerReport,
};
pub use schema::{SCHEMA_VERSION, bootstrap, current_version};
pub use staleness::{
    QuickStalenessCheck, RecommendedAction, StalenessConfig, StalenessLevel, StalenessReport,
    StalenessStats, StorageBackedStaleness,
};
