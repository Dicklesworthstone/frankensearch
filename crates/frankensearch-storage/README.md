# frankensearch-storage

FrankenSQLite-backed metadata and embedding job storage for frankensearch.

## Overview

This crate owns the persistent storage layer for frankensearch, backed by FrankenSQLite. It manages schema bootstrapping, document metadata persistence, content-hash deduplication, an embedding job queue, search history, bookmarks, index build metadata, and staleness detection. It serves as the bridge between frankensearch's in-memory search pipeline and durable on-disk state.

## Key Types

### Storage and Connection

- `Storage` - main storage handle wrapping a FrankenSQLite connection
- `StorageConfig` - configuration for storage initialization

### Document Management

- `DocumentRecord` - stored document metadata record
- `upsert_document` - insert or update a document with dedup
- `list_document_ids` / `count_documents` - document enumeration and counting
- `EmbeddingStatus` / `StatusCounts` - per-document embedding state tracking

### Content Hashing and Deduplication

- `ContentHasher` - SHA-256 content hashing for change detection
- `DeduplicationDecision` - whether to re-embed or skip a document
- `lookup_content_hash` / `record_content_hash` - hash lookup and persistence

### Job Queue

- `PersistentJobQueue` - durable embedding job queue with claim/complete/fail lifecycle
- `ClaimedJob` / `EnqueueRequest` - job queue request and claim types
- `ClaimOutcome` - applied, lost-claim, or superseded-attempt result of a guarded operation
- `JobQueueConfig` / `JobQueueMetrics` - queue configuration and telemetry
- `QueueDepth` - current queue depth by status

### Pipeline

- `StorageBackedJobRunner` - orchestrates document ingestion through the embedding pipeline
- `IngestRequest` / `IngestResult` / `IngestAction` - ingestion request/result types
- `PipelineConfig` / `PipelineMetrics` - pipeline configuration and performance metrics
- `EmbeddingVectorSink` / `InMemoryVectorSink` - sinks for produced embedding vectors

Persistent jobs require an identity-aware embedder. The runner calls
`Embedder::embed_bound`, validates the returned vector and complete identity,
and rejects a producer that changes identity during inference. Custom
embedders must provide `Embedder::identity`; a model name and dimension alone
do not authorize persistence. Cancellation after inference leaves the claim
unfinished for the existing lease-recovery path, without writing a vector or
recording a job failure.

Every claimed attempt carries a durable epoch and worker ID. The runner checks
that exact owner and binds the catalog's current content hash before each item,
then checks again after inference and immediately before persistence. Reclaiming
a job under the same worker name still invalidates the old attempt. Completion,
terminal failure, and skip update the queue and document embedding status in
one guarded transaction. Retries validate ownership and document revision but
do not mark a terminal catalog status. Stale responses and stale errors cannot
complete, fail, or overwrite a replacement job. Reports expose these local
attempts as `jobs_suppressed`,
separately from actual skipped jobs and failures. The current document hash is
the authority: a document that changes from A to B and back to A can still admit
an A response even while an obsolete B job remains queued.

A sink must still admit the producer against its own index generation. The
runner deliberately holds no database transaction across the synchronous sink
call. If ownership or content changes during that call, its guarded completion
leaves the newer queue and catalog untouched, but vector bytes may already have
been written. Atomic vector publication requires a version-aware sink or a
compare-and-swap publication protocol; these queue fences alone do not provide
that boundary.

### History and Bookmarks

- `record_search` / `list_search_history` - search history recording and retrieval
- `add_bookmark` / `list_bookmarks` / `is_bookmarked` - document bookmarking

### Index Metadata and Staleness

- `IndexMetadata` / `IndexBuildRecord` - index build tracking
- `StalenessCheck` / `StalenessReport` - index freshness detection
- `StorageBackedStaleness` - staleness checking backed by stored metadata

### Schema

- `bootstrap` - creates/migrates the storage schema
- `SCHEMA_VERSION` / `current_version` - schema versioning

## Features

| Feature | Description |
|---------|-------------|
| `fts5` | Enables `Fts5LexicalSearch` adapter using FrankenSQLite FTS5 |

## Usage

```rust
use frankensearch_storage::{Storage, StorageConfig, bootstrap};

// Open or create a storage database
let config = StorageConfig::default();
let storage = Storage::open("/path/to/search.db", config)
    .expect("open storage");

// Bootstrap schema
bootstrap(&storage).expect("bootstrap schema");

// Upsert a document
// upsert_document(&storage, &doc).expect("upsert");
```

## Dependency Graph Position

```
frankensearch-core
  ^
  |
frankensearch-storage
  ^
  |-- frankensearch-fsfs
  |-- frankensearch (root, optional, feature: storage)
```

## License

MIT
