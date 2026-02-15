//! Convenience API for building frankensearch indexes in a single method chain.
//!
//! [`IndexBuilder`] handles all the complexity of coordinating embedders,
//! vector index writers, and optional lexical indexing behind a fluent API.
//!
//! # Example
//!
//! ```rust,ignore
//! use frankensearch::IndexBuilder;
//!
//! let stats = IndexBuilder::new("./my_index")
//!     .add_document("doc-1", "Hello world")
//!     .add_document("doc-2", "Distributed consensus algorithms")
//!     .build(&cx)
//!     .await?;
//!
//! println!("Indexed {} docs in {:.1}ms", stats.doc_count, stats.total_ms);
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use asupersync::Cx;
use tracing::instrument;

use frankensearch_core::config::TwoTierConfig;
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::traits::{Embedder, MetricsExporter};
use frankensearch_core::types::{EmbeddingMetrics, IndexMetrics, IndexableDocument};
use frankensearch_embed::auto_detect::EmbedderStack;
use frankensearch_index::{
    TwoTierIndex, TwoTierIndexBuilder, VECTOR_INDEX_FALLBACK_FILENAME, VECTOR_INDEX_FAST_FILENAME,
    VECTOR_INDEX_QUALITY_FILENAME,
};

/// Statistics from a completed index build.
#[derive(Debug, Clone)]
pub struct IndexBuildStats {
    /// Number of documents successfully indexed.
    pub doc_count: usize,
    /// Number of documents that failed to embed (skipped).
    pub error_count: usize,
    /// Per-document errors (`doc_id`, error message).
    pub errors: Vec<(String, String)>,
    /// Total build time in milliseconds.
    pub total_ms: f64,
    /// Time spent on embedding in milliseconds.
    pub embed_ms: f64,
    /// Whether a quality-tier index was built.
    pub has_quality_index: bool,
}

/// Progress update during index building.
#[derive(Debug, Clone)]
pub struct IndexProgress {
    /// Documents processed so far.
    pub completed: usize,
    /// Total documents to process.
    pub total: usize,
    /// Current phase description.
    pub phase: &'static str,
}

/// Fluent builder for creating frankensearch indexes.
///
/// Handles embedder auto-detection, vector index creation, batch embedding,
/// and error aggregation behind a simple API.
pub struct IndexBuilder {
    data_dir: PathBuf,
    config: TwoTierConfig,
    documents: Vec<IndexableDocument>,
    embedder_stack: Option<EmbedderStack>,
    batch_size: usize,
    on_progress: Option<Box<dyn FnMut(IndexProgress) + Send>>,
}

impl IndexBuilder {
    /// Create a new builder targeting the given directory.
    #[must_use]
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            data_dir: data_dir.into(),
            config: TwoTierConfig::default(),
            documents: Vec::new(),
            embedder_stack: None,
            batch_size: 32,
            on_progress: None,
        }
    }

    /// Override the search/index configuration.
    #[must_use]
    pub fn with_config(mut self, config: TwoTierConfig) -> Self {
        self.config = config;
        self
    }

    /// Use a specific embedder stack instead of auto-detecting.
    #[must_use]
    pub fn with_embedder_stack(mut self, stack: EmbedderStack) -> Self {
        self.embedder_stack = Some(stack);
        self
    }

    /// Set the batch size for embedding operations. Default: 32.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    /// Set a progress callback.
    #[must_use]
    pub fn with_progress(mut self, callback: impl FnMut(IndexProgress) + Send + 'static) -> Self {
        self.on_progress = Some(Box::new(callback));
        self
    }

    /// Add a single document to be indexed.
    #[must_use]
    pub fn add_document(mut self, id: impl Into<String>, content: impl Into<String>) -> Self {
        self.documents
            .push(IndexableDocument::new(id.into(), content.into()));
        self
    }

    /// Add a document with title.
    #[must_use]
    pub fn add_document_with_title(
        mut self,
        id: impl Into<String>,
        content: impl Into<String>,
        title: impl Into<String>,
    ) -> Self {
        self.documents
            .push(IndexableDocument::new(id.into(), content.into()).with_title(title.into()));
        self
    }

    /// Add multiple documents.
    #[must_use]
    pub fn add_documents(mut self, docs: impl IntoIterator<Item = IndexableDocument>) -> Self {
        self.documents.extend(docs);
        self
    }

    /// Build the index, embedding all documents and writing FSVI files.
    ///
    /// Returns build statistics including per-document errors.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` if no documents were added.
    /// Returns `SearchError::Io` if the data directory cannot be created.
    /// Individual document embedding failures are collected in `IndexBuildStats.errors`
    /// rather than aborting the build.
    #[allow(clippy::too_many_lines)]
    #[instrument(skip_all, fields(doc_count = self.documents.len(), data_dir = %self.data_dir.display()))]
    pub async fn build(mut self, cx: &Cx) -> SearchResult<IndexBuildStats> {
        let start = Instant::now();
        let metrics_exporter = self.config.metrics_exporter.clone();

        if self.documents.is_empty() {
            let error = SearchError::InvalidConfig {
                field: "documents".to_owned(),
                value: "0".to_owned(),
                reason: "at least one document is required".to_owned(),
            };
            export_error(metrics_exporter.as_ref(), &error);
            return Err(error);
        }

        // Resolve embedder stack.
        let stack = match self.embedder_stack.take() {
            Some(stack) => stack,
            None => EmbedderStack::auto_detect_with(Some(&self.data_dir))?,
        };

        let fast_embedder = stack.fast_arc();
        let quality_embedder = stack.quality_arc();

        // Create index builder.
        let mut index_builder = match TwoTierIndex::create(&self.data_dir, self.config) {
            Ok(builder) => builder,
            Err(error) => {
                export_error(metrics_exporter.as_ref(), &error);
                return Err(error);
            }
        };
        index_builder.set_fast_embedder_id(fast_embedder.id());
        if let Some(ref qe) = quality_embedder {
            index_builder.set_quality_embedder_id(qe.id());
        }

        let total = self.documents.len();
        let mut errors = Vec::new();
        let mut doc_count = 0usize;
        let mut embed_ms = 0.0f64;

        // Process documents in batches.
        for (batch_idx, batch) in self.documents.chunks(self.batch_size).enumerate() {
            let batch_start = Instant::now();

            for doc in batch {
                match Self::embed_and_add(
                    cx,
                    &fast_embedder,
                    quality_embedder.as_deref(),
                    &mut index_builder,
                    doc,
                    metrics_exporter.as_ref(),
                )
                .await
                {
                    Ok(()) => {
                        doc_count += 1;
                    }
                    Err(err) => {
                        tracing::warn!(doc_id = %doc.id, error = %err, "failed to embed document");
                        errors.push((doc.id.clone(), err.to_string()));
                    }
                }
            }

            embed_ms += batch_start.elapsed().as_secs_f64() * 1000.0;

            if let Some(ref mut callback) = self.on_progress {
                let completed = (batch_idx + 1).saturating_mul(self.batch_size);
                callback(IndexProgress {
                    completed: completed.min(total),
                    total,
                    phase: "embedding",
                });
            }
        }

        // Finalize index files.
        if doc_count == 0 {
            let error = SearchError::InvalidConfig {
                field: "documents".to_owned(),
                value: format!("{total}"),
                reason: format!("all {total} documents failed to embed"),
            };
            export_error(metrics_exporter.as_ref(), &error);
            return Err(error);
        }

        let _index = match index_builder.finish() {
            Ok(index) => index,
            Err(error) => {
                export_error(metrics_exporter.as_ref(), &error);
                return Err(error);
            }
        };

        let has_quality = quality_embedder.is_some();
        let index_size_bytes = compute_index_size_bytes(&self.data_dir);
        export_index_updated(
            metrics_exporter.as_ref(),
            doc_count,
            index_size_bytes,
            doc_count,
        );

        tracing::info!(
            doc_count,
            error_count = errors.len(),
            has_quality,
            total_ms = start.elapsed().as_secs_f64() * 1000.0,
            "index build complete"
        );

        Ok(IndexBuildStats {
            doc_count,
            error_count: errors.len(),
            errors,
            total_ms: start.elapsed().as_secs_f64() * 1000.0,
            embed_ms,
            has_quality_index: has_quality,
        })
    }

    /// Embed a single document and add it to the index builder.
    async fn embed_and_add(
        cx: &Cx,
        fast_embedder: &Arc<dyn Embedder>,
        quality_embedder: Option<&dyn Embedder>,
        builder: &mut TwoTierIndexBuilder,
        doc: &IndexableDocument,
        metrics_exporter: Option<&Arc<dyn MetricsExporter>>,
    ) -> SearchResult<()> {
        let text = doc.content.as_str();

        // Fast embedding (required).
        let fast_start = Instant::now();
        let fast_vec = match fast_embedder.embed(cx, text).await {
            Ok(fast_vec) => {
                let duration_ms = fast_start.elapsed().as_secs_f64() * 1000.0;
                export_embedding_completed(metrics_exporter, fast_embedder.as_ref(), duration_ms);
                fast_vec
            }
            Err(error) => {
                export_error(metrics_exporter, &error);
                return Err(error);
            }
        };
        builder.add_fast_record(&doc.id, &fast_vec)?;

        // Quality embedding (optional).
        if let Some(qe) = quality_embedder {
            let quality_start = Instant::now();
            match qe.embed(cx, text).await {
                Ok(quality_vec) => {
                    let duration_ms = quality_start.elapsed().as_secs_f64() * 1000.0;
                    export_embedding_completed(metrics_exporter, qe, duration_ms);
                    builder.add_quality_record(&doc.id, &quality_vec)?;
                }
                Err(error) => {
                    export_error(metrics_exporter, &error);
                    tracing::debug!(
                        doc_id = %doc.id,
                        error = %error,
                        "quality embedding failed, fast-only for this document"
                    );
                }
            }
        }

        Ok(())
    }
}

fn export_error(metrics_exporter: Option<&Arc<dyn MetricsExporter>>, error: &SearchError) {
    if let Some(exporter) = metrics_exporter {
        exporter.on_error(error);
    }
}

fn export_embedding_completed(
    metrics_exporter: Option<&Arc<dyn MetricsExporter>>,
    embedder: &dyn Embedder,
    duration_ms: f64,
) {
    let Some(exporter) = metrics_exporter else {
        return;
    };
    let payload = EmbeddingMetrics {
        embedder_id: embedder.id().to_owned(),
        batch_size: 1,
        duration_ms,
        dimension: embedder.dimension(),
        is_semantic: embedder.is_semantic(),
    };
    exporter.on_embedding_completed(&payload);
}

fn export_index_updated(
    metrics_exporter: Option<&Arc<dyn MetricsExporter>>,
    doc_count: usize,
    index_size_bytes: u64,
    updated_docs: usize,
) {
    let Some(exporter) = metrics_exporter else {
        return;
    };
    let payload = IndexMetrics {
        doc_count,
        index_size_bytes,
        updated_docs,
        staleness_detected: false,
    };
    exporter.on_index_updated(&payload);
}

fn compute_index_size_bytes(data_dir: &Path) -> u64 {
    let fast_path = data_dir.join(VECTOR_INDEX_FAST_FILENAME);
    let fallback_path = data_dir.join(VECTOR_INDEX_FALLBACK_FILENAME);
    let quality_path = data_dir.join(VECTOR_INDEX_QUALITY_FILENAME);

    let fast_bytes = if fast_path.exists() {
        file_size_bytes(&fast_path)
    } else {
        file_size_bytes(&fallback_path)
    };

    fast_bytes.saturating_add(file_size_bytes(&quality_path))
}

fn file_size_bytes(path: &Path) -> u64 {
    std::fs::metadata(path).map_or(0, |metadata| metadata.len())
}

impl std::fmt::Debug for IndexBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexBuilder")
            .field("data_dir", &self.data_dir)
            .field("doc_count", &self.documents.len())
            .field("batch_size", &self.batch_size)
            .field("has_embedder_stack", &self.embedder_stack.is_some())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::Mutex;

    use frankensearch_core::traits::{MetricsExporter, ModelCategory, SearchFuture};
    use frankensearch_core::types::{EmbeddingMetrics, IndexMetrics, SearchMetrics};

    use super::*;

    struct StubEmbedder {
        id: &'static str,
        dim: usize,
    }

    impl Embedder for StubEmbedder {
        fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            let dim = self.dim;
            Box::pin(async move {
                let mut vec = vec![0.0; dim];
                // Simple deterministic embedding from text length.
                vec[text.len() % dim] = 1.0;
                Ok(vec)
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

    #[derive(Debug, Default)]
    struct RecordingExporter {
        search: Mutex<Vec<SearchMetrics>>,
        embedding: Mutex<Vec<EmbeddingMetrics>>,
        index: Mutex<Vec<IndexMetrics>>,
        errors: Mutex<Vec<String>>,
    }

    impl MetricsExporter for RecordingExporter {
        fn on_search_completed(&self, metrics: &SearchMetrics) {
            self.search
                .lock()
                .expect("search metrics lock")
                .push(metrics.clone());
        }

        fn on_embedding_completed(&self, metrics: &EmbeddingMetrics) {
            self.embedding
                .lock()
                .expect("embedding metrics lock")
                .push(metrics.clone());
        }

        fn on_index_updated(&self, metrics: &IndexMetrics) {
            self.index
                .lock()
                .expect("index metrics lock")
                .push(metrics.clone());
        }

        fn on_error(&self, error: &SearchError) {
            self.errors
                .lock()
                .expect("errors lock")
                .push(error.to_string());
        }
    }

    fn stub_stack() -> EmbedderStack {
        let fast = Arc::new(StubEmbedder {
            id: "stub-fast",
            dim: 4,
        });
        let quality = Arc::new(StubEmbedder {
            id: "stub-quality",
            dim: 4,
        });
        EmbedderStack::from_parts(fast, Some(quality))
    }

    fn fast_only_stack() -> EmbedderStack {
        let fast = Arc::new(StubEmbedder {
            id: "stub-fast",
            dim: 4,
        });
        EmbedderStack::from_parts(fast, None)
    }

    #[test]
    fn build_happy_path() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let stats = IndexBuilder::new(dir.path())
                .with_embedder_stack(stub_stack())
                .add_document("doc-1", "Hello world")
                .add_document("doc-2", "Distributed consensus")
                .add_document("doc-3", "Vector search algorithms")
                .build(&cx)
                .await
                .unwrap();

            assert_eq!(stats.doc_count, 3);
            assert_eq!(stats.error_count, 0);
            assert!(stats.has_quality_index);
            assert!(stats.total_ms > 0.0);
            assert!(stats.embed_ms > 0.0);

            // Verify the index can be opened.
            let index = TwoTierIndex::open(dir.path(), TwoTierConfig::default()).unwrap();
            assert_eq!(index.doc_count(), 3);
            assert!(index.has_quality_index());
        });
    }

    #[test]
    fn build_fast_only() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let stats = IndexBuilder::new(dir.path())
                .with_embedder_stack(fast_only_stack())
                .add_document("doc-1", "Test content")
                .build(&cx)
                .await
                .unwrap();

            assert_eq!(stats.doc_count, 1);
            assert!(!stats.has_quality_index);

            let index = TwoTierIndex::open(dir.path(), TwoTierConfig::default()).unwrap();
            assert!(!index.has_quality_index());
        });
    }

    #[test]
    fn build_empty_documents_returns_error() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let result = IndexBuilder::new(dir.path())
                .with_embedder_stack(stub_stack())
                .build(&cx)
                .await;

            assert!(result.is_err());
        });
    }

    #[test]
    fn build_with_progress_callback() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let progress_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let counter = progress_count.clone();

            let stats = IndexBuilder::new(dir.path())
                .with_embedder_stack(stub_stack())
                .with_batch_size(2)
                .add_document("doc-1", "First")
                .add_document("doc-2", "Second")
                .add_document("doc-3", "Third")
                .with_progress(move |_p| {
                    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                })
                .build(&cx)
                .await
                .unwrap();

            assert_eq!(stats.doc_count, 3);
            assert!(progress_count.load(std::sync::atomic::Ordering::Relaxed) > 0);
        });
    }

    #[test]
    fn build_with_title() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let stats = IndexBuilder::new(dir.path())
                .with_embedder_stack(stub_stack())
                .add_document_with_title("doc-1", "Content here", "My Title")
                .build(&cx)
                .await
                .unwrap();

            assert_eq!(stats.doc_count, 1);
        });
    }

    #[test]
    fn build_with_multiple_documents() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let docs = vec![
                IndexableDocument::new("a", "Alpha content"),
                IndexableDocument::new("b", "Beta content"),
                IndexableDocument::new("c", "Gamma content"),
            ];

            let stats = IndexBuilder::new(dir.path())
                .with_embedder_stack(stub_stack())
                .add_documents(docs)
                .build(&cx)
                .await
                .unwrap();

            assert_eq!(stats.doc_count, 3);
        });
    }

    #[test]
    fn debug_impl() {
        let builder = IndexBuilder::new("/tmp/test").add_document("doc-1", "content");
        let debug = format!("{builder:?}");
        assert!(debug.contains("IndexBuilder"));
        assert!(debug.contains("doc_count"));
    }

    #[test]
    fn batch_size_zero_clamped_to_one() {
        let builder = IndexBuilder::new("/tmp/test").with_batch_size(0);
        assert_eq!(builder.batch_size, 1);
    }

    #[test]
    fn batch_size_one_still_works() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let stats = IndexBuilder::new(dir.path())
                .with_embedder_stack(stub_stack())
                .with_batch_size(1)
                .add_document("doc-1", "First document")
                .add_document("doc-2", "Second document")
                .build(&cx)
                .await
                .unwrap();

            assert_eq!(stats.doc_count, 2);
            assert_eq!(stats.error_count, 0);
        });
    }

    #[test]
    fn index_build_stats_debug_clone() {
        let stats = IndexBuildStats {
            doc_count: 5,
            error_count: 1,
            errors: vec![("bad-doc".into(), "embed failed".into())],
            total_ms: 42.0,
            embed_ms: 30.0,
            has_quality_index: true,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.doc_count, 5);
        assert_eq!(cloned.error_count, 1);
        assert_eq!(cloned.errors.len(), 1);
        assert!(cloned.has_quality_index);
        let dbg = format!("{stats:?}");
        assert!(dbg.contains("IndexBuildStats"));
    }

    #[test]
    fn index_progress_debug_clone() {
        let progress = IndexProgress {
            completed: 50,
            total: 100,
            phase: "embedding",
        };
        let cloned = progress.clone();
        assert_eq!(cloned.completed, 50);
        assert_eq!(cloned.total, 100);
        assert_eq!(cloned.phase, "embedding");
        let dbg = format!("{progress:?}");
        assert!(dbg.contains("IndexProgress"));
    }

    #[test]
    fn build_emits_embedding_and_index_metrics() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let exporter = Arc::new(RecordingExporter::default());
            let config = TwoTierConfig::default().with_metrics_exporter(exporter.clone());

            let stats = IndexBuilder::new(dir.path())
                .with_config(config)
                .with_embedder_stack(stub_stack())
                .add_document("doc-1", "Hello world")
                .add_document("doc-2", "Distributed consensus")
                .build(&cx)
                .await
                .unwrap();

            assert_eq!(stats.doc_count, 2);
            assert_eq!(stats.error_count, 0);

            let embedding_count = {
                let embedding_events = exporter.embedding.lock().expect("embedding lock");
                embedding_events.len()
            };
            let (index_count, indexed_docs, indexed_bytes) = {
                let index_events = exporter.index.lock().expect("index lock");
                (
                    index_events.len(),
                    index_events.first().map_or(0, |event| event.doc_count),
                    index_events
                        .first()
                        .map_or(0, |event| event.index_size_bytes),
                )
            };
            let error_count = {
                let errors = exporter.errors.lock().expect("errors lock");
                errors.len()
            };

            assert!(embedding_count >= 4);
            assert_eq!(index_count, 1);
            assert_eq!(indexed_docs, 2);
            assert!(indexed_bytes > 0);
            assert_eq!(error_count, 0);
        });
    }
}
