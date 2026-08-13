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
//!     .await?; // errors if auto-detect finds no semantic model
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
use frankensearch_core::traits::LexicalRead;
use frankensearch_core::traits::{Embedder, MetricsExporter};
use frankensearch_core::types::{EmbeddingMetrics, IndexMetrics, IndexableDocument};
#[cfg(all(feature = "durability", feature = "quill"))]
use frankensearch_durability::FileProtector;
#[cfg(feature = "durability")]
use frankensearch_durability::{DefaultSymbolCodec, DurabilityConfig, FsviProtector};
use frankensearch_embed::auto_detect::{EmbedderStack, TwoTierAvailability};
use frankensearch_fusion::SyncTwoTierSearcher;
use frankensearch_index::{
    FsviV2IdentityBinding, TwoTierIndex, TwoTierIndexBuilder, TwoTierIndexPaths,
    VECTOR_INDEX_FALLBACK_FILENAME, VECTOR_INDEX_FAST_FILENAME, VECTOR_INDEX_QUALITY_FILENAME,
};
// bd-6281c: this module opens Tantivy in exactly two configurations — the
// blue-green/direct arms of the `quill` reader, and the standalone
// `lexical`-without-`quill` reader/writer below. `lexical-tantivy` on its own
// (the cass-compat lane) only re-exports the namespace for foreign consumers,
// so gating this import on that feature alone left it unused there.
#[cfg(any(
    all(feature = "quill", feature = "lexical-tantivy"),
    all(feature = "lexical", not(feature = "quill"))
))]
use frankensearch_lexical::TantivyIndex;
#[cfg(feature = "quill")]
use frankensearch_quill::{
    BlueGreenEngine, LexicalLayout, QuillConfig, QuillIndex, RootBoundQuillSearchIndex,
    inspect_lexical_layout,
};

/// Per-arm byte accounting for a completed build (bd-8nqz.3).
///
/// Vector-only size reporting hid the lexical arm entirely; every arm that
/// wrote bytes must appear here so no aggregate can mask a missing arm.
#[derive(Debug, Clone, Copy, Default)]
pub struct IndexSizeBreakdown {
    /// Sum of all arms below.
    pub total: u64,
    /// Fast-tier FSVI bytes (dedicated or fallback filename).
    pub vector_fast: u64,
    /// Quality-tier FSVI bytes (0 when no quality index was built).
    pub vector_quality: u64,
    /// Recursive size of the lexical index directory (0 when absent).
    pub lexical: u64,
}

/// Receipt for the lexical indexing arm of a build (bd-8nqz.3).
///
/// Lexical admission is independent of embedding outcome: every valid source
/// document is attempted here even when its embeddings failed, so the
/// documents most in need of lexical fallback remain lexically searchable.
#[derive(Debug, Clone)]
pub struct LexicalArmReceipt {
    /// Active backend: `"quill"` or `"tantivy"`.
    pub backend: &'static str,
    /// Directory the lexical index was written to.
    pub path: PathBuf,
    /// Documents attempted (all valid source documents).
    pub attempted: usize,
    /// Documents successfully indexed.
    pub indexed: usize,
    /// Per-document lexical errors (`doc_id`, error message).
    pub errors: Vec<(String, String)>,
    /// Published manifest generation. Currently `None`: the keeper snapshot
    /// does not expose its generation publicly yet; the root-bound reader
    /// work (bd-8nqz.2) adds that accessor and fills this in.
    pub generation: Option<u64>,
    /// Whether the lexical index was published (bulk seal / commit reached).
    pub published: bool,
}

/// Statistics from a completed index build.
#[derive(Debug, Clone)]
pub struct IndexBuildStats {
    /// Total valid source documents submitted to the build.
    pub source_count: usize,
    /// Number of documents successfully indexed into the fast vector tier.
    pub doc_count: usize,
    /// Number of documents whose fast embedding failed (absent from the
    /// vector tiers; still admitted to the lexical arm when enabled).
    pub error_count: usize,
    /// Per-document fast-embedding errors (`doc_id`, error message).
    pub errors: Vec<(String, String)>,
    /// Documents successfully indexed into the quality vector tier.
    /// Zero when no quality embedder is configured.
    pub quality_indexed: usize,
    /// Per-document quality-embedding errors (`doc_id`, error message).
    /// These documents remain in the fast tier (fast-only degradation).
    pub quality_errors: Vec<(String, String)>,
    /// Receipt for the lexical arm. `None` when lexical indexing is compiled
    /// out or no document reached lexical staging.
    pub lexical: Option<LexicalArmReceipt>,
    /// Per-arm byte accounting (replaces vector-only size reporting).
    pub size_bytes: IndexSizeBreakdown,
    /// Total build time in milliseconds.
    pub total_ms: f64,
    /// Time spent on embedding in milliseconds.
    pub embed_ms: f64,
    /// Time spent building the lexical arm in milliseconds (0 when skipped).
    pub lexical_ms: f64,
    /// Whether a quality-tier index was built.
    pub has_quality_index: bool,
    /// Embedder availability this generation was actually built with.
    ///
    /// The index is written with the vector identity of whatever embedder was
    /// resolved, so a [`TwoTierAvailability::HashOnly`] build produces a
    /// permanently non-semantic generation: installing a model afterwards
    /// cannot repair it, because the stored vectors are hashes. Callers that
    /// care about semantic quality must inspect this rather than assume the
    /// build was semantic — a degraded build otherwise succeeds silently and
    /// returns plausible-looking hits at query time (`bd-a6zt`).
    pub embedder_availability: TwoTierAvailability,
}

impl IndexBuildStats {
    /// Whether this generation was built with a degraded embedder stack.
    ///
    /// `true` means the index content itself is degraded, not merely the
    /// current process configuration.
    #[must_use]
    pub const fn is_degraded_generation(&self) -> bool {
        self.embedder_availability.is_degraded()
    }
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
    #[cfg(all(
        any(feature = "lexical", feature = "quill"),
        feature = "bench-internals"
    ))]
    clone_lexical_staging_for_benchmark: bool,
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
            #[cfg(all(
                any(feature = "lexical", feature = "quill"),
                feature = "bench-internals"
            ))]
            clone_lexical_staging_for_benchmark: false,
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

    /// Retain the former deep-clone staging path for same-binary performance comparisons.
    #[cfg(all(
        any(feature = "lexical", feature = "quill"),
        feature = "bench-internals"
    ))]
    #[doc(hidden)]
    #[must_use]
    pub fn with_clone_lexical_staging_for_benchmark(mut self) -> Self {
        self.clone_lexical_staging_for_benchmark = true;
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
    /// rather than aborting the build. Structured cancellation is never collected as a
    /// document failure: it is returned immediately as [`SearchError::Cancelled`].
    #[allow(clippy::too_many_lines)]
    #[instrument(skip_all, fields(doc_count = self.documents.len(), data_dir = %self.data_dir.display()))]
    pub async fn build(mut self, cx: &Cx) -> SearchResult<IndexBuildStats> {
        let start = Instant::now();
        let metrics_exporter = self.config.metrics_exporter.clone();

        build_checkpoint(cx, "index build start")?;

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
        let explicit_stack = self.embedder_stack.is_some();
        let stack = match self.embedder_stack.take() {
            Some(stack) => stack,
            None => EmbedderStack::auto_detect_with(Some(&self.data_dir))?,
        };
        build_checkpoint(cx, "index builder initialization")?;

        // A degraded stack here is not a transient runtime condition: the
        // vectors written below carry this embedder's identity, so the
        // generation is permanently degraded and a later model install cannot
        // repair it — only a compatible rebuild can. Auto-detect HashOnly is
        // never a silent success: that is how semantic search "just works"
        // as lexical-only forever (`bd-a6zt`). Hash remains reachable only
        // through an explicit stack (tests / control policy).
        let embedder_availability = stack.availability();
        if matches!(embedder_availability, TwoTierAvailability::HashOnly) && !explicit_stack {
            let error = SearchError::EmbedderUnavailable {
                model: "semantic".to_owned(),
                reason: stack.degradation_message().unwrap_or_else(|| {
                    "auto-detect found no semantic model; refusing to write a hash generation"
                        .to_owned()
                }),
            };
            export_error(metrics_exporter.as_ref(), &error);
            return Err(error);
        }
        if embedder_availability.is_degraded() {
            tracing::warn!(
                availability = %embedder_availability,
                data_dir = %self.data_dir.display(),
                fast_embedder = %stack.fast().id(),
                detail = stack.degradation_message().unwrap_or_default(),
                "building index with a DEGRADED embedder stack; this generation is written with \
                 that embedder's vector identity and installing a model later will NOT repair it \
                 — a compatible rebuild is required",
            );
        }

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
        // bd-9xuj T2-C2: the warning above promises the written vectors carry
        // this embedder's identity — perform that binding for real. When the
        // embedder supplies its complete identity bundle, thread it through
        // the builder so the built index carries the typed producing identity
        // (validated, dimension-checked at finish) and its header revision,
        // not just the id string. A legacy embedder without a bundle stays a
        // typed legacy-unidentified build: absence is routed, never
        // fabricated from id strings.
        match fast_embedder.identity() {
            Ok(identity) => {
                if let Err(error) = index_builder.set_fast_identity(identity) {
                    export_error(metrics_exporter.as_ref(), &error);
                    return Err(error);
                }
            }
            Err(reason) => {
                tracing::debug!(
                    fast_embedder = %fast_embedder.id(),
                    %reason,
                    "fast embedder supplies no identity bundle; this generation is \
                     built legacy-unidentified"
                );
            }
        }
        if let Some(ref qe) = quality_embedder {
            match qe.identity() {
                Ok(identity) => {
                    if let Err(error) = index_builder.set_quality_identity(identity) {
                        export_error(metrics_exporter.as_ref(), &error);
                        return Err(error);
                    }
                }
                Err(reason) => {
                    tracing::debug!(
                        quality_embedder = %qe.id(),
                        %reason,
                        "quality embedder supplies no identity bundle; this generation's \
                         quality tier is built legacy-unidentified"
                    );
                }
            }
        }

        let total = self.documents.len();
        let mut errors = Vec::new();
        let mut doc_count = 0usize;
        let mut quality_indexed = 0usize;
        let mut quality_errors: Vec<(String, String)> = Vec::new();
        let mut embed_ms = 0.0f64;
        #[cfg(any(feature = "lexical", feature = "quill"))]
        let mut lexical_docs = Vec::with_capacity(total);

        // Keep the old borrowed loop available only for the same-binary benchmark arm. This is the
        // exact former residency behavior: all originals stay in `self.documents` while successful
        // documents are deep-cloned into lexical staging.
        #[cfg(all(
            any(feature = "lexical", feature = "quill"),
            feature = "bench-internals"
        ))]
        if self.clone_lexical_staging_for_benchmark {
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
                        // Benchmark arm: pins the exact former residency AND
                        // former admission behavior (embed-gated lexical
                        // staging); quality receipts are not collected here.
                        Ok(_) => {
                            doc_count += 1;
                            lexical_docs.push(doc.clone());
                        }
                        Err(error @ SearchError::Cancelled { .. }) => return Err(error),
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
        } else {
            // `build` owns the input documents, so move successful values into lexical staging.
            let mut documents = std::mem::take(&mut self.documents).into_iter();
            let batch_count = total.div_ceil(self.batch_size);
            for batch_idx in 0..batch_count {
                let batch_start = Instant::now();
                for doc in documents.by_ref().take(self.batch_size) {
                    match Self::embed_and_add(
                        cx,
                        &fast_embedder,
                        quality_embedder.as_deref(),
                        &mut index_builder,
                        &doc,
                        metrics_exporter.as_ref(),
                    )
                    .await
                    {
                        Ok(quality_error) => {
                            doc_count += 1;
                            if let Some(message) = quality_error {
                                quality_errors.push((doc.id.clone(), message));
                            } else if quality_embedder.is_some() {
                                quality_indexed += 1;
                            }
                            lexical_docs.push(doc);
                        }
                        Err(error @ SearchError::Cancelled { .. }) => return Err(error),
                        Err(err) => {
                            tracing::warn!(doc_id = %doc.id, error = %err, "failed to embed document");
                            errors.push((doc.id.clone(), err.to_string()));
                            // bd-8nqz.3: lexical admission is independent of
                            // embedding outcome — the documents most in need
                            // of lexical fallback must stay lexically
                            // searchable.
                            lexical_docs.push(doc);
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
        }

        #[cfg(all(
            any(feature = "lexical", feature = "quill"),
            not(feature = "bench-internals")
        ))]
        {
            // `build` owns the input documents, so move successful values into lexical staging.
            let mut documents = std::mem::take(&mut self.documents).into_iter();
            let batch_count = total.div_ceil(self.batch_size);
            for batch_idx in 0..batch_count {
                let batch_start = Instant::now();
                for doc in documents.by_ref().take(self.batch_size) {
                    match Self::embed_and_add(
                        cx,
                        &fast_embedder,
                        quality_embedder.as_deref(),
                        &mut index_builder,
                        &doc,
                        metrics_exporter.as_ref(),
                    )
                    .await
                    {
                        Ok(quality_error) => {
                            doc_count += 1;
                            if let Some(message) = quality_error {
                                quality_errors.push((doc.id.clone(), message));
                            } else if quality_embedder.is_some() {
                                quality_indexed += 1;
                            }
                            lexical_docs.push(doc);
                        }
                        Err(error @ SearchError::Cancelled { .. }) => return Err(error),
                        Err(err) => {
                            tracing::warn!(doc_id = %doc.id, error = %err, "failed to embed document");
                            errors.push((doc.id.clone(), err.to_string()));
                            // bd-8nqz.3: lexical admission is independent of
                            // embedding outcome — the documents most in need
                            // of lexical fallback must stay lexically
                            // searchable.
                            lexical_docs.push(doc);
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
        }

        // Without lexical indexing there is no staging clone to remove, so retain the former path
        // and its metrics/drop timing exactly.
        #[cfg(not(any(feature = "lexical", feature = "quill")))]
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
                    Ok(quality_error) => {
                        doc_count += 1;
                        if let Some(message) = quality_error {
                            quality_errors.push((doc.id.clone(), message));
                        } else if quality_embedder.is_some() {
                            quality_indexed += 1;
                        }
                    }
                    Err(error @ SearchError::Cancelled { .. }) => return Err(error),
                    Err(err) => {
                        tracing::warn!(doc_id = %doc.id, error = %err, "failed to embed document");
                        errors.push((doc.id.clone(), err.to_string()));
                    }
                }
            }
            embed_ms += batch_start.elapsed().as_secs_f64() * 1_000.0;
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
        build_checkpoint(cx, "vector index finalize")?;
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
        build_checkpoint(cx, "vector index finalized")?;

        #[cfg(not(any(feature = "lexical", feature = "quill")))]
        let (lexical_receipt, lexical_ms): (Option<LexicalArmReceipt>, f64) = (None, 0.0);
        #[cfg(any(feature = "lexical", feature = "quill"))]
        let (lexical_receipt, lexical_ms) = if lexical_docs.is_empty() {
            (None, 0.0)
        } else {
            build_checkpoint(cx, "lexical index build")?;
            let lexical_path = self.data_dir.join("lexical");
            let lexical_start = Instant::now();
            match build_lexical_index(cx, &lexical_path, &lexical_docs).await {
                Ok(receipt) => (
                    Some(receipt),
                    lexical_start.elapsed().as_secs_f64() * 1000.0,
                ),
                // Publication failure (create/seal/commit) stays fatal: a
                // half-written lexical index is worse than an absent arm.
                // Per-document indexing errors are NOT fatal; they are
                // reported in the receipt.
                Err(error) => {
                    export_error(metrics_exporter.as_ref(), &error);
                    return Err(error);
                }
            }
        };

        #[cfg(feature = "durability")]
        {
            build_checkpoint(cx, "durability sidecar protection")?;
            if let Err(error) = protect_durability_sidecars(&self.data_dir) {
                export_error(metrics_exporter.as_ref(), &error);
                return Err(error);
            }
            build_checkpoint(cx, "durability sidecars protected")?;
        }

        build_checkpoint(cx, "index build completion")?;
        let has_quality = quality_embedder.is_some();
        let size_bytes = compute_size_breakdown(&self.data_dir);
        export_index_updated(
            metrics_exporter.as_ref(),
            doc_count,
            size_bytes.total,
            doc_count,
        );

        tracing::info!(
            doc_count,
            error_count = errors.len(),
            has_quality,
            total_ms = start.elapsed().as_secs_f64() * 1000.0,
            "index build complete"
        );

        let stats = IndexBuildStats {
            source_count: total,
            doc_count,
            error_count: errors.len(),
            errors,
            quality_indexed,
            quality_errors,
            lexical: lexical_receipt,
            size_bytes,
            total_ms: start.elapsed().as_secs_f64() * 1000.0,
            embed_ms,
            lexical_ms,
            has_quality_index: has_quality,
            embedder_availability,
        };

        Ok(stats)
    }

    /// Embed a single document and add it to the index builder.
    ///
    /// Returns `Ok(None)` when every attempted arm succeeded, and
    /// `Ok(Some(message))` when the fast tier succeeded but the quality
    /// embedding failed (fast-only degradation for this document). A fast
    /// embedding failure is a hard `Err`: the document enters no vector tier.
    async fn embed_and_add(
        cx: &Cx,
        fast_embedder: &Arc<dyn Embedder>,
        quality_embedder: Option<&dyn Embedder>,
        builder: &mut TwoTierIndexBuilder,
        doc: &IndexableDocument,
        metrics_exporter: Option<&Arc<dyn MetricsExporter>>,
    ) -> SearchResult<Option<String>> {
        let text = doc.content.as_str();

        // Fast embedding (required).
        build_checkpoint(cx, "fast document embedding")?;
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
        build_checkpoint(cx, "fast document embedding completion")?;
        builder.add_fast_record(&doc.id, &fast_vec)?;

        // Quality embedding (optional).
        if let Some(qe) = quality_embedder {
            build_checkpoint(cx, "quality document embedding")?;
            let quality_start = Instant::now();
            match qe.embed(cx, text).await {
                Ok(quality_vec) => {
                    build_checkpoint(cx, "quality document embedding completion")?;
                    let duration_ms = quality_start.elapsed().as_secs_f64() * 1000.0;
                    export_embedding_completed(metrics_exporter, qe, duration_ms);
                    builder.add_quality_record(&doc.id, &quality_vec)?;
                }
                Err(error @ SearchError::Cancelled { .. }) => {
                    export_error(metrics_exporter, &error);
                    return Err(error);
                }
                Err(error) => {
                    export_error(metrics_exporter, &error);
                    tracing::debug!(
                        doc_id = %doc.id,
                        error = %error,
                        "quality embedding failed, fast-only for this document"
                    );
                    return Ok(Some(error.to_string()));
                }
            }
        }

        Ok(None)
    }
}

fn build_checkpoint(cx: &Cx, phase: &'static str) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: phase.to_owned(),
        reason: cx
            .cancel_reason()
            .map_or_else(|| error.to_string(), |reason| reason.to_string()),
    })
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

fn compute_size_breakdown(data_dir: &Path) -> IndexSizeBreakdown {
    let fast_path = data_dir.join(VECTOR_INDEX_FAST_FILENAME);
    let fallback_path = data_dir.join(VECTOR_INDEX_FALLBACK_FILENAME);
    let quality_path = data_dir.join(VECTOR_INDEX_QUALITY_FILENAME);

    let vector_fast = if fast_path.exists() {
        file_size_bytes(&fast_path)
    } else {
        file_size_bytes(&fallback_path)
    };
    let vector_quality = file_size_bytes(&quality_path);
    let lexical = dir_size_bytes(&data_dir.join("lexical"));

    IndexSizeBreakdown {
        total: vector_fast
            .saturating_add(vector_quality)
            .saturating_add(lexical),
        vector_fast,
        vector_quality,
        lexical,
    }
}

/// Recursive byte size of a directory tree; 0 when the directory is absent.
fn dir_size_bytes(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries.filter_map(Result::ok).fold(0u64, |acc, entry| {
        let path = entry.path();
        let size = if path.is_dir() {
            dir_size_bytes(&path)
        } else {
            file_size_bytes(&path)
        };
        acc.saturating_add(size)
    })
}

/// The lexical engine admitted by [`open_hybrid`].
///
/// [`Self::Quill`] is the default facade authority. [`Self::TantivyOracle`]
/// is an explicitly selected rollback/oracle reader; it never means that the
/// default `lexical` facade alias resolved to Tantivy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LexicalReaderBackend {
    /// The default Quill lexical reader.
    Quill,
    /// The explicit Tantivy oracle or accepted rollback reader.
    TantivyOracle,
}

/// The opened arms of a hybrid index directory (bd-8nqz.3).
#[derive(Clone)]
pub struct HybridIndexParts {
    /// Two-tier vector index.
    pub vectors: Arc<TwoTierIndex>,
    /// Active lexical reader for `<dir>/lexical`, when one exists and a
    /// lexical backend is compiled in.
    pub lexical: Option<Arc<dyn LexicalRead>>,
    /// Actual lexical engine behind [`Self::lexical`].
    ///
    /// A caller that requires Quill's semantic behavior can fail closed when
    /// this is [`Some(LexicalReaderBackend::TantivyOracle)`]. It is `None`
    /// exactly when [`Self::lexical`] is absent.
    pub lexical_backend: Option<LexicalReaderBackend>,
}

impl std::fmt::Debug for HybridIndexParts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridIndexParts")
            .field("has_lexical", &self.lexical.is_some())
            .field("lexical_backend", &self.lexical_backend)
            .finish_non_exhaustive()
    }
}

/// Open every arm of an index directory produced by [`IndexBuilder`].
///
/// The advertised hybrid flow used to build lexical data and then construct
/// a searcher without attaching it; this helper makes the correct wiring the
/// ergonomic default:
///
/// ```rust,ignore
/// let parts = open_hybrid(&cx, "./my_index", TwoTierConfig::default()).await?;
/// let mut searcher = TwoTierSearcher::new(parts.vectors, fast_embedder, config);
/// if let Some(lexical) = parts.lexical {
///     searcher = searcher.with_lexical(lexical);
/// }
/// ```
///
/// # Errors
///
/// Returns an error when the vector index cannot be opened, or when a
/// lexical directory exists but its index fails to open (a corrupt lexical
/// arm is reported, never silently dropped). A missing lexical directory
/// yields `lexical: None` and `lexical_backend: None`, as does a build without
/// a lexical backend compiled in. When the arm is present,
/// `lexical_backend` records whether it is the default Quill reader or an
/// explicit Tantivy oracle/rollback reader.
pub async fn open_hybrid(
    cx: &Cx,
    data_dir: impl AsRef<Path>,
    config: TwoTierConfig,
) -> SearchResult<HybridIndexParts> {
    let data_dir = data_dir.as_ref();
    let vectors = Arc::new(TwoTierIndex::open(data_dir, config)?);

    let lexical_dir = data_dir.join("lexical");
    let (lexical, lexical_backend) = if lexical_dir.is_dir() {
        open_lexical_reader(cx, &lexical_dir).await?
    } else {
        (None, None)
    };

    Ok(HybridIndexParts {
        vectors,
        lexical,
        lexical_backend,
    })
}

/// Open the default synchronous product for exactly admitted FSVI v2 tiers.
///
/// Unlike [`open_hybrid`], this opener requires the caller's v2 identity
/// bindings because a v2 artifact has no legitimate path-only open. It
/// performs the same admitted-product open used by the shipping index API,
/// then routes the retained owners through the in-memory synchronous product
/// and its optional generation-keyed residual cache. A missing, corrupt, or
/// unavailable cache leaves the exact flat scan selected; it does not select a
/// different retrieval algorithm.
///
/// `residual_cache_dir` is shared safely by both tiers because cache artifact
/// names are generation-keyed. It need not exist: an unavailable optional cache
/// is a flat-exact fallback rather than an opening error.
///
/// # Errors
///
/// Returns exact v2 admission errors, including a missing quality binding for a
/// configured quality path, plus source-vector loading errors from the
/// synchronous product.
pub fn open_admitted_v2_sync_with_residual_sidecar_cache(
    paths: &TwoTierIndexPaths,
    fast_binding: &FsviV2IdentityBinding,
    quality_binding: Option<&FsviV2IdentityBinding>,
    residual_cache_dir: impl AsRef<Path>,
    config: TwoTierConfig,
) -> SearchResult<SyncTwoTierSearcher> {
    // This opener consumes only retained admitted FSVI owners to construct the
    // in-memory exact product. Strip optional ANN paths before `TwoTierIndex`
    // assembly: otherwise its `ann` path can build and persist HNSW sidecars
    // that are immediately discarded with this temporary owner container.
    // Fast/quality index paths remain intact, so exact tier admission,
    // cross-tier publication validation, and identity checks are unchanged.
    let mut owner_paths = TwoTierIndexPaths::new(paths.fast_index().to_path_buf());
    if let Some(quality_path) = paths.quality_index() {
        owner_paths = owner_paths.with_quality_index(quality_path.to_path_buf());
    }
    let admitted = TwoTierIndex::open_admitted_v2_with_paths(
        &owner_paths,
        config.clone(),
        fast_binding,
        quality_binding,
    )?;
    let fast_source = admitted
        .fast_admitted_owner()
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "admitted_v2.fast_owner".to_owned(),
            value: paths.fast_index().display().to_string(),
            reason: "exact v2 product open did not retain its fast owner".to_owned(),
        })?;
    let cache_dir = residual_cache_dir.as_ref();
    let quality_source = admitted
        .quality_admitted_owner()
        .map(|source| (source, cache_dir));
    SyncTwoTierSearcher::from_admitted_v2_with_residual_sidecar_cache(
        fast_source,
        cache_dir,
        quality_source,
        config,
    )
}

#[cfg(feature = "quill")]
async fn open_lexical_reader(
    cx: &Cx,
    dir: &Path,
) -> SearchResult<(Option<Arc<dyn LexicalRead>>, Option<LexicalReaderBackend>)> {
    // bd-8nqz.2: the Quill path stays bound to the lexical *root*, not the
    // engine child selected during this call. `RootBoundQuillSearchIndex`
    // validates CURRENT before its atomic refresh swap, so a future refresh
    // can follow a later Quill publication without ever mutating this root.
    let layout = inspect_lexical_layout(dir).map_err(|source| SearchError::SubsystemError {
        subsystem: "facade.lexical.layout",
        source: Box::new(source),
    })?;
    match layout {
        LexicalLayout::Empty => Ok((None, None)),
        // NOTE: deliberately two arms, not an or-pattern — `pointer` is only
        // bound in the BlueGreen variant, so `DirectQuill | BlueGreen {..} if
        // pointer.engine() == ...` is E0408 (pointer not bound in all
        // patterns) whenever this fn is compiled (feature "quill" enabled by
        // downstream workspaces such as mcp_agent_mail_rust).
        LexicalLayout::DirectQuill => {
            let index = RootBoundQuillSearchIndex::open(cx, dir, QuillConfig::default()).await?;
            Ok((Some(Arc::new(index)), Some(LexicalReaderBackend::Quill)))
        }
        LexicalLayout::BlueGreen { ref pointer, .. }
            if pointer.engine() == BlueGreenEngine::Quill =>
        {
            let index = RootBoundQuillSearchIndex::open(cx, dir, QuillConfig::default()).await?;
            Ok((Some(Arc::new(index)), Some(LexicalReaderBackend::Quill)))
        }
        #[cfg(feature = "lexical-tantivy")]
        LexicalLayout::DirectTantivy => {
            let index = TantivyIndex::open(dir)?;
            Ok((
                Some(Arc::new(index)),
                Some(LexicalReaderBackend::TantivyOracle),
            ))
        }
        #[cfg(feature = "lexical-tantivy")]
        LexicalLayout::BlueGreen { ref pointer, .. }
            if pointer.engine() == BlueGreenEngine::Tantivy =>
        {
            let index = TantivyIndex::open(&pointer.engine_dir(dir))?;
            Ok((
                Some(Arc::new(index)),
                Some(LexicalReaderBackend::TantivyOracle),
            ))
        }
        ref layout => Err(SearchError::InvalidConfig {
            field: "data_dir/lexical".to_owned(),
            value: dir.display().to_string(),
            reason: format!(
                "lexical layout is {}, which this build cannot open",
                layout.label()
            ),
        }),
    }
}

#[cfg(all(feature = "lexical", not(feature = "quill")))]
async fn open_lexical_reader(
    _cx: &Cx,
    dir: &Path,
) -> SearchResult<(Option<Arc<dyn LexicalRead>>, Option<LexicalReaderBackend>)> {
    let index = TantivyIndex::open(dir)?;
    Ok((
        Some(Arc::new(index)),
        Some(LexicalReaderBackend::TantivyOracle),
    ))
}

#[cfg(not(any(feature = "lexical", feature = "quill")))]
async fn open_lexical_reader(
    _cx: &Cx,
    _dir: &Path,
) -> SearchResult<(Option<Arc<dyn LexicalRead>>, Option<LexicalReaderBackend>)> {
    Ok((None, None))
}

#[cfg(feature = "quill")]
async fn build_lexical_index(
    cx: &Cx,
    data_dir: &Path,
    documents: &[IndexableDocument],
) -> SearchResult<LexicalArmReceipt> {
    build_checkpoint(cx, "Quill lexical index initialization")?;
    // bd-8nqz.2: never initialize Quill on top of a foreign, damaged,
    // blue-green, or ambiguous layout — MANIFEST absence is NOT emptiness.
    // Empty proceeds; DirectQuill preserves the existing create-over-own
    // behavior; everything else is a typed refusal.
    match inspect_lexical_layout(data_dir).map_err(|source| SearchError::SubsystemError {
        subsystem: "facade.lexical.layout",
        source: Box::new(source),
    })? {
        LexicalLayout::Empty | LexicalLayout::DirectQuill => {}
        layout => {
            return Err(SearchError::InvalidConfig {
                field: "data_dir/lexical".to_owned(),
                value: data_dir.display().to_string(),
                reason: format!(
                    "refusing to initialize Quill over a {} lexical layout; \
                     inspect or repair the directory first",
                    layout.label()
                ),
            });
        }
    }

    let config = QuillConfig {
        bulk_load_mode: true,
        ..QuillConfig::default()
    };

    #[cfg(feature = "durability")]
    let lexical = {
        let protector =
            FileProtector::new(Arc::new(DefaultSymbolCodec), DurabilityConfig::default())?;
        QuillIndex::create_durable(cx, data_dir, config, protector).await?
    };
    #[cfg(not(feature = "durability"))]
    let lexical = QuillIndex::create(cx, data_dir, config).await?;
    build_checkpoint(cx, "Quill lexical document indexing")?;

    // Per-document indexing so one rejected document (duplicate id, oversized
    // field) cannot silently void the whole arm; failures land in the
    // receipt, not in an aggregate error.
    let mut indexed = 0usize;
    let mut errors: Vec<(String, String)> = Vec::new();
    let mut documents_iter = documents.iter();
    while let Some(document) = documents_iter.next() {
        build_checkpoint(cx, "Quill lexical document indexing")?;
        match lexical.index_document(cx, document).await {
            Ok(()) => indexed += 1,
            // Cancellation is a caller contract, not a document defect: abort
            // the build with the typed error instead of laundering it into
            // the receipt and spinning the recovery machinery.
            Err(error @ frankensearch_quill::QuillIndexError::Cancelled { .. }) => {
                return Err(error.into());
            }
            Err(error) => {
                tracing::warn!(
                    doc_id = %document.id,
                    error = %error,
                    "lexical indexing failed for document"
                );
                errors.push((document.id.clone(), error.to_string()));
                // A failed batch arms Quill's fail-closed retry guard: the
                // batch is ambiguous until a commit reconciles its accepted
                // prefix (Quill contract test: successful_sealed_batches_
                // compose_but_failed_batches_require_commit_retry). Reconcile
                // so one rejected document cannot void the rest of the arm.
                match lexical.commit(cx).await {
                    Ok(_) => {}
                    Err(error @ frankensearch_quill::QuillIndexError::Cancelled { .. }) => {
                        return Err(error.into());
                    }
                    Err(recovery_error) => {
                        tracing::warn!(
                            error = %recovery_error,
                            "lexical writer recovery failed; remaining documents skipped"
                        );
                        // Exact accounting: every unattempted document is recorded,
                        // never silently dropped.
                        for skipped in documents_iter {
                            errors.push((
                                skipped.id.clone(),
                                format!(
                                    "skipped: lexical writer recovery failed after prior \
                                     error: {recovery_error}"
                                ),
                            ));
                        }
                        break;
                    }
                }
            }
        }
    }
    build_checkpoint(cx, "Quill lexical publication")?;
    let _ = lexical.finish_bulk_load(cx).await?;
    build_checkpoint(cx, "Quill lexical publication complete")?;
    Ok(LexicalArmReceipt {
        backend: "quill",
        path: data_dir.to_path_buf(),
        attempted: documents.len(),
        indexed,
        errors,
        generation: None,
        published: true,
    })
}

#[cfg(all(feature = "lexical", not(feature = "quill")))]
async fn build_lexical_index(
    cx: &Cx,
    data_dir: &Path,
    documents: &[IndexableDocument],
) -> SearchResult<LexicalArmReceipt> {
    // bd-b7pz: the LexicalRead flip (0220d5c5) left this write-side arm
    // without the trait that provides index_documents/commit; this cfg combo
    // (lexical without quill) is not built by the default-feature gates.
    use frankensearch_core::traits::LexicalWrite;

    build_checkpoint(cx, "Tantivy lexical index initialization")?;
    let lexical = TantivyIndex::create(data_dir)?;
    let mut indexed = 0usize;
    let mut errors: Vec<(String, String)> = Vec::new();
    for document in documents {
        build_checkpoint(cx, "Tantivy lexical document indexing")?;
        match lexical
            .index_documents(cx, std::slice::from_ref(document))
            .await
        {
            Ok(()) => indexed += 1,
            Err(error @ SearchError::Cancelled { .. }) => return Err(error),
            Err(error) => {
                tracing::warn!(
                    doc_id = %document.id,
                    error = %error,
                    "lexical indexing failed for document"
                );
                errors.push((document.id.clone(), error.to_string()));
            }
        }
    }
    build_checkpoint(cx, "Tantivy lexical publication")?;
    lexical.commit(cx).await?;
    build_checkpoint(cx, "Tantivy lexical publication complete")?;
    Ok(LexicalArmReceipt {
        backend: "tantivy",
        path: data_dir.to_path_buf(),
        attempted: documents.len(),
        indexed,
        errors,
        generation: None,
        published: true,
    })
}

#[cfg(feature = "durability")]
fn protect_durability_sidecars(data_dir: &Path) -> SearchResult<()> {
    let protector = FsviProtector::new(Arc::new(DefaultSymbolCodec), DurabilityConfig::default())?;

    let fast_path = {
        let dedicated = data_dir.join(VECTOR_INDEX_FAST_FILENAME);
        if dedicated.exists() {
            dedicated
        } else {
            data_dir.join(VECTOR_INDEX_FALLBACK_FILENAME)
        }
    };
    if fast_path.exists() {
        protector.protect_atomic(&fast_path)?;
    }

    let quality_path = data_dir.join(VECTOR_INDEX_QUALITY_FILENAME);
    if quality_path.exists() {
        protector.protect_atomic(&quality_path)?;
    }

    Ok(())
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
    #[cfg(target_os = "linux")]
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
    #[cfg(not(any(feature = "lexical", feature = "quill")))]
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[cfg(not(any(feature = "lexical", feature = "quill")))]
    use asupersync::types::CancelKind;
    #[cfg(all(feature = "quill", feature = "lexical-tantivy"))]
    use frankensearch_core::traits::LexicalWrite;
    use frankensearch_core::traits::{MetricsExporter, ModelCategory, SearchFuture};
    use frankensearch_core::types::{EmbeddingMetrics, IndexMetrics, SearchMetrics};
    #[cfg(feature = "durability")]
    use frankensearch_durability::FsviProtector;
    #[cfg(all(feature = "durability", feature = "quill"))]
    use frankensearch_durability::{DefaultSymbolCodec, DurabilityConfig, FsviVerifyResult};
    #[cfg(any(
        all(feature = "quill", feature = "lexical-tantivy"),
        all(feature = "lexical", not(feature = "quill"))
    ))]
    use frankensearch_lexical::TantivyIndex;
    #[cfg(feature = "quill")]
    use frankensearch_quill::{
        BlueGreenEngine, DEFAULT_SCHEMA, EncodedSegment, SectionInput, SectionKind,
        SegmentHeaderInput, SegmentReader, load_manifest_pair,
    };
    #[cfg(all(feature = "quill", feature = "lexical-tantivy"))]
    use frankensearch_quill::{CurrentPointer, publish_current};

    use super::*;

    #[cfg(target_os = "linux")]
    fn owned_admitted_v2_sync_dir() -> std::path::PathBuf {
        static NONCE: AtomicU64 = AtomicU64::new(0);
        let parent = std::env::temp_dir().join("frankensearch_admitted_v2_sync_tests");
        std::fs::create_dir_all(&parent).expect("create durable test parent");
        for _ in 0..1024 {
            let nonce = NONCE.fetch_add(1, AtomicOrdering::Relaxed);
            let dir = parent.join(format!(
                "{}-{}-{nonce}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            ));
            match std::fs::create_dir(&dir) {
                Ok(()) => return dir,
                // A name this process already minted: fall through to the next
                // nonce. The match is the loop body's final expression, so an
                // empty arm retries exactly as an explicit `continue` did.
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => panic!("create unique admitted-v2 sync directory: {error}"),
            }
        }
        panic!("exhausted admitted-v2 sync test directory names")
    }

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

    struct SelectiveFailEmbedder;

    impl Embedder for SelectiveFailEmbedder {
        fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async move {
                if text.contains("fail-fast-embedding") {
                    return Err(SearchError::EmbeddingFailed {
                        model: "selective-fail".to_owned(),
                        source: Box::new(std::io::Error::other("intentional test failure")),
                    });
                }
                Ok(vec![1.0, 0.0, 0.0, 0.0])
            })
        }

        fn dimension(&self) -> usize {
            4
        }

        fn id(&self) -> &'static str {
            "selective-fail"
        }

        fn model_name(&self) -> &'static str {
            "selective-fail"
        }

        fn is_semantic(&self) -> bool {
            true
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::StaticEmbedder
        }
    }

    #[cfg(not(any(feature = "lexical", feature = "quill")))]
    struct CancelOnFirstEmbedder {
        id: &'static str,
        calls: Arc<AtomicUsize>,
        return_typed_error: bool,
    }

    #[cfg(not(any(feature = "lexical", feature = "quill")))]
    impl Embedder for CancelOnFirstEmbedder {
        fn embed<'a>(&'a self, cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            let id = self.id;
            let calls = Arc::clone(&self.calls);
            let return_typed_error = self.return_typed_error;
            Box::pin(async move {
                let call = calls.fetch_add(1, Ordering::SeqCst);
                assert_eq!(call, 0, "{id} was called after cancelling the build");
                cx.cancel_with(CancelKind::User, Some("cancel-on-first embedder"));
                if return_typed_error {
                    Err(SearchError::Cancelled {
                        phase: format!("{id} embedding"),
                        reason: "cancel-on-first embedder".to_owned(),
                    })
                } else {
                    Ok(vec![1.0, 0.0, 0.0, 0.0])
                }
            })
        }

        fn dimension(&self) -> usize {
            4
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

    #[cfg(not(any(feature = "lexical", feature = "quill")))]
    fn cancel_on_first_stack(
        cancelled_tier: &'static str,
        calls: Arc<AtomicUsize>,
    ) -> EmbedderStack {
        let cancelling: Arc<dyn Embedder> = Arc::new(CancelOnFirstEmbedder {
            id: cancelled_tier,
            calls,
            return_typed_error: cancelled_tier != "fast-canceller",
        });
        if cancelled_tier == "fast-canceller" {
            EmbedderStack::from_parts(cancelling, None)
        } else {
            let fast: Arc<dyn Embedder> = Arc::new(StubEmbedder {
                id: "stub-fast",
                dim: 4,
            });
            EmbedderStack::from_parts(fast, Some(cancelling))
        }
    }

    /// Identity-aware stub (bd-9xuj T2-C2): same deterministic vectors as
    /// [`StubEmbedder`], but it supplies a complete identity bundle the way
    /// production embedders do, so builds through it must bind the typed
    /// producing identity — not just the id string.
    struct IdentityStubEmbedder {
        id: &'static str,
        dim: usize,
        identity: frankensearch_core::generation::EmbeddingIdentityBundleV1,
    }

    impl IdentityStubEmbedder {
        fn new(id: &'static str, dim: usize) -> Self {
            Self {
                id,
                dim,
                identity:
                    frankensearch_core::generation::EmbeddingIdentityBundleV1::explicit_test_model(
                        id,
                        u32::try_from(dim).expect("test dimension fits u32"),
                    ),
            }
        }
    }

    impl Embedder for IdentityStubEmbedder {
        fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            let dim = self.dim;
            Box::pin(async move {
                let mut vec = vec![0.0; dim];
                if let Some(slot) = vec.get_mut(text.len() % dim) {
                    *slot = 1.0;
                }
                Ok(vec)
            })
        }

        fn identity(
            &self,
        ) -> SearchResult<&frankensearch_core::generation::EmbeddingIdentityBundleV1> {
            Ok(&self.identity)
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

    fn fast_only_stack() -> EmbedderStack {
        let fast = Arc::new(StubEmbedder {
            id: "stub-fast",
            dim: 4,
        });
        EmbedderStack::from_parts(fast, None)
    }

    /// A hash-only build must report itself as a DEGRADED generation
    /// (`bd-a6zt` Cause A). The vectors written carry the hash embedder's
    /// identity, so this index can never answer semantically and installing a
    /// model later cannot repair it — yet the build succeeds and returns
    /// plausible results at query time. Without a machine-readable signal on
    /// the build result there is nothing for a caller to check, which is
    /// exactly how this shipped unnoticed.
    #[cfg(feature = "hash")]
    #[test]
    fn build_reports_hash_only_generation_as_degraded() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let hash_stack = EmbedderStack::from_parts(
                Arc::new(frankensearch_embed::HashEmbedder::default_256()) as Arc<dyn Embedder>,
                None,
            );
            let stats = IndexBuilder::new(dir.path())
                .with_embedder_stack(hash_stack)
                .add_document("doc-1", "Hello world")
                .add_document("doc-2", "Distributed consensus")
                .build(&cx)
                .await
                .unwrap();

            assert_eq!(stats.doc_count, 2, "the degraded build still succeeds");
            assert_eq!(
                stats.embedder_availability,
                TwoTierAvailability::HashOnly,
                "a hash-only stack must be reported, not inferred",
            );
            assert!(
                stats.is_degraded_generation(),
                "hash-only generations are permanently non-semantic and must say so",
            );
        });
    }

    #[test]
    fn build_refuses_auto_detected_hash_only_generation() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let error = IndexBuilder::new(dir.path())
                .add_document("doc-1", "Hello world")
                .build(&cx)
                .await
                .expect_err("auto-detect hash must not write a generation");
            assert!(
                matches!(error, SearchError::EmbedderUnavailable { .. }),
                "expected EmbedderUnavailable, got: {error:?}"
            );
        });
    }

    /// The converse, so the signal cannot be a constant `true`: a healthy
    /// two-tier build must report `Full` and must NOT be flagged degraded.
    #[test]
    fn build_reports_healthy_generation_as_not_degraded() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let stats = IndexBuilder::new(dir.path())
                .with_embedder_stack(stub_stack())
                .add_document("doc-1", "Hello world")
                .build(&cx)
                .await
                .unwrap();

            assert_eq!(stats.embedder_availability, TwoTierAvailability::Full);
            assert!(!stats.is_degraded_generation());
        });
    }

    /// A semantic fast tier with no quality tier is degraded too, but it is a
    /// *different* degradation: the generation is still semantic, so it is
    /// repairable by adding a quality model without a rebuild. Pinning both
    /// arms keeps the two cases from collapsing into one boolean.
    #[test]
    fn build_reports_fast_only_generation_as_degraded_but_semantic() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let stats = IndexBuilder::new(dir.path())
                .with_embedder_stack(fast_only_stack())
                .add_document("doc-1", "Hello world")
                .build(&cx)
                .await
                .unwrap();

            assert_eq!(stats.embedder_availability, TwoTierAvailability::FastOnly);
            assert!(stats.is_degraded_generation());
            assert_ne!(
                stats.embedder_availability,
                TwoTierAvailability::HashOnly,
                "fast-only must not be conflated with the non-semantic hash case",
            );
        });
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

    /// bd-9xuj T2-C2: `build` must thread the embedders' REAL identities into
    /// the built index, not just their id strings. Observable through the
    /// persisted v1 headers: the id string AND the space's immutable revision
    /// survive a reopen — while the space fingerprint does not (v1 persists
    /// no identity), and that absence must stay typed, never re-fabricated
    /// from the surviving strings.
    #[test]
    fn build_threads_typed_identity_into_persisted_headers() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let fast = IdentityStubEmbedder::new("identity-fast", 4);
            let quality = IdentityStubEmbedder::new("identity-quality", 4);
            let fast_revision = fast.identity.space.immutable_revision.clone();
            let quality_revision = quality.identity.space.immutable_revision.clone();
            let stack = EmbedderStack::from_parts(Arc::new(fast), Some(Arc::new(quality)));

            let stats = IndexBuilder::new(dir.path())
                .with_embedder_stack(stack)
                .add_document("doc-1", "Hello world")
                .add_document("doc-2", "Distributed consensus")
                .build(&cx)
                .await
                .unwrap();
            assert_eq!(stats.doc_count, 2);

            let reopened = TwoTierIndex::open(dir.path(), TwoTierConfig::default()).unwrap();
            assert_eq!(reopened.fast_embedder_id(), "identity-fast");
            assert_eq!(
                reopened.fast_embedder_revision(),
                fast_revision.as_str(),
                "the fast header must carry the identity's immutable revision, \
                 which only exists if build threaded the typed identity through"
            );
            assert_eq!(reopened.quality_embedder_id(), Some("identity-quality"));
            assert_eq!(
                reopened.quality_embedder_revision(),
                Some(quality_revision.as_str())
            );
            assert_eq!(
                reopened.fast_space_fingerprint_hex(),
                None,
                "v1 artifacts persist no space identity; reopen is typed \
                 legacy-unidentified, never fabricated from header strings"
            );
            assert_eq!(reopened.quality_space_fingerprint_hex(), None);
        });
    }

    /// The legacy arm stays legacy: a stack of identity-less embedders keeps
    /// building exactly as before — empty header revisions, no identity —
    /// so the typed threading cannot have made identity a requirement.
    #[test]
    fn build_without_identity_bundles_stays_legacy_unidentified() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let stats = IndexBuilder::new(dir.path())
                .with_embedder_stack(stub_stack())
                .add_document("doc-1", "Hello world")
                .build(&cx)
                .await
                .unwrap();
            assert_eq!(stats.doc_count, 1);

            let reopened = TwoTierIndex::open(dir.path(), TwoTierConfig::default()).unwrap();
            assert_eq!(reopened.fast_embedder_id(), "stub-fast");
            assert_eq!(reopened.fast_embedder_revision(), "");
            assert_eq!(reopened.quality_embedder_revision(), Some(""));
            assert_eq!(reopened.fast_space_fingerprint_hex(), None);
            assert_eq!(reopened.quality_space_fingerprint_hex(), None);
        });
    }

    /// bd-9xuj T2-C2 follow-up (review #8151): threading identities through
    /// `build` introduced a NEW production failure mode — a build that
    /// previously succeeded now fails when an embedder's declared identity
    /// does not describe the vectors it actually emits. Pin the failure:
    /// declared space dimension 8, emitted vectors dimension 4 →
    /// `TwoTierIndexBuilder::finish` rejects with typed `InvalidConfig` at
    /// `fast_identity.space.dimension` — never a panic, and never an index
    /// carrying an identity that lies about its vectors.
    #[test]
    fn build_fails_typed_when_declared_identity_dimension_mismatches_vectors() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            // Emits 4-dim vectors but declares an 8-dim space: exactly the
            // self-inconsistent embedder the finish()-time check exists for.
            let lying = IdentityStubEmbedder {
                id: "identity-lying-fast",
                dim: 4,
                identity:
                    frankensearch_core::generation::EmbeddingIdentityBundleV1::explicit_test_model(
                        "identity-lying-fast",
                        8,
                    ),
            };
            let stack = EmbedderStack::from_parts(Arc::new(lying), None);
            let error = IndexBuilder::new(dir.path())
                .with_embedder_stack(stack)
                .add_document("doc-1", "Hello world")
                .build(&cx)
                .await
                .expect_err(
                    "a declared identity that does not describe the emitted vectors \
                     must fail the build with a typed error",
                );
            assert!(
                matches!(
                    &error,
                    SearchError::InvalidConfig {
                        field,
                        value,
                        reason,
                    } if field == "fast_identity.space.dimension"
                        && value == "8"
                        && reason.contains("does not describe")
                ),
                "expected typed InvalidConfig at fast_identity.space.dimension \
                 (value 8, refusing an identity that does not describe the \
                 written vectors), got {error:?}"
            );
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

    #[cfg(any(feature = "lexical", feature = "quill"))]
    #[test]
    fn build_wires_lexical_index_when_feature_enabled() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let stats = IndexBuilder::new(dir.path())
                .with_embedder_stack(stub_stack())
                .add_document("doc-1", "Alpha retrieval content")
                .add_document("doc-2", "Beta ranking content")
                .build(&cx)
                .await
                .unwrap();

            assert_eq!(stats.doc_count, 2);

            #[cfg(feature = "quill")]
            let hits = {
                let lexical =
                    QuillIndex::open(&cx, dir.path().join("lexical"), QuillConfig::default())
                        .await
                        .unwrap();
                lexical.search_results(&cx, "Alpha", 5).unwrap()
            };
            #[cfg(all(feature = "lexical", not(feature = "quill")))]
            let hits = {
                let lexical = TantivyIndex::open(&dir.path().join("lexical")).unwrap();
                lexical.search(&cx, "Alpha", 5).await.unwrap()
            };
            assert!(!hits.is_empty());
        });
    }

    /// Eager TERMDICT admission deliberately upgrades malformed dictionary
    /// bytes from a query-time subsystem failure to an open-time corruption
    /// diagnosis. Pin that public facade contract with a structurally valid,
    /// checksum-consistent FSLX image whose dictionary payload is malformed.
    #[cfg(feature = "quill")]
    #[test]
    fn open_hybrid_reports_malformed_termdict_as_index_corrupted() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            IndexBuilder::new(dir.path())
                .with_embedder_stack(stub_stack())
                .add_document("doc-alpha", "alpha retrieval sentinel")
                .build(&cx)
                .await
                .expect("build intact hybrid fixture");

            let intact = open_hybrid(&cx, dir.path(), TwoTierConfig::default())
                .await
                .expect("open intact hybrid fixture");
            let lexical = intact.lexical.as_ref().expect("Quill lexical arm");
            let hits = lexical
                .search(&cx, "alpha", 10)
                .await
                .expect("search intact Quill arm");
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].doc_id.as_str(), "doc-alpha");
            drop(intact);

            let lexical_dir = dir.path().join("lexical");
            let segment_path = std::fs::read_dir(&lexical_dir)
                .expect("read lexical fixture directory")
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .find(|path| {
                    path.extension()
                        .is_some_and(|extension| extension == "fslx")
                })
                .expect("published FSLX fixture");
            let reader = SegmentReader::from_owned(
                std::fs::read(&segment_path).expect("read intact FSLX fixture"),
                DEFAULT_SCHEMA,
            )
            .expect("parse intact FSLX fixture");
            let header = reader.header();
            let malformed_termdict = 1_u32.to_le_bytes();
            let payloads = reader
                .section_entries()
                .iter()
                .map(|entry| {
                    let bytes = if entry.kind == SectionKind::TERMDICT {
                        malformed_termdict.to_vec()
                    } else {
                        reader
                            .section(entry.kind)
                            .expect("verify intact section")
                            .expect("known section")
                            .to_vec()
                    };
                    (entry.kind, entry.flags, bytes)
                })
                .collect::<Vec<_>>();
            let sections = payloads
                .iter()
                .map(|(kind, flags, bytes)| SectionInput {
                    kind: *kind,
                    flags: *flags,
                    bytes,
                })
                .collect::<Vec<_>>();
            let corrupted = EncodedSegment::encode(
                SegmentHeaderInput {
                    segment_id: header.segment_id,
                    schema: DEFAULT_SCHEMA,
                    docid_lo: header.docid_lo,
                    docid_hi: header.docid_hi,
                    doc_count: header.doc_count,
                    created_unix_s: header.created_unix_s,
                    engine_version: header.engine_version,
                },
                &sections,
            )
            .expect("encode checksum-consistent malformed TERMDICT fixture");
            std::fs::write(&segment_path, corrupted.as_bytes())
                .expect("publish malformed FSLX fixture");

            let mut manifest = load_manifest_pair(&lexical_dir)
                .expect("load lexical fixture manifest")
                .manifest;
            let witness = manifest
                .segments
                .iter_mut()
                .find(|segment| segment.segment_id == header.segment_id)
                .expect("manifest witness for fixture segment");
            witness.file_len = corrupted.file_len();
            witness.file_xxh3 = corrupted.file_xxh3();
            let manifest_bytes = manifest
                .to_bytes()
                .expect("encode updated fixture manifest");
            std::fs::write(lexical_dir.join("MANIFEST"), &manifest_bytes)
                .expect("publish updated fixture manifest");
            let previous_manifest = lexical_dir.join("MANIFEST.prev");
            if previous_manifest.exists() {
                std::fs::write(previous_manifest, &manifest_bytes)
                    .expect("keep equal-generation fixture manifests byte-identical");
            }

            let error = open_hybrid(&cx, dir.path(), TwoTierConfig::default())
                .await
                .expect_err("malformed TERMDICT must fail during facade open");
            assert!(
                matches!(&error, SearchError::IndexCorrupted { .. }),
                "expected IndexCorrupted, got {error:?}"
            );
            if let SearchError::IndexCorrupted { path, detail } = error {
                assert_eq!(path, segment_path);
                assert!(
                    detail.contains("TERMDICT declares 1 blocks in only 4 bytes"),
                    "corruption detail must identify the malformed dictionary: {detail}"
                );
            }
        });
    }

    /// bd-8nqz.3: lexical admission is independent of embedding outcome. A
    /// document whose fast embedding fails is exactly the document that needs
    /// lexical fallback, so it MUST be lexically searchable — the previous
    /// contract (embed-gated staging) silently dropped it from both arms.
    #[cfg(any(feature = "lexical", feature = "quill"))]
    #[test]
    fn lexical_admission_survives_fast_embedding_failures() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let stack = EmbedderStack::from_parts(Arc::new(SelectiveFailEmbedder), None);
            let stats = IndexBuilder::new(dir.path())
                .with_embedder_stack(stack)
                .with_batch_size(2)
                .add_document("doc-first", "first-success sentinel")
                .add_document("doc-failed", "fail-fast-embedding admitted sentinel")
                .add_document("doc-last", "last-success sentinel")
                .build(&cx)
                .await
                .unwrap();

            assert_eq!(stats.source_count, 3);
            assert_eq!(stats.doc_count, 2, "vector arm holds only embedded docs");
            assert_eq!(stats.error_count, 1);
            assert_eq!(stats.errors[0].0, "doc-failed");

            let receipt = stats.lexical.as_ref().expect("lexical arm receipt");
            assert_eq!(receipt.attempted, 3, "every valid source doc attempted");
            assert_eq!(receipt.indexed, 3);
            assert!(receipt.errors.is_empty());
            assert!(receipt.published);
            assert!(
                stats.size_bytes.lexical > 0,
                "lexical bytes must be visible"
            );
            assert_eq!(
                stats.size_bytes.total,
                stats.size_bytes.vector_fast
                    + stats.size_bytes.vector_quality
                    + stats.size_bytes.lexical,
            );

            // The vector arm must NOT contain the failed document...
            let index = TwoTierIndex::open(dir.path(), TwoTierConfig::default()).unwrap();
            assert_eq!(index.doc_count(), 2);

            // ...but the lexical arm MUST.
            #[cfg(feature = "quill")]
            let admitted_ids = {
                let lexical =
                    QuillIndex::open(&cx, dir.path().join("lexical"), QuillConfig::default())
                        .await
                        .unwrap();
                lexical
                    .search_doc_ids(&cx, "admitted", 10)
                    .unwrap()
                    .into_iter()
                    .map(|hit| hit.document_id)
                    .collect::<Vec<_>>()
            };
            #[cfg(all(feature = "lexical", not(feature = "quill")))]
            let admitted_ids = {
                let lexical = TantivyIndex::open(&dir.path().join("lexical")).unwrap();
                lexical
                    .search_doc_ids(&cx, "admitted", 10)
                    .unwrap()
                    .into_iter()
                    .map(|hit| hit.doc_id.to_string())
                    .collect::<Vec<_>>()
            };
            assert_eq!(admitted_ids, vec!["doc-failed".to_owned()]);
        });
    }

    #[cfg(feature = "durability")]
    #[test]
    fn build_wires_durability_sidecars_when_feature_enabled() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let stats = IndexBuilder::new(dir.path())
                .with_embedder_stack(stub_stack())
                .add_document("doc-1", "Durability alpha")
                .add_document("doc-2", "Durability beta")
                .build(&cx)
                .await
                .unwrap();

            assert_eq!(stats.doc_count, 2);

            let fast_path = {
                let dedicated = dir.path().join(super::VECTOR_INDEX_FAST_FILENAME);
                if dedicated.exists() {
                    dedicated
                } else {
                    dir.path().join(super::VECTOR_INDEX_FALLBACK_FILENAME)
                }
            };
            let fast_sidecar = FsviProtector::sidecar_path(&fast_path);
            assert!(fast_sidecar.exists());

            #[cfg(feature = "quill")]
            {
                let lexical_dir = dir.path().join("lexical");
                let protected_sidecars = std::fs::read_dir(&lexical_dir)
                    .unwrap()
                    .filter_map(Result::ok)
                    .map(|entry| entry.path())
                    .filter(|path| path.extension().is_some_and(|extension| extension == "fec"))
                    .collect::<Vec<_>>();
                assert!(
                    protected_sidecars
                        .iter()
                        .any(|path| path.to_string_lossy().ends_with(".fslx.fec")),
                    "bulk-built Quill segment must have a generic FileProtector sidecar: \
                     {protected_sidecars:?}"
                );
                assert!(
                    protected_sidecars
                        .iter()
                        .any(|path| path.file_name().is_some_and(|name| name == "MANIFEST.fec")),
                    "published Quill manifest must have a generic FileProtector sidecar: \
                     {protected_sidecars:?}"
                );

                let fslx_sidecar = protected_sidecars
                    .iter()
                    .find(|path| path.to_string_lossy().ends_with(".fslx.fec"))
                    .expect("protected Quill FSLX segment");
                let fslx_path = std::path::PathBuf::from(
                    fslx_sidecar
                        .to_string_lossy()
                        .strip_suffix(".fec")
                        .expect("FSLX sidecar suffix"),
                );
                let original = std::fs::read(&fslx_path).expect("read protected FSLX");
                assert!(!original.is_empty());

                let protector =
                    FsviProtector::new(Arc::new(DefaultSymbolCodec), DurabilityConfig::default())
                        .expect("construct FSLX verifier");
                assert_eq!(
                    protector.verify(&fslx_path).expect("verify intact FSLX"),
                    FsviVerifyResult::Intact
                );

                let mut corrupted = original.clone();
                corrupted[0] ^= 0xff;
                std::fs::write(&fslx_path, &corrupted).expect("corrupt FSLX fixture");
                assert!(matches!(
                    protector
                        .verify(&fslx_path)
                        .expect("detect FSLX corruption"),
                    FsviVerifyResult::Corrupted { repairable: true }
                ));
                assert!(
                    protector
                        .verify_and_repair(&fslx_path)
                        .expect("repair FSLX from sidecar")
                );
                assert_eq!(
                    std::fs::read(&fslx_path).expect("read repaired FSLX"),
                    original
                );
                eprintln!(
                    "{}",
                    serde_json::json!({
                        "schema": "quill-consumer-durability-e2e-v1",
                        "fixture_id": "index-builder-fslx-repair",
                        "source_bytes": original.len(),
                        "corruption": "single-byte-xor",
                        "verify_before": "intact",
                        "verify_corrupt": "repairable",
                        "verify_after": "intact",
                    })
                );
            }
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
            source_count: 6,
            doc_count: 5,
            error_count: 1,
            errors: vec![("bad-doc".into(), "embed failed".into())],
            quality_indexed: 4,
            quality_errors: vec![("slow-doc".into(), "quality timeout".into())],
            lexical: Some(LexicalArmReceipt {
                backend: "quill",
                path: PathBuf::from("/tmp/idx/lexical"),
                attempted: 6,
                indexed: 6,
                errors: Vec::new(),
                generation: None,
                published: true,
            }),
            size_bytes: IndexSizeBreakdown {
                total: 300,
                vector_fast: 100,
                vector_quality: 50,
                lexical: 150,
            },
            total_ms: 42.0,
            embed_ms: 30.0,
            lexical_ms: 5.0,
            has_quality_index: true,
            embedder_availability: TwoTierAvailability::Full,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.source_count, 6);
        assert_eq!(cloned.doc_count, 5);
        assert_eq!(cloned.error_count, 1);
        assert_eq!(cloned.errors.len(), 1);
        assert_eq!(cloned.quality_indexed, 4);
        assert_eq!(cloned.quality_errors.len(), 1);
        assert_eq!(cloned.lexical.as_ref().map(|arm| arm.indexed), Some(6));
        assert_eq!(cloned.size_bytes.total, 300);
        assert!(cloned.has_quality_index);
        let dbg = format!("{stats:?}");
        assert!(dbg.contains("IndexBuildStats"));
        assert!(dbg.contains("LexicalArmReceipt"));
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

    /// Per-arm quality receipts (bd-8nqz.3): a document whose quality
    /// embedding fails stays in the fast tier and is reported per-document,
    /// not silently absorbed into an aggregate.
    #[test]
    fn quality_receipts_track_partial_quality_failures() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let stack = EmbedderStack::from_parts(
                Arc::new(StubEmbedder {
                    id: "stub-fast",
                    dim: 4,
                }),
                Some(Arc::new(SelectiveFailEmbedder)),
            );
            let stats = IndexBuilder::new(dir.path())
                .with_embedder_stack(stack)
                .add_document("doc-clean", "quality succeeds here")
                .add_document("doc-marked", "fail-fast-embedding marker text")
                .build(&cx)
                .await
                .unwrap();

            assert_eq!(stats.source_count, 2);
            assert_eq!(stats.doc_count, 2, "fast tier holds both documents");
            assert_eq!(stats.error_count, 0, "no fast embedding failed");
            assert_eq!(stats.quality_indexed, 1);
            assert_eq!(stats.quality_errors.len(), 1);
            assert_eq!(stats.quality_errors[0].0, "doc-marked");
        });
    }

    /// The preserved gate: when every fast embedding fails, the build still
    /// errors — an index generation without a single vector record is not
    /// representable by `TwoTierIndexBuilder::finish` (empty-generation
    /// support is bd-tqhc index-crate scope, not facade scope).
    #[test]
    fn all_embeddings_failing_still_errors() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let stack = EmbedderStack::from_parts(Arc::new(SelectiveFailEmbedder), None);
            let result = IndexBuilder::new(dir.path())
                .with_embedder_stack(stack)
                .add_document("doc-a", "fail-fast-embedding alpha")
                .add_document("doc-b", "fail-fast-embedding beta")
                .build(&cx)
                .await;

            assert!(result.is_err());
        });
    }

    /// Empty-content documents are valid source documents: they embed (the
    /// embedder decides what an empty text means) and they are admitted to
    /// the lexical arm (which may index zero tokens for them).
    #[test]
    fn empty_content_documents_are_admitted() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let stats = IndexBuilder::new(dir.path())
                .with_embedder_stack(stub_stack())
                .add_document("doc-empty", "")
                .add_document("doc-full", "real content")
                .build(&cx)
                .await
                .unwrap();

            assert_eq!(stats.source_count, 2);
            assert_eq!(stats.doc_count, 2);
            assert_eq!(stats.error_count, 0);
            #[cfg(any(feature = "lexical", feature = "quill"))]
            {
                let receipt = stats.lexical.as_ref().expect("lexical arm receipt");
                assert_eq!(receipt.attempted, 2);
                assert_eq!(
                    receipt.indexed + receipt.errors.len(),
                    receipt.attempted,
                    "every attempted document is accounted for exactly once",
                );
            }
        });
    }

    /// Duplicate IDs: pin that the lexical arm reports per-document errors in
    /// the receipt instead of voiding the whole arm, and that accounting
    /// stays exact.
    #[cfg(feature = "quill")]
    #[test]
    fn duplicate_ids_land_in_lexical_receipt_not_aggregate_failure() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let stats = IndexBuilder::new(dir.path())
                .with_embedder_stack(stub_stack())
                .add_document("doc-dup", "alpha duplicate content")
                .add_document("doc-dup", "beta duplicate content")
                .add_document("doc-ok", "gamma unique content")
                .build(&cx)
                .await
                .unwrap();

            assert_eq!(stats.source_count, 3);
            let receipt = stats.lexical.as_ref().expect("lexical arm receipt");
            assert_eq!(receipt.attempted, 3);
            assert_eq!(receipt.indexed, 2, "first doc-dup and doc-ok both index");
            assert_eq!(receipt.errors.len(), 1);
            assert_eq!(receipt.errors[0].0, "doc-dup");
            assert!(
                receipt.errors[0].1.contains("duplicate live document id"),
                "the receipt carries the typed duplicate rejection: {:?}",
                receipt.errors[0].1,
            );
            assert!(receipt.published);

            // The clean document AFTER the rejected duplicate survived — the
            // reconcile-commit recovery keeps one bad document from voiding
            // the rest of the arm.
            let lexical = QuillIndex::open(&cx, dir.path().join("lexical"), QuillConfig::default())
                .await
                .unwrap();
            let gamma_ids = lexical
                .search_doc_ids(&cx, "gamma", 10)
                .unwrap()
                .into_iter()
                .map(|hit| hit.document_id)
                .collect::<Vec<_>>();
            assert_eq!(gamma_ids, vec!["doc-ok".to_owned()]);
        });
    }

    /// bd-8nqz.2: a tantivy meta.json squatting in the lexical dir must be a
    /// typed refusal, not a silent Quill initialization beside it (MANIFEST
    /// absence is not emptiness).
    #[cfg(feature = "quill")]
    #[test]
    fn build_refuses_quill_init_over_foreign_lexical_layout() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let lexical_dir = dir.path().join("lexical");
            std::fs::create_dir_all(&lexical_dir).unwrap();
            std::fs::write(lexical_dir.join("meta.json"), b"{}").unwrap();

            let error = IndexBuilder::new(dir.path())
                .with_embedder_stack(stub_stack())
                .add_document("doc-1", "content")
                .build(&cx)
                .await
                .expect_err("foreign lexical layout must refuse Quill init");
            let message = error.to_string();
            assert!(
                message.contains("direct-tantivy"),
                "error must carry the typed layout label: {message}"
            );
            assert!(
                !lexical_dir.join("MANIFEST").exists(),
                "no Quill artifacts may appear beside the foreign index"
            );
        });
    }

    /// bd-8nqz.2: `open_hybrid` reports a mixed lexical layout as a typed
    /// error instead of silently picking an engine.
    #[cfg(feature = "quill")]
    #[test]
    fn open_hybrid_reports_mixed_layout_as_typed_error() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            IndexBuilder::new(dir.path())
                .with_embedder_stack(stub_stack())
                .add_document("doc-1", "Alpha content")
                .build(&cx)
                .await
                .unwrap();
            std::fs::write(dir.path().join("lexical").join("meta.json"), b"{}").unwrap();

            let error = open_hybrid(&cx, dir.path(), TwoTierConfig::default())
                .await
                .expect_err("mixed layout must be a typed error");
            assert!(
                error.to_string().contains("mixed"),
                "error must carry the typed layout label: {error}"
            );
        });
    }

    /// bd-8nqz.2: when both backends are compiled, a published Tantivy
    /// generation is a supported rollback target. The root inspector decides
    /// the engine; no Quill open is attempted against Tantivy's `meta.json`.
    #[cfg(all(feature = "quill", feature = "lexical-tantivy"))]
    #[test]
    fn open_lexical_reader_opens_current_tantivy_generation() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let lexical_root = root.path().join("lexical");
            let tantivy_dir = lexical_root.join("tantivy-v1");
            std::fs::create_dir_all(&tantivy_dir).unwrap();

            let tantivy = TantivyIndex::create(&tantivy_dir).unwrap();
            LexicalWrite::index_document(
                &tantivy,
                &cx,
                &IndexableDocument::new("tantivy-doc", "rollback sentinel"),
            )
            .await
            .unwrap();
            LexicalWrite::commit(&tantivy, &cx).await.unwrap();
            // Tantivy's IndexWriter holds an exclusive directory lock, so the
            // rollback generation has to be CLOSED before a reader may open
            // it. Without this the open below fails LockBusy — which is what
            // it did, undetected, from the day this test was written: it is
            // gated on all(quill, lexical-tantivy) and no lane ever enabled
            // both, so `-p frankensearch --lib open_lexical_reader` reported a
            // vacuous "ok. 0 passed" in every single-feature selection
            // (bd-8nqz.2, bd-jt7b2's defect class one layer up).
            drop(tantivy);
            publish_current(
                &lexical_root,
                &CurrentPointer::new(BlueGreenEngine::Tantivy, "tantivy-v1", 0).unwrap(),
            )
            .unwrap();

            let (lexical, backend) = open_lexical_reader(&cx, &lexical_root).await.unwrap();
            assert_eq!(backend, Some(LexicalReaderBackend::TantivyOracle));
            let lexical = lexical.expect("published Tantivy generation must open");
            let hits = lexical.search(&cx, "rollback", 5).await.unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].doc_id, "tantivy-doc");
        });
    }

    /// A fast embedder that cancels the caller context but completes its own
    /// work must still stop the vector-only build at the post-embed checkpoint,
    /// before the second document is touched.
    #[cfg(not(any(feature = "lexical", feature = "quill")))]
    #[test]
    fn vector_only_fast_cancellation_stops_later_documents() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let calls = Arc::new(AtomicUsize::new(0));
            let error = IndexBuilder::new(dir.path())
                .with_embedder_stack(cancel_on_first_stack("fast-canceller", Arc::clone(&calls)))
                .add_document("doc-1", "first")
                .add_document("doc-2", "must not be embedded")
                .build(&cx)
                .await
                .expect_err("fast cancellation must abort the vector-only build");

            assert!(
                matches!(
                    &error,
                    SearchError::Cancelled { phase, reason }
                        if phase == "fast document embedding completion"
                            && reason.contains("cancel-on-first embedder")
                ),
                "caller-context cancellation must stay typed, got {error:?}"
            );
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        });
    }

    /// Quality cancellation must not be downgraded into a fast-only success
    /// receipt. It aborts the vector-only build before the next document.
    #[cfg(not(any(feature = "lexical", feature = "quill")))]
    #[test]
    fn vector_only_quality_cancellation_is_not_degraded() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let calls = Arc::new(AtomicUsize::new(0));
            let error = IndexBuilder::new(dir.path())
                .with_embedder_stack(cancel_on_first_stack(
                    "quality-canceller",
                    Arc::clone(&calls),
                ))
                .add_document("doc-1", "first")
                .add_document("doc-2", "must not be embedded")
                .build(&cx)
                .await
                .expect_err("quality cancellation must abort the vector-only build");

            assert!(
                matches!(
                    &error,
                    SearchError::Cancelled { phase, reason }
                        if phase == "quality-canceller embedding"
                            && reason == "cancel-on-first embedder"
                ),
                "typed quality cancellation must survive unchanged, got {error:?}"
            );
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        });
    }

    /// bd-8nqz.3 cancellation matrix: a cancelled `Cx` aborts the build with
    /// the typed cancellation error — never a per-document receipt entry,
    /// never recovery-machinery churn — and a cleared `Cx` builds clean.
    #[cfg(feature = "quill")]
    #[test]
    fn build_rejects_cancelled_cx_with_typed_error() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            cx.set_cancel_requested(true);
            let error = IndexBuilder::new(dir.path())
                .with_embedder_stack(stub_stack())
                .add_document("doc-1", "content")
                .build(&cx)
                .await
                .expect_err("cancelled cx must reject the build");
            assert!(
                matches!(error, SearchError::Cancelled { .. }),
                "typed cancellation must survive to the caller, got {error:?}"
            );

            // Retry-clean: a cleared cx builds successfully from scratch.
            cx.set_cancel_requested(false);
            let fresh = tempfile::tempdir().unwrap();
            let stats = IndexBuilder::new(fresh.path())
                .with_embedder_stack(stub_stack())
                .add_document("doc-1", "content")
                .build(&cx)
                .await
                .expect("cleared cx must build clean");
            assert_eq!(stats.doc_count, 1);
        });
    }

    /// `open_hybrid` (bd-8nqz.3): the ergonomic opener must attach the
    /// lexical arm the advertised examples previously dropped, and the
    /// attached reader must actually answer through the trait object.
    #[cfg(any(feature = "lexical", feature = "quill"))]
    #[test]
    fn open_hybrid_attaches_lexical_arm() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            IndexBuilder::new(dir.path())
                .with_embedder_stack(stub_stack())
                .add_document("doc-1", "Alpha retrieval content")
                .add_document("doc-2", "Beta ranking content")
                .build(&cx)
                .await
                .unwrap();

            let parts = open_hybrid(&cx, dir.path(), TwoTierConfig::default())
                .await
                .unwrap();
            assert_eq!(parts.vectors.doc_count(), 2);
            #[cfg(feature = "quill")]
            assert_eq!(parts.lexical_backend, Some(LexicalReaderBackend::Quill));
            let lexical = parts.lexical.expect("lexical arm must be attached");
            let hits = lexical.search(&cx, "Alpha", 5).await.unwrap();
            assert!(!hits.is_empty(), "trait-object search must answer");
        });
    }

    /// Without a lexical backend compiled in, `open_hybrid` still opens the
    /// vector arms and reports the lexical arm as absent rather than erroring.
    #[cfg(not(any(feature = "lexical", feature = "quill")))]
    #[test]
    fn open_hybrid_without_lexical_backend_reports_absent_arm() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            IndexBuilder::new(dir.path())
                .with_embedder_stack(stub_stack())
                .add_document("doc-1", "Alpha retrieval content")
                .build(&cx)
                .await
                .unwrap();

            let parts = open_hybrid(&cx, dir.path(), TwoTierConfig::default())
                .await
                .unwrap();
            assert_eq!(parts.vectors.doc_count(), 1);
            assert!(parts.lexical.is_none());
            assert!(parts.lexical_backend.is_none());
        });
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn admitted_v2_sync_opener_constructs_the_shipping_residual_cache_product() {
        use frankensearch_core::generation::{
            ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1, QuantizationFormat,
        };
        use frankensearch_core::{BoundQueryEmbedding, TieredQueryEmbeddings};

        let dir = owned_admitted_v2_sync_dir();
        let source_path = dir.join("fast.fsvi");
        let cache_dir = dir.join("residual-cache");
        std::fs::create_dir(&cache_dir).expect("create a fresh owned cache directory");

        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("facade-route", 2);
        "fsvi-v2".clone_into(&mut identity.storage.format);
        identity.storage.quantization = QuantizationFormat::F16;
        "little-endian".clone_into(&mut identity.storage.endianness);
        let binding = FsviV2IdentityBinding::new(
            ArtifactGenerationIdentityV1::new(91, [0x6d; 16]).expect("create a test generation"),
            identity.freeze().expect("freeze test identity"),
        )
        .expect("create a valid v2 binding");
        let mut writer = frankensearch_index::VectorIndex::create_v2(&source_path, binding.clone())
            .expect("create an admitted-v2 fixture");
        writer
            .write_record("exact-winner", &[0.0, 1.0])
            .expect("write winner");
        writer
            .write_record("other", &[1.0, 0.0])
            .expect("write other row");
        writer.finish().expect("seal fixture");

        let paths = TwoTierIndexPaths::new(&source_path);
        #[cfg(feature = "ann")]
        let discarded_ann_path = dir.join("discarded-by-sync-product.hnsw");
        #[cfg(feature = "ann")]
        let paths = paths.with_fast_ann(&discarded_ann_path);
        let searcher = open_admitted_v2_sync_with_residual_sidecar_cache(
            &paths,
            &binding,
            None,
            &cache_dir,
            TwoTierConfig {
                fast_only: true,
                // Force the ordinary two-tier opener's ANN threshold. The
                // synchronous residual facade must still not build or persist
                // this configured sidecar because it discards that container.
                hnsw_threshold: 0,
                ..TwoTierConfig::default()
            },
        )
        .expect("the facade opener constructs the shipping sync product");
        let query = TieredQueryEmbeddings::fast_only(
            BoundQueryEmbedding::new(
                vec![0.0, 1.0],
                EmbeddingIdentityBundleV1::explicit_test_model("facade-route", 2),
            )
            .expect("bind v2 query identity"),
        );
        let (results, _) = searcher
            .search_collect(&query, 1)
            .expect("search through the facade product opener");
        assert_eq!(results[0].doc_id, "exact-winner");
        assert_eq!(
            std::fs::read_dir(&cache_dir)
                .expect("inspect owned cache directory")
                .flatten()
                .count(),
            1,
            "the default facade opener must route through residual-cache publication"
        );
        #[cfg(feature = "ann")]
        assert!(
            !discarded_ann_path.exists(),
            "the owner-only synchronous facade must strip ANN paths instead of persisting an \
             artifact its in-memory product discards"
        );
    }
}
