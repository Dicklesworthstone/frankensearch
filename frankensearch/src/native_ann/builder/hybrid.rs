//! Native build-to-query integration with a privately owned Quill lexical arm.

use std::sync::Arc;

use frankensearch_quill::{QuillConfig, QuillIndex, QuillSearchIndex};

use super::super::{checkpoint, invalid};
use super::{NativeBuiltIndex, NativeIndexBuilder};
use crate::native_ann::NativeProgressiveSearch;
use crate::{Cx, LexicalRead, LexicalWrite, Reranker, ScoredResult, SearchResult};

mod snapshot;
use snapshot::LexicalSeal;
pub use snapshot::NativeHybridReopenLimits;

impl NativeIndexBuilder {
    /// Build native fast/quality vectors and Quill from the same owned documents.
    ///
    /// All arms are required and must finish before any hybrid handle is returned.
    /// Quill uses deterministic single-shard bulk ingest and is finalized once.
    /// Before returning, open a read-only Quill publication while the private
    /// writer still holds its lease, then release that writer. Serving never
    /// retains a writer lease or refreshes lexical independently of the vectors.
    ///
    /// This consumes the prepared content exactly as [`Self::build`] does and
    /// adds native lexical indexing of the complete original documents. It does
    /// not adopt an existing lexical directory or accept an unrelated reader.
    /// The source cohort is retained for final cross-encoder input. No CURRENT
    /// pointer or durable composite-generation manifest is published.
    ///
    /// # Errors
    /// Propagates vector build, lexical create/index/finalize, membership and
    /// cancellation errors. Failures may leave an unselected directory but never
    /// produce a partial hybrid success or overwrite an older generation.
    pub async fn build_hybrid(self, cx: &Cx) -> SearchResult<NativeBuiltHybridIndex> {
        let vectors = Box::pin(self.build(cx)).await?;
        checkpoint(cx, "native_ann.builder.lexical_start")?;
        let path = vectors.directory.join("lexical");
        std::fs::create_dir(&path)?;
        let response = Box::pin(QuillIndex::create(
            cx,
            &path,
            QuillConfig {
                bulk_load_mode: true,
                deterministic_ingest: true,
                max_ingest_shards: 1,
                ..QuillConfig::default()
            },
        ))
        .await;
        checkpoint(cx, "native_ann.builder.lexical_created")?;
        let lexical = response?;
        for document in vectors.documents.iter() {
            checkpoint(cx, "native_ann.builder.lexical_document")?;
            let response = LexicalWrite::index_document(&lexical, cx, document).await;
            checkpoint(cx, "native_ann.builder.lexical_document_complete")?;
            response?;
        }
        let response = Box::pin(lexical.finish_bulk_load(cx)).await;
        checkpoint(cx, "native_ann.builder.lexical_finalized")?;
        response?;
        if LexicalRead::doc_count(&lexical)? != vectors.documents.len() {
            return Err(invalid(
                "builder.lexical_membership",
                "cardinality",
                "Quill must contain the complete source cohort",
            ));
        }
        // The writer remains alive until the reader has admitted its sealed
        // publication. Another writer cannot publish between finalize and open.
        // Capture bytes here, NOT at a later seal: a newer publication on disk
        // must never be blessed as belonging to these retained source vectors.
        let seal = LexicalSeal::capture(cx, &path, lexical.search_snapshot()?.keeper_generation())?;
        let response = Box::pin(QuillSearchIndex::open(cx, &path, QuillConfig::default())).await;
        checkpoint(cx, "native_ann.builder.lexical_reader")?;
        let reader = response?;
        let built = NativeBuiltHybridIndex::from_readers(cx, vectors, reader, seal)?;
        drop(lexical);
        Ok(built)
    }
}

type SourceTextLookup = dyn Fn(&str) -> Option<String> + Send + Sync;

/// Complete process-local native hybrid generation built from one source cohort.
///
/// Vector tiers retain their producing models; Quill is read-only and never
/// exposed for independent refresh. No writer lease is retained after building.
/// Every progressive phase borrows this same object. Final reranking resolves
/// text from its owned source documents, not a closure over mutable/current files.
/// This enforces build-time cohort ownership, not a durable cross-process
/// publication protocol or an attestation for externally supplied artifacts.
/// [`Self::seal_for_reopen`] selects this exact lexical/source/vector cohort for
/// [`Self::open_selected`]. Keep its directory trusted and immutable throughout
/// sealing, reopening and the lifetime of Quill's mapped readers. An independent
/// writer's append-only publication does not refresh an already-retained view.
pub struct NativeBuiltHybridIndex {
    vectors: NativeBuiltIndex,
    lexical: QuillSearchIndex,
    lexical_seal: LexicalSeal,
    text: Box<SourceTextLookup>,
}

impl NativeBuiltHybridIndex {
    // Both callers are private, completed build/reopen paths. Do not expose an
    // arbitrary lexical+vector constructor that could bypass cohort admission.
    fn from_readers(
        cx: &Cx,
        vectors: NativeBuiltIndex,
        lexical: QuillSearchIndex,
        lexical_seal: LexicalSeal,
    ) -> SearchResult<Self> {
        if LexicalRead::doc_count(&lexical)? != vectors.documents.len()
            || lexical.keeper_generation() != lexical_seal.generation()
        {
            return Err(invalid(
                "builder.lexical_membership",
                "cardinality",
                "the sealed lexical reader must contain the complete source cohort",
            ));
        }
        let source = Arc::clone(&vectors.documents);
        let text = Box::new(move |id: &str| {
            source
                .binary_search_by(|document| document.id.as_str().cmp(id))
                .ok()
                .map(|position| source[position].content.clone())
        });
        checkpoint(cx, "native_ann.builder.hybrid_complete")?;
        Ok(Self {
            vectors,
            lexical,
            lexical_seal,
            text,
        })
    }

    /// Read-only native tiers and exact source documents used for this build.
    #[must_use]
    pub const fn vectors(&self) -> &NativeBuiltIndex {
        &self.vectors
    }

    /// Read-only lexical access. No writer, commit or replacement API is exposed.
    #[must_use]
    pub fn lexical(&self) -> &dyn LexicalRead {
        &self.lexical
    }

    /// Fast-plus-lexical search with the producing model and pinned hydration.
    ///
    /// # Errors
    /// Propagates native hybrid admission, inference, lexical and cancellation errors.
    pub async fn search(&self, cx: &Cx, text: &str, k: usize) -> SearchResult<Vec<ScoredResult>> {
        let fast = self.vectors.fast();
        fast.index()
            .search_hybrid_text(cx, fast.embedder(), &self.lexical, text, k)
            .await
    }

    /// Independent fast and quality retrieval, blended with lexical candidates.
    /// A generation built without quality performs its declared fast-only search;
    /// a configured quality failure never triggers an inference fallback here.
    ///
    /// # Errors
    /// Propagates all configured retrieval and hydration errors.
    pub async fn search_refined(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        let fast = self.vectors.fast();
        match self.vectors.quality() {
            Some(quality) => {
                fast.index()
                    .search_hybrid_refined_text(
                        cx,
                        fast.embedder(),
                        (quality.index(), quality.embedder()),
                        &self.lexical,
                        text,
                        k,
                    )
                    .await
            }
            None => self.search(cx, text, k).await,
        }
    }

    /// Primary quality-plus-lexical retrieval without running the fast model.
    ///
    /// # Errors
    /// Refuses a missing quality tier; otherwise propagates native quality errors.
    pub async fn search_quality(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        checkpoint(cx, "native_ann.builder.quality_query")?;
        let quality = self.vectors.quality().ok_or_else(|| {
            invalid(
                "builder.quality",
                "absent",
                "quality-primary search requires a built quality tier",
            )
        })?;
        quality
            .index()
            .search_hybrid_quality_text(cx, quality.embedder(), &self.lexical, text, k)
            .await
    }

    /// Prepare the existing lazy native phase sequence without provider work.
    ///
    /// # Errors
    /// Refuses invalid query topology, identity, candidate budget or cancellation.
    pub fn progressive<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
        k: usize,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        let fast = self.vectors.fast();
        fast.index().search_hybrid_progressive(
            cx,
            fast.embedder(),
            self.vectors
                .quality()
                .map(|quality| (quality.index(), quality.embedder())),
            &self.lexical,
            text,
            k,
        )
    }

    /// Prepare lazy native retrieval and cross-encoder reranking using the EXACT
    /// source content from this build. The caller supplies only the reranker,
    /// never a "latest text" resolver that could silently change document versions.
    /// `window` may exceed the displayed `k` so undisplayed candidates can win.
    /// Text resolution and inference begin only when the final phase is requested.
    ///
    /// # Errors
    /// Combines [`Self::progressive`] admission and native rerank configuration errors.
    pub fn progressive_with_reranker<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
        k: usize,
        reranker: &'a dyn Reranker,
        window: usize,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        self.progressive(cx, text, k)?
            .with_reranker(reranker, self.text.as_ref(), window)
    }
}

#[cfg(test)]
mod tests;
