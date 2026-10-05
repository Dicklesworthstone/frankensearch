//! One global Quill population joined to a complete source/vector shard inventory.
//! No per-shard BM25 normalization or second fusion engine is introduced here.

use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use frankensearch_core::generation::{ArtifactGenerationIdentityV1, GenerationComponentReceiptV1};
use frankensearch_quill::{QuillConfig, QuillSearchIndex};
use serde::{Deserialize, Serialize};

use super::{NativeBuiltShardedIndex, NativeShardedReopenLimits};
use crate::native_ann::builder::NativeIndexBuilder;
use crate::native_ann::builder::hybrid::snapshot::{LexicalSeal, NativeHybridReopenLimits};
use crate::native_ann::builder::hybrid::{cohort, create_lexical};
use crate::native_ann::builder::snapshot::{
    Artifact, checked_directory, create_private_new, ensure_absent, read_selected,
    require_seal_platform, sync_directory,
};
use crate::native_ann::{NativeShardedProgressiveSearch, NativeShardedResult, checkpoint, invalid};
use crate::{Cx, Embedder, IndexableDocument, LexicalRead, Reranker, SearchError, SearchResult};

const HYBRID_FILE: &str = "native.sharded-hybrid.json";
const HYBRID_SCHEMA: &str = "frankensearch.native-sharded-hybrid.v1";
const MAX_DESCRIPTOR_BYTES: u64 = 32 * 1024 * 1024;

type SourceTextLookup = dyn Fn(&str) -> Option<String> + Send + Sync;

/// Independent total vector/source and lexical input ceilings for hybrid reopen.
/// These do not bound decoded-object overhead, model memory or peak RSS.
#[derive(Debug, Clone, Copy)]
pub struct NativeShardedHybridReopenLimits {
    /// Complete ordered vector inventory's aggregate and per-partition limits.
    pub vectors: NativeShardedReopenLimits,
    /// Maximum files in the one global lexical inventory.
    pub max_lexical_files: usize,
    /// Maximum bytes in any single lexical artifact.
    pub max_lexical_file_bytes: u64,
    /// Maximum total bytes across all selected lexical files.
    pub max_lexical_bytes: u64,
}

impl Default for NativeShardedHybridReopenLimits {
    fn default() -> Self {
        let lexical = NativeHybridReopenLimits::default();
        Self {
            vectors: NativeShardedReopenLimits::default(),
            max_lexical_files: lexical.max_lexical_files,
            max_lexical_file_bytes: lexical.max_lexical_file_bytes,
            max_lexical_bytes: lexical.max_lexical_bytes,
        }
    }
}

impl NativeShardedHybridReopenLimits {
    fn lexical(self) -> NativeHybridReopenLimits {
        NativeHybridReopenLimits {
            vectors: self.vectors.partition,
            max_lexical_files: self.max_lexical_files,
            max_lexical_file_bytes: self.max_lexical_file_bytes,
            max_lexical_bytes: self.max_lexical_bytes,
        }
    }

    fn validate(self) -> SearchResult<()> {
        self.vectors.validate()?;
        self.lexical().validate()
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    generation: ArtifactGenerationIdentityV1,
    documents: usize,
    vectors: Artifact,
    lexical: LexicalSeal,
}

/// Complete read-only sharded hybrid cohort, including the exact retained source.
///
/// A single Quill population preserves global term statistics and keyword ranks.
/// The existing sharded engine embeds once per tier, merges vector candidates
/// globally, independently retrieves quality, and fuses with this lexical view.
/// Physical rows are returned with their originating tier and partition; a bare
/// `ScoredResult.index` is never used to misidentify a cross-shard row.
///
/// Construction and selected reopen compare every lexical document's original
/// fields to the retained source. Count equality or individually valid files are
/// not enough. No public arbitrary-reader constructor bypasses that census.
/// Keep all directories trusted and immutable, including Quill's mapped files
/// for the lifetime of old readers. This is not a durable selector, rollback
/// floor, distributed index, or a request to run destructive retention.
pub struct NativeBuiltShardedHybridIndex {
    vectors: NativeBuiltShardedIndex,
    lexical: QuillSearchIndex,
    lexical_seal: LexicalSeal,
    text: Box<SourceTextLookup>,
}

impl NativeIndexBuilder {
    /// Build complete vector partitions and one global lexical index over them.
    ///
    /// Sources move into partitions; lexical construction iterates references to
    /// the original bodies. Existing native and Quill builders do all engine
    /// work. The lexical writer remains alive until full source admission ends,
    /// and is dropped before returning the read-only cohort. A failed lexical
    /// stage never returns vector-only success. No files are deleted on failure.
    ///
    /// # Errors
    /// Includes every partition, required-tier, source, lexical and cancellation
    /// failure. Existing roots are refused, including sealed ancestors.
    pub async fn build_sharded_hybrid(
        self,
        cx: &Cx,
        max_documents_per_shard: usize,
    ) -> SearchResult<NativeBuiltShardedHybridIndex> {
        let vectors = Box::pin(self.build_sharded(cx, max_documents_per_shard)).await?;
        let path = vectors.directory().join("lexical");
        let (writer, reader, seal) = create_lexical(cx, &path, vectors.documents()).await?;
        let built = NativeBuiltShardedHybridIndex::from_readers(cx, vectors, reader, seal)?;
        drop(writer);
        Ok(built)
    }
}

impl NativeBuiltShardedHybridIndex {
    fn from_readers(
        cx: &Cx,
        vectors: NativeBuiltShardedIndex,
        lexical: QuillSearchIndex,
        lexical_seal: LexicalSeal,
    ) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.sharded_hybrid.admission")?;
        if usize::try_from(lexical.doc_count()?).ok() != Some(vectors.document_count())
            || lexical.keeper_generation() != lexical_seal.generation()
        {
            return Err(rejected(
                "lexical_membership",
                "global lexical population differs from its source cohort",
            ));
        }
        // One pointer per source, not cloned bodies/metadata/vector images.
        let documents = vectors.documents().collect::<Vec<_>>();
        cohort::validate(cx, &lexical, &documents)?;
        drop(documents);
        let sources: Vec<Arc<[IndexableDocument]>> = vectors
            .partitions
            .iter()
            .map(|partition| Arc::clone(&partition.documents))
            .collect();
        let text = Box::new(move |id: &str| {
            let ordinal = sources.partition_point(|documents| {
                documents.last().is_some_and(|last| last.id.as_str() < id)
            });
            let documents = sources.get(ordinal)?;
            let position = documents
                .binary_search_by(|document| document.id.as_str().cmp(id))
                .ok()?;
            Some(documents[position].content.clone())
        });
        checkpoint(cx, "native_ann.sharded_hybrid.complete")?;
        Ok(Self {
            vectors,
            lexical,
            lexical_seal,
            text,
        })
    }

    /// Original ordered partitions, sources and independent tier inventories.
    #[must_use]
    pub const fn vectors(&self) -> &NativeBuiltShardedIndex {
        &self.vectors
    }

    /// One unrefreshable global keyword reader, not per-partition BM25 scores.
    #[must_use]
    pub fn lexical(&self) -> &dyn LexicalRead {
        &self.lexical
    }

    /// Fast and lexical retrieval with one query embedding across all partitions.
    ///
    /// # Errors
    /// Propagates existing sharded hybrid identity, query and hydration failures.
    pub async fn search(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<NativeShardedResult>> {
        self.vectors
            .fast
            .search_hybrid_text(
                cx,
                self.vectors.partitions[0].fast.embedder(),
                &self.lexical,
                text,
                k,
            )
            .await
    }

    /// Independent quality retrieval, not rescoring only the fast candidates.
    /// A declared fast-only cohort executes fast search; required-tier errors
    /// never silently remove quality or omit a failing partition.
    ///
    /// # Errors
    /// Propagates all configured retrieval, fusion and hydration failures.
    pub async fn search_refined(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<NativeShardedResult>> {
        let first = &self.vectors.partitions[0];
        match (&self.vectors.quality, &first.quality) {
            (Some(shards), Some(tier)) => {
                self.vectors
                    .fast
                    .search_hybrid_refined_text(
                        cx,
                        first.fast.embedder(),
                        (shards, tier.embedder()),
                        &self.lexical,
                        text,
                        k,
                    )
                    .await
            }
            (None, None) => self.search(cx, text, k).await,
            _ => Err(rejected(
                "quality",
                "quality inventory and provider must travel together",
            )),
        }
    }

    /// Quality-primary plus lexical retrieval without invoking the fast provider.
    ///
    /// # Errors
    /// Refuses absent quality, including zero-result queries, and propagates errors.
    pub async fn search_quality(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<NativeShardedResult>> {
        checkpoint(cx, "native_ann.sharded_hybrid.quality")?;
        let shards = self
            .vectors
            .quality
            .as_ref()
            .ok_or_else(|| rejected("quality", "quality tier was not configured"))?;
        let tier = self.vectors.partitions[0]
            .quality
            .as_ref()
            .ok_or_else(|| rejected("quality", "quality provider was not configured"))?;
        shards
            .search_hybrid_quality_text(cx, tier.embedder(), &self.lexical, text, k)
            .await
    }

    /// Prepare the existing lazy sharded phase sequence without provider work.
    ///
    /// # Errors
    /// Refuses invalid identity, topology, candidate budgets or cancellation.
    pub fn progressive<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
        k: usize,
    ) -> SearchResult<NativeShardedProgressiveSearch<'a>> {
        let first = &self.vectors.partitions[0];
        let quality = self
            .vectors
            .quality
            .as_ref()
            .zip(first.quality.as_ref())
            .map(|(shards, tier)| (shards, tier.embedder()));
        self.vectors.fast.search_hybrid_progressive(
            cx,
            first.fast.embedder(),
            quality,
            &self.lexical,
            text,
            k,
        )
    }

    /// Add lazy cross-encoder ordering of source text from this exact cohort.
    /// The window can exceed displayed k. Source and per-tier shard rows remain
    /// tied to this handle throughout every phase, including scorer failure.
    ///
    /// # Errors
    /// Combines progressive admission and native reranker configuration errors.
    pub fn progressive_with_reranker<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
        k: usize,
        reranker: &'a dyn Reranker,
        window: usize,
    ) -> SearchResult<NativeShardedProgressiveSearch<'a>> {
        self.progressive(cx, text, k)?
            .with_reranker(reranker, self.text.as_ref(), window)
    }

    /// Seal the global lexical view and the COMPLETE ordered vector inventory.
    ///
    /// Child receipts plus the global lexical inventory are authenticated by one
    /// returned receipt. Preserve it in the caller's trusted catalog; directory
    /// existence does not select a generation. Lexical bytes were captured during
    /// construction, never recaptured from a later independent publication.
    /// Cancellation is checked before the final bounded descriptor write; an
    /// error after writing can leave an inert partial seal, never a claimed abort
    /// of another publisher. Existing seals cannot be overwritten or retried.
    ///
    /// # Errors
    /// Refuses changed artifacts, preexisting seals, unsupported platforms,
    /// cancellation, encoding, I/O and durability failures.
    pub fn seal_for_reopen(&self, cx: &Cx) -> SearchResult<GenerationComponentReceiptV1> {
        checkpoint(cx, "native_ann.sharded_hybrid.seal")?;
        require_seal_platform()?;
        let directory = checked_directory(self.vectors.directory())?;
        let path = directory.join(HYBRID_FILE);
        ensure_absent(&path)?;
        let lexical_path = checked_directory(&directory.join("lexical"))?;
        self.lexical_seal
            .verify(cx, &lexical_path, NativeHybridReopenLimits::default(), true)?;
        let vectors = self.vectors.seal_for_reopen(cx)?;
        self.lexical_seal.verify(
            cx,
            &lexical_path,
            NativeHybridReopenLimits::default(),
            false,
        )?;
        sync_directory(&lexical_path)?;
        let manifest = Manifest {
            schema: HYBRID_SCHEMA.to_owned(),
            generation: self.vectors.generation(),
            documents: self.vectors.document_count(),
            vectors: Artifact {
                byte_len: vectors.byte_len,
                sha256: vectors.sha256,
            },
            lexical: self.lexical_seal.clone(),
        };
        let bytes = serde_json::to_vec(&manifest).map_err(|_| {
            rejected(
                "encoding",
                "cannot encode complete sharded hybrid descriptor",
            )
        })?;
        if bytes.len() as u64 > MAX_DESCRIPTOR_BYTES {
            return Err(rejected(
                "descriptor_size",
                "hybrid descriptor exceeds its size bound",
            ));
        }
        checkpoint(cx, "native_ann.sharded_hybrid.seal_commit")?;
        let mut file = create_private_new(&path)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        sync_directory(&directory)?;
        if let Some(parent) = directory.parent() {
            sync_directory(parent)?;
        }
        Ok(Artifact::from_bytes(&bytes).receipt())
    }

    /// Reopen the exact selected shards and global lexical population together.
    ///
    /// Models are caller-supplied and must match their original producer identity.
    /// The root receipt authenticates the ordered child receipts and exact lexical
    /// inventory. Every required artifact and stored source field must pass normal
    /// admission. No partial cohort, lexical-only success or ANN fallback is
    /// returned. Reopen runs no inference and neither discovers nor repairs files.
    ///
    /// Vector/source and lexical totals have separate explicit budgets. Original
    /// child and Quill decoder limits remain enforced. This is synchronous file
    /// and graph work plus async lexical opening on the caller's lane, not a
    /// hidden runtime, bounded-latency promise or hostile-directory authority.
    ///
    /// # Errors
    /// Propagates any selection, input-budget, identity, source/lexical mismatch,
    /// artifact integrity, I/O or cancellation error without mutating old readers.
    pub async fn open_selected(
        cx: &Cx,
        directory: impl AsRef<Path>,
        expected: &GenerationComponentReceiptV1,
        fast: Arc<dyn Embedder>,
        quality: Option<Arc<dyn Embedder>>,
        limits: NativeShardedHybridReopenLimits,
    ) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.sharded_hybrid.open")?;
        limits.validate()?;
        let directory = checked_directory(directory.as_ref())?;
        let bytes = read_selected(
            cx,
            &directory.join(HYBRID_FILE),
            Artifact {
                byte_len: expected.byte_len,
                sha256: expected.sha256,
            },
            MAX_DESCRIPTOR_BYTES,
        )?;
        let saved: Manifest = serde_json::from_slice(&bytes)
            .map_err(|_| rejected("schema", "malformed sharded hybrid descriptor"))?;
        if saved.schema != HYBRID_SCHEMA || saved.documents > limits.vectors.max_documents {
            return Err(rejected(
                "schema",
                "invalid sharded hybrid schema or source count",
            ));
        }
        saved.generation.validate()?;
        saved.vectors.validate()?;
        saved.lexical.validate(limits.lexical())?;
        let lexical_path = checked_directory(&directory.join("lexical"))?;
        saved
            .lexical
            .verify(cx, &lexical_path, limits.lexical(), false)?;
        let vectors = NativeBuiltShardedIndex::open_selected(
            cx,
            &directory,
            &saved.vectors.receipt(),
            fast,
            quality,
            limits.vectors,
        )?;
        if vectors.generation() != saved.generation || vectors.document_count() != saved.documents {
            return Err(rejected(
                "membership",
                "vector inventory differs from selected hybrid generation",
            ));
        }
        let response = Box::pin(QuillSearchIndex::open(
            cx,
            &lexical_path,
            QuillConfig::default(),
        ))
        .await;
        checkpoint(cx, "native_ann.sharded_hybrid.lexical_opened")?;
        let lexical = response?;
        saved
            .lexical
            .verify(cx, &lexical_path, limits.lexical(), false)?;
        Self::from_readers(cx, vectors, lexical, saved.lexical)
    }
}

fn rejected(field: &str, reason: &str) -> SearchError {
    invalid(&format!("sharded_hybrid.{field}"), "rejected", reason)
}

#[cfg(test)]
mod tests;
