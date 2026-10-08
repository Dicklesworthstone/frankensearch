//! Atomic mixed document mutations over an explicitly selected complete generation.
//!
//! This is a document API, not a filesystem discovery policy. IDs are opaque and
//! bodies are supplied by the caller. One independently copied successor receives
//! all lexical, vector, catalog and membership changes before one pointer switch.
//! Untouched documents are never canonicalized or embedded by this operation.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use asupersync::Cx;
use frankensearch_core::{
    Canonicalizer, DefaultCanonicalizer, Embedder, EmbeddingIdentityBundleV1, SearchError,
    SearchResult,
};

use super::{
    FSFS_INDEX_MANIFEST_FILE_NAME, FSFS_LEXICAL_MANIFEST_FILE, FSFS_VECTOR_INDEX_FILE,
    FSFS_VECTOR_QUALITY_INDEX_FILE, FsfsRuntime, IndexManifestEntry, IngestionClass,
    LEXICAL_CANONICALIZER, LexicalMutation, PipelineStorageConfig, SearchExecutionMode, Storage,
    VectorIndex, index_source_hash_hex, ingestion_class_label, pressure_timestamp_ms,
    protect_vector_generations, reclaim_unpublished_lexical_garbage, retained_reuse,
    retained_search_checkpoint, semantic_windows, validate_retained_catalog_path,
};
use crate::generation_store::{
    CompleteGenerationStore, GenerationPublication, PublishedGeneration,
};
use crate::lifecycle::PublicationLease;

mod inference;
mod inherited_reuse;
use inference::InferenceBatch;

const MAX_OPERATIONS: usize = 4096;
const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;
const MAX_INPUT_BYTES: usize = 64 * 1024 * 1024;
const MAX_VECTOR_BYTES: usize = 128 * 1024 * 1024;

/// One ordered document mutation. The last operation for an ID wins.
///
/// IDs are not read as paths. Every operation is validated, including superseded
/// operations. Upserts replace the complete document body, just like append-batch;
/// this API does not merge properties or infer deletions from missing source files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetainedMutation {
    /// Insert or replace a document, using the ordinary embedding/lexical input policies.
    Upsert { id: String, text: String },
    /// Remove an exact document ID from every present tier. Never a prefix match.
    Delete { id: String },
}

/// Outcome of an atomic mixed batch. Counts refer to final, distinct IDs.
#[derive(Debug)]
#[must_use]
pub struct RetainedBatchResult {
    /// None for empty input or a batch consisting only of absent deletions.
    /// Inspect the publication variant before asserting crash durability.
    pub publication: Option<GenerationPublication>,
    /// Final upserts, including replacements of already present documents.
    pub upserted: usize,
    /// Final deletions that matched the admitted predecessor's membership.
    pub deleted: usize,
}

#[derive(Debug)]
struct Body {
    embedding: String,
    lexical: String,
    source: Option<SourceAttributes>,
}

type PreparedBatch = BTreeMap<String, Option<Body>>;

/// Supplied only by a source owner after normal discovery/classification. The
/// public document API never infers filesystem authority from an opaque ID.
#[derive(Debug, Clone)]
pub(super) struct SourceAttributes {
    pub modified_ms: u64,
    pub ingestion_class: IngestionClass,
    pub title: Option<String>,
    pub metadata: HashMap<String, String>,
}

/// Prepared source input owns only the changed documents. Preparation performs
/// no filesystem or model work and never changes publication state. A source
/// owner may attach additional evidence that must still hold after the common
/// post-seal precommit check, before the pointer is changed.
pub(super) struct SourceBatch {
    documents: PreparedBatch,
    source_check: Option<SourceCheck>,
}

type SourceCheck = Box<dyn FnOnce(&Cx) -> SearchResult<()> + Send>;

impl SourceBatch {
    /// Keep source-only authority out of the public opaque-document API. Checks
    /// compose instead of replacing previously attached evidence, and all stay
    /// owned by this batch; dropping a build never runs them independently.
    #[must_use]
    pub(super) fn with_source_check<F>(mut self, check: F) -> Self
    where
        F: FnOnce(&Cx) -> SearchResult<()> + Send + 'static,
    {
        let previous = self.source_check.take();
        self.source_check = Some(Box::new(move |cx| {
            if let Some(previous) = previous {
                previous(cx)?;
            }
            retained_search_checkpoint(cx)?;
            check(cx)
        }));
        self
    }
}

pub(super) fn prepare_source_batch(
    cx: &Cx,
    operations: &[RetainedMutation],
    sources: &BTreeMap<String, SourceAttributes>,
) -> SearchResult<Option<SourceBatch>> {
    let mut documents = match prepare(cx, operations) {
        Ok(documents) => documents,
        // A source requiring the full indexer's skip/large-document handling
        // is not a partially admitted delta. Cancellation always propagates.
        Err(SearchError::InvalidConfig { .. }) => return Ok(None),
        Err(error) => return Err(error),
    };
    attach_sources(cx, &mut documents, sources)?;
    Ok(Some(SourceBatch {
        documents,
        source_check: None,
    }))
}

impl Body {
    fn class(&self, default: IngestionClass) -> IngestionClass {
        self.source
            .as_ref()
            .map_or(default, |source| source.ingestion_class)
    }

    fn semantic(&self) -> bool {
        self.class(IngestionClass::FullSemanticLexical) == IngestionClass::FullSemanticLexical
    }

    fn revision(&self, default: i64) -> i64 {
        // Source timestamps are checked by attach_sources before any I/O.
        self.source.as_ref().map_or(default, |source| {
            i64::try_from(source.modified_ms).unwrap_or(i64::MAX)
        })
    }

    fn reason(&self, default: IngestionClass) -> &'static str {
        if self.source.is_some() {
            super::ingestion_plan_reason(self.class(default))
        } else {
            "retained_batch"
        }
    }
}

fn attach_sources(
    cx: &Cx,
    documents: &mut PreparedBatch,
    sources: &BTreeMap<String, SourceAttributes>,
) -> SearchResult<()> {
    if sources.len() != documents.values().filter(|body| body.is_some()).count() {
        return Err(invalid(
            "source attributes must cover exactly the final upserts",
        ));
    }
    let mut bytes = 0_usize;
    for (id, body) in documents {
        retained_search_checkpoint(cx)?;
        bytes = bytes
            .checked_add(id.len())
            .filter(|bytes| *bytes <= MAX_INPUT_BYTES)
            .ok_or_else(|| invalid("source payload length overflow"))?;
        let Some(body) = body else { continue };
        let source = sources
            .get(id)
            .ok_or_else(|| invalid("a source upsert is missing its attributes"))?;
        if i64::try_from(source.modified_ms).is_err()
            || !matches!(
                source.ingestion_class,
                IngestionClass::FullSemanticLexical | IngestionClass::LexicalOnly
            )
        {
            return Err(invalid(
                "source attributes require a valid timestamp and indexed text class",
            ));
        }
        bytes = bytes
            .checked_add(body.embedding.len())
            .and_then(|bytes| bytes.checked_add(body.lexical.len()))
            .and_then(|bytes| bytes.checked_add(source.title.as_ref().map_or(0, String::len)))
            .ok_or_else(|| invalid("source payload length overflow"))?;
        for (key, value) in &source.metadata {
            bytes = bytes
                .checked_add(key.len())
                .and_then(|bytes| bytes.checked_add(value.len()))
                .ok_or_else(|| invalid("source metadata length overflow"))?;
        }
        if bytes > MAX_INPUT_BYTES || source.metadata.len() > 64 {
            return Err(invalid(
                "source payload or metadata exceeds the retained batch bound",
            ));
        }
        body.source = Some(source.clone());
    }
    retained_search_checkpoint(cx)
}

fn invalid(reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: "complete_generation.batch".to_owned(),
        value: String::new(),
        reason: reason.to_owned(),
    }
}

fn prepare(cx: &Cx, operations: &[RetainedMutation]) -> SearchResult<PreparedBatch> {
    retained_search_checkpoint(cx)?;
    if operations.len() > MAX_OPERATIONS {
        return Err(invalid("a retained batch accepts at most 4096 operations"));
    }
    let mut input_bytes = 0_usize;
    let mut final_bodies = BTreeMap::new();
    for operation in operations {
        retained_search_checkpoint(cx)?;
        let (id, text) = match operation {
            RetainedMutation::Upsert { id, text } => (id, Some(text)),
            RetainedMutation::Delete { id } => (id, None),
        };
        if id.trim().is_empty() || id.contains('\0') || id.len() > usize::from(u16::MAX) {
            return Err(invalid(
                "IDs must be nonblank, NUL-free and at most 65535 bytes",
            ));
        }
        if text.is_some_and(|text| text.len() > MAX_BODY_BYTES) {
            return Err(invalid("a retained document body exceeds 8 MiB"));
        }
        input_bytes = input_bytes
            .checked_add(id.len())
            .and_then(|bytes| bytes.checked_add(text.map_or(0, String::len)))
            .filter(|bytes| *bytes <= MAX_INPUT_BYTES)
            .ok_or_else(|| invalid("retained batch input exceeds 64 MiB"))?;
        final_bodies.insert(id.as_str(), text);
    }
    let canonicalizer = DefaultCanonicalizer::default();
    let mut canonical_bytes = 0_usize;
    let mut prepared = BTreeMap::new();
    for (id, text) in final_bodies {
        retained_search_checkpoint(cx)?;
        let body = if let Some(text) = text {
            let embedding = canonicalizer.canonicalize(text);
            retained_search_checkpoint(cx)?;
            let lexical = LEXICAL_CANONICALIZER.canonicalize(text);
            retained_search_checkpoint(cx)?;
            if embedding.trim().is_empty() || lexical.trim().is_empty() {
                return Err(invalid("a final upsert has empty canonical text"));
            }
            if embedding.len() > MAX_BODY_BYTES || lexical.len() > MAX_BODY_BYTES {
                return Err(invalid("a canonical document body exceeds 8 MiB"));
            }
            canonical_bytes = canonical_bytes
                .checked_add(embedding.len())
                .and_then(|bytes| bytes.checked_add(lexical.len()))
                .ok_or_else(|| invalid("canonical batch length overflow"))?;
            Some(Body {
                embedding,
                lexical,
                source: None,
            })
        } else {
            None
        };
        canonical_bytes = canonical_bytes
            .checked_add(id.len())
            .filter(|bytes| *bytes <= MAX_INPUT_BYTES)
            .ok_or_else(|| invalid("canonical retained payload exceeds 64 MiB"))?;
        prepared.insert(id.to_owned(), body);
    }
    retained_search_checkpoint(cx)?;
    Ok(prepared)
}

struct Producer {
    embedder: Arc<dyn Embedder>,
    identity: EmbeddingIdentityBundleV1,
    role: &'static str,
}

impl Producer {
    fn admit(runtime: &FsfsRuntime, root: &Path, quality: bool) -> SearchResult<Self> {
        let (embedder, relative, role) = if quality {
            (
                runtime.resolve_quality_embedder()?.ok_or_else(|| {
                    invalid("the existing quality tier requires its verified producer")
                })?,
                FSFS_VECTOR_QUALITY_INDEX_FILE,
                "fsfs.retained_batch.quality",
            )
        } else {
            (
                runtime.resolve_fast_embedder()?,
                FSFS_VECTOR_INDEX_FILE,
                "fsfs.retained_batch.fast",
            )
        };
        let index = VectorIndex::open_read_only(&root.join(relative))?;
        if quality {
            FsfsRuntime::admit_quality_generation_for_embedder(&index, embedder.as_ref())?;
        } else {
            semantic_windows::load_mapping(root, &index)?;
            FsfsRuntime::admit_vector_generation_for_embedder(&index, embedder.as_ref())?;
        }
        let identity = embedder.identity()?.clone();
        identity.validate()?;
        if index.embedder_revision() != identity.fingerprint() {
            return Err(SearchError::UnverifiableRemoteSpace {
                producer: role.to_owned(),
                reason: "producer changed during retained-batch admission".to_owned(),
            });
        }
        Ok(Self {
            embedder,
            identity,
            role,
        })
    }

    fn recheck(&self) -> SearchResult<()> {
        if !self
            .embedder
            .identity()
            .is_ok_and(|actual| actual == &self.identity)
            || usize::try_from(self.identity.space.dimension).ok()
                != Some(self.embedder.dimension())
        {
            return Err(SearchError::UnverifiableRemoteSpace {
                producer: self.role.to_owned(),
                reason: "producer changed after retained-batch admission".to_owned(),
            });
        }
        Ok(())
    }

    async fn infer(&self, cx: &Cx, text: &str) -> SearchResult<Vec<f32>> {
        retained_search_checkpoint(cx)?;
        let admission = self.recheck();
        retained_search_checkpoint(cx)?;
        admission?;
        // Preserve bound-single overrides. No raw inference or weaker retry can
        // attach an advertised identity to an otherwise unverified response.
        let result = self.embedder.embed_bound(cx, text).await;
        retained_search_checkpoint(cx)?;
        let result = match result {
            Err(error @ SearchError::Cancelled { .. }) => return Err(error),
            result => result,
        };
        let admission = self.recheck();
        retained_search_checkpoint(cx)?;
        admission?;
        let response = result?;
        // Check foreign contracts before validation can quote provider-owned fields.
        if response.identity != self.identity {
            return Err(SearchError::UnverifiableRemoteSpace {
                producer: self.role.to_owned(),
                reason: "response does not carry the admitted producer identity".to_owned(),
            });
        }
        response.validate()?;
        Ok(response.values)
    }
}

fn charge_vector(bytes: &mut usize, values: &[f32]) -> SearchResult<()> {
    *bytes = values
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|length| bytes.checked_add(length))
        .filter(|total| *total <= MAX_VECTOR_BYTES)
        .ok_or_else(|| invalid("retained batch vector payload exceeds 128 MiB"))?;
    Ok(())
}

impl FsfsRuntime {
    /// Apply a mixed upsert/delete batch with one complete-generation publication.
    ///
    /// `expected` is the caller's admitted base receipt. A different selected
    /// generation is a conflict, never permission to overwrite another writer.
    /// Final operations are last-write-wins per exact ID. Only final upserts are
    /// canonicalized/embedded; unchanged documents and their metadata survive in
    /// an independent copy. All present vector tiers must remain compatible.
    /// Existing source-reuse evidence is retained only for untouched documents,
    /// under its original executable/configuration scope. Changed documents lose
    /// their old reuse proof even when their resulting metadata happens to match.
    ///
    /// Empty input performs no filesystem or model work. A nonempty no-op still
    /// checks its base. Input (4096 operations, 8 MiB per body, 64 MiB total) and
    /// staged vectors (128 MiB) are bounded, not a process-wide RSS guarantee.
    /// Each tier uses native bound inference groups of at most 64 rows and
    /// 512 KiB of text when its provider explicitly supports them. A single
    /// larger input keeps the existing document limit. Other providers retain
    /// custom bound-single dispatch; no failure triggers raw inference or retry.
    /// This does not scan source paths, download models, prune generations, or
    /// alter watcher discovery policy. Synchronous filesystem work retains the
    /// ordinary fsfs caller-owned blocking-lane contract.
    ///
    /// # Errors
    /// Invalid input, changed base, producer/response mismatch, inference failure,
    /// cancellation or pre-publication I/O leaves the old selection intact.
    /// Post-rename sync failure is an explicit visible-but-uncertain publication,
    /// not an aborted batch. No partial success or per-document retry is returned.
    #[allow(clippy::future_not_send)]
    pub async fn apply_retained_batch(
        &self,
        cx: &Cx,
        store_root: &Path,
        expected: &PublishedGeneration,
        operations: &[RetainedMutation],
    ) -> SearchResult<RetainedBatchResult> {
        self.apply_retained_batch_with_precommit(cx, store_root, expected, operations, |_| Ok(()))
            .await
    }

    #[allow(clippy::future_not_send)]
    pub(super) async fn apply_retained_batch_with_precommit<F>(
        &self,
        cx: &Cx,
        store_root: &Path,
        expected: &PublishedGeneration,
        operations: &[RetainedMutation],
        precommit: F,
    ) -> SearchResult<RetainedBatchResult>
    where
        F: FnOnce(&Cx) -> SearchResult<()> + Send,
    {
        let documents = prepare(cx, operations)?;
        self.apply_prepared_retained_batch(cx, store_root, expected, documents, precommit)
            .await
    }

    /// Private source-aware route. The source owner retains discovery, decoding
    /// and the final precommit check; mutation still uses the same atomic writer.
    #[allow(clippy::future_not_send)]
    pub(super) async fn apply_retained_source_batch_with_precommit<F>(
        &self,
        cx: &Cx,
        store_root: &Path,
        expected: &PublishedGeneration,
        batch: SourceBatch,
        precommit: F,
    ) -> SearchResult<RetainedBatchResult>
    where
        F: FnOnce(&Cx) -> SearchResult<()> + Send,
    {
        let SourceBatch {
            documents,
            source_check,
        } = batch;
        self.apply_prepared_retained_batch(cx, store_root, expected, documents, move |cx| {
            // Discovery/backend/selection admission remains first. Additional
            // source evidence cannot override a failure or acknowledge a source
            // recreated during that admission, even when discovery hides it.
            precommit(cx)?;
            retained_search_checkpoint(cx)?;
            if let Some(check) = source_check {
                check(cx)?;
            }
            retained_search_checkpoint(cx)
        })
        .await
    }

    #[allow(clippy::future_not_send, clippy::too_many_lines)]
    async fn apply_prepared_retained_batch<F>(
        &self,
        cx: &Cx,
        store_root: &Path,
        expected: &PublishedGeneration,
        mut documents: PreparedBatch,
        precommit: F,
    ) -> SearchResult<RetainedBatchResult>
    where
        F: FnOnce(&Cx) -> SearchResult<()> + Send,
    {
        retained_search_checkpoint(cx)?;
        if documents.is_empty() {
            return Ok(RetainedBatchResult {
                publication: None,
                upserted: 0,
                deleted: 0,
            });
        }
        validate_retained_catalog_path(&self.config.storage.db_path)?;
        let store = CompleteGenerationStore::open(cx, store_root)?;
        let build = store.begin(cx)?;
        let predecessor = store
            .active(cx)?
            .ok_or_else(|| invalid("no generation is selected"))?;
        if &predecessor != expected {
            return Err(invalid(
                "selected generation differs from the expected batch base",
            ));
        }
        Self::validate_search_generation_at_root(predecessor.path(), SearchExecutionMode::Full)?;
        let mut manifests = Self::read_matching_manifest_generation(predecessor.path())?
            .ok_or_else(|| invalid("selected membership manifests disagree"))?;
        documents.retain(|id, body| body.is_some() || manifests.contains_key(id));
        let upserted = documents.values().filter(|body| body.is_some()).count();
        let deleted = documents.len() - upserted;
        if documents.is_empty() {
            retained_search_checkpoint(cx)?;
            return Ok(RetainedBatchResult {
                publication: None,
                upserted,
                deleted,
            });
        }
        let lexical_only = Self::is_lexical_only_generation(predecessor.path());
        if lexical_only
            && documents
                .values()
                .flatten()
                .any(|body| body.source.is_some() && body.semantic())
        {
            return Err(invalid(
                "source policy requires vectors absent from the selected generation",
            ));
        }
        let semantic_upserts = documents.values().flatten().any(Body::semantic);
        let window_maximum = if lexical_only {
            1
        } else {
            Self::fast_window_policy_at_root(predecessor.path())?
        };
        let fast = if lexical_only || !semantic_upserts {
            None
        } else {
            Some(Producer::admit(self, predecessor.path(), false)?)
        };
        let quality = if semantic_upserts
            && predecessor
                .path()
                .join(FSFS_VECTOR_QUALITY_INDEX_FILE)
                .try_exists()?
        {
            Some(Producer::admit(self, predecessor.path(), true)?)
        } else {
            None
        };
        let mut windows = BTreeMap::new();
        // Finish window planning before retaining any borrowed input slices.
        // Groups may span documents, but the source and its plans remain owned
        // until both independent tiers have finished inference.
        if fast.is_some() && window_maximum > 1 {
            for (id, body) in &documents {
                retained_search_checkpoint(cx)?;
                if let Some(body) = body
                    && body.semantic()
                {
                    windows.insert(
                        id.clone(),
                        semantic_windows::plan(&body.lexical, window_maximum)?,
                    );
                }
            }
        }
        let mut fast_batch = fast.as_ref().map(InferenceBatch::new);
        let mut quality_batch = quality.as_ref().map(InferenceBatch::new);
        let mut vector_bytes = 0;
        for (id, body) in &documents {
            retained_search_checkpoint(cx)?;
            let Some(body) = body else { continue };
            if !body.semantic() {
                continue;
            }
            if let Some(batch) = &mut fast_batch {
                let texts = if window_maximum > 1 {
                    windows
                        .get(id)
                        .ok_or_else(|| invalid("missing prepared window plan"))?
                        .texts(&body.lexical)?
                } else {
                    vec![body.embedding.as_str()]
                };
                for (ordinal, text) in texts.into_iter().enumerate() {
                    let row = semantic_windows::row_id(id, ordinal);
                    if row.len() > usize::from(u16::MAX) {
                        return Err(invalid(
                            "document ID leaves insufficient room for window IDs",
                        ));
                    }
                    batch.push(cx, row, text, &mut vector_bytes).await?;
                }
            }
            if let Some(batch) = &mut quality_batch {
                batch
                    .push(cx, id.clone(), &body.embedding, &mut vector_bytes)
                    .await?;
            }
        }
        let fast_entries = match fast_batch {
            Some(batch) => batch.finish(cx, &mut vector_bytes).await?,
            None => Vec::new(),
        };
        let quality_entries = match quality_batch {
            Some(batch) => batch.finish(cx, &mut vector_bytes).await?,
            None => Vec::new(),
        };
        retained_search_checkpoint(cx)?;
        let mut input = self.cli_input.clone();
        input.index_dir = Some(build.path().to_path_buf());
        input.daemon = false;
        input.daemon_socket = None;
        let candidate = self.clone().with_cli_input(input);
        if retained_reuse::copy_selected_generation(cx, &candidate, &store, build.path())?
            != predecessor
        {
            return Err(invalid(
                "selected predecessor changed while copying the batch base",
            ));
        }
        let lease = PublicationLease::acquire(build.path())?;
        let timestamp = pressure_timestamp_ms();
        let revision = manifests
            .values()
            .map(|entry| entry.revision)
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| invalid("document revision exhausted"))?
            .max(i64::try_from(timestamp).map_err(|_| invalid("document timestamp overflow"))?);
        let class = if lexical_only {
            IngestionClass::LexicalOnly
        } else {
            IngestionClass::FullSemanticLexical
        };
        let mutations = documents
            .iter()
            .map(|(id, body)| {
                body.as_ref().map_or_else(
                    || {
                        LexicalMutation::delete(
                            id.clone(),
                            timestamp,
                            IngestionClass::Skip,
                            "retained_batch",
                        )
                    },
                    |body| {
                        let modified = body
                            .source
                            .as_ref()
                            .map_or(timestamp, |source| source.modified_ms);
                        let mut mutation = LexicalMutation::upsert(
                            id.clone(),
                            modified,
                            body.class(class),
                            body.lexical.clone(),
                            body.reason(class),
                        );
                        if let Some(source) = &body.source {
                            mutation.title.clone_from(&source.title);
                            mutation.metadata.clone_from(&source.metadata);
                        }
                        mutation
                    },
                )
            })
            .collect::<Vec<_>>();
        lease.fence("retained batch lexical mutation")?;
        candidate
            .apply_one_shot_lexical_mutations(cx, build.path(), &mutations)
            .await?;
        for (relative, producer, entries) in [
            (FSFS_VECTOR_INDEX_FILE, fast.as_ref(), &fast_entries),
            (
                FSFS_VECTOR_QUALITY_INDEX_FILE,
                quality.as_ref(),
                &quality_entries,
            ),
        ] {
            retained_search_checkpoint(cx)?;
            let path = build.path().join(relative);
            if !path.try_exists()? {
                continue;
            }
            lease.fence("retained batch vector mutation")?;
            let mut index = Self::open_vector_index_for_mutation(&path)?;
            if let Some(producer) = producer {
                producer.recheck()?;
                if relative == FSFS_VECTOR_INDEX_FILE {
                    Self::admit_vector_generation_for_embedder(&index, producer.embedder.as_ref())?;
                } else {
                    Self::admit_quality_generation_for_embedder(
                        &index,
                        producer.embedder.as_ref(),
                    )?;
                }
                index.append_batch(entries)?;
            }
            let new_rows = entries
                .iter()
                .map(|(id, _)| id.as_str())
                .collect::<HashSet<_>>();
            let retired = index
                .live_doc_ids()?
                .into_iter()
                .filter(|row| {
                    let source = if relative == FSFS_VECTOR_INDEX_FILE {
                        semantic_windows::source_id(row)
                    } else {
                        row.as_str()
                    };
                    documents.contains_key(source) && !new_rows.contains(row.as_str())
                })
                .collect::<Vec<_>>();
            index.soft_delete_batch(&retired.iter().map(String::as_str).collect::<Vec<_>>())?;
            index.compact()?;
            index.vacuum()?;
        }
        retained_search_checkpoint(cx)?;
        lease.fence("retained batch catalog mutation")?;
        let catalog_path = candidate.resolve_storage_db_path()?;
        if catalog_path.try_exists()? {
            let storage = Storage::open(PipelineStorageConfig {
                db_path: catalog_path,
                ..PipelineStorageConfig::default()
            })?;
            for (id, body) in &documents {
                retained_search_checkpoint(cx)?;
                if let Some(body) = body {
                    let revision = body.revision(revision);
                    let created = storage
                        .get_document(id)?
                        .map_or(revision, |old| old.created_at);
                    storage.upsert_document(&frankensearch_storage::DocumentRecord::new(
                        id,
                        body.embedding.chars().take(400).collect::<String>(),
                        frankensearch_storage::ContentHasher::hash(&body.embedding),
                        body.embedding.chars().count(),
                        created,
                        revision.max(created),
                    ))?;
                    if body.semantic() {
                        for producer in [fast.as_ref(), quality.as_ref()].into_iter().flatten() {
                            storage.mark_embedded(id, producer.embedder.id())?;
                        }
                    }
                } else {
                    storage.delete_document(id)?;
                }
            }
        }
        let changed_ids = documents.keys().cloned().collect::<HashSet<_>>();
        for (id, body) in documents {
            if let Some(body) = body {
                let entry = IndexManifestEntry {
                    file_key: id.clone(),
                    revision: body.revision(revision),
                    ingestion_class: ingestion_class_label(body.class(class)).to_owned(),
                    canonical_bytes: u64::try_from(body.embedding.len())
                        .map_err(|_| invalid("canonical document length overflow"))?,
                    reason_code: body.reason(class).to_owned(),
                    fast_windows: windows.remove(&id),
                };
                manifests.insert(id, entry);
            } else {
                manifests.remove(&id);
            }
        }
        let manifests = manifests.into_values().collect::<Vec<_>>();
        let layout = Self::resolve_lexical_engine(build.path())?;
        let lexical_manifest = if layout.lexical_root() == build.path() {
            layout
                .engine_dir()
                .map(|path| path.join(FSFS_INDEX_MANIFEST_FILE_NAME))
                .unwrap_or_else(|| build.path().join(FSFS_LEXICAL_MANIFEST_FILE))
        } else {
            build.path().join(FSFS_LEXICAL_MANIFEST_FILE)
        };
        candidate.write_index_artifacts(build.path(), &lexical_manifest, &manifests)?;
        let mut sentinel = Self::read_index_sentinel(build.path())?
            .ok_or_else(|| invalid("candidate completion sentinel is absent"))?;
        "retained-batch".clone_into(&mut sentinel.command);
        sentinel.generated_at_ms = timestamp;
        sentinel.indexed_files = manifests.len();
        sentinel.discovered_files = sentinel.discovered_files.max(manifests.len());
        sentinel.skipped_files = sentinel.discovered_files.saturating_sub(manifests.len());
        sentinel.total_canonical_bytes = manifests.iter().fold(0_u64, |sum, entry| {
            sum.saturating_add(entry.canonical_bytes)
        });
        sentinel.source_hash_hex = index_source_hash_hex(&manifests);
        for reason in protect_vector_generations(build.path(), "retained batch") {
            if !sentinel.reason_codes.contains(&reason) {
                sentinel.reason_codes.push(reason);
            }
        }
        candidate.write_index_sentinel(build.path(), &sentinel)?;
        lease.fence("retained batch candidate complete")?;
        drop(lease);
        reclaim_unpublished_lexical_garbage(cx, build.path()).await?;
        let resources = Box::pin(
            candidate.prepare_search_execution_resources_at_root_with_modes(
                cx,
                build.path(),
                SearchExecutionMode::Full,
                SearchExecutionMode::Full,
            ),
        )
        .await?;
        drop(resources);
        inherited_reuse::retain_unchanged(cx, &predecessor, build.path(), &changed_ids)?;
        for producer in [fast.as_ref(), quality.as_ref()].into_iter().flatten() {
            producer.recheck()?;
        }
        retained_search_checkpoint(cx)?;
        let publication = build.publish_with_precommit(
            cx,
            |_, path| Self::validate_search_generation_at_root(path, SearchExecutionMode::Full),
            precommit,
        )?;
        Ok(RetainedBatchResult {
            publication: Some(publication),
            upserted,
            deleted,
        })
    }
}

#[cfg(all(test, unix))]
mod tests;

#[cfg(all(test, unix))]
mod source_tests;
